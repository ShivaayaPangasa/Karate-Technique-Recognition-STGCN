"""
Train and evaluate the ST-GCN karate technique classifier.

Primary experiment:
- Input: 100 x 39 x 3 XYZ marker coordinates
- 39 anatomical markers
- 3 spatial channels: X, Y, Z
- Subject-independent train/validation/test split
- Seed: 42
- No velocity channels in the primary experiment
- Metrics are calculated without scikit-learn
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.graph import build_adjacency
from src.models.stgcn_model import STGCN

# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = "data/processed_index.csv"
OUTPUT_DIR = Path("outputs")

SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, num_classes):
    """
    Calculate:
    - Accuracy
    - Macro-F1
    - Macro-precision
    - Macro-recall

    This avoids the scikit-learn dependency, which is useful on
    Windows systems where sklearn's compiled extensions may be blocked.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Accuracy
    accuracy = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1_scores = []

    for cls in range(num_classes):

        true_positive = np.sum(
            (y_true == cls) & (y_pred == cls)
        )

        false_positive = np.sum(
            (y_true != cls) & (y_pred == cls)
        )

        false_negative = np.sum(
            (y_true == cls) & (y_pred != cls)
        )

        # Precision
        if (true_positive + false_positive) > 0:
            precision = (
                true_positive
                / (true_positive + false_positive)
            )
        else:
            precision = 0.0

        # Recall
        if (true_positive + false_negative) > 0:
            recall = (
                true_positive
                / (true_positive + false_negative)
            )
        else:
            recall = 0.0

        # F1
        if (precision + recall) > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
    }


# ============================================================
# DATASET
# ============================================================

class GraphKarateDataset(Dataset):
    """
    Dataset used for ST-GCN training.

    Each sample is expected to have shape:

        (100, 39, 3)

    where:
        100 = temporal frames
        39  = anatomical markers
        3   = XYZ coordinates
    """

    def __init__(
        self,
        split,
        index_path=INDEX_PATH,
        augment_data=False,
    ):

        with open(
            index_path,
            newline="",
            encoding="utf-8"
        ) as f:

            rows = list(csv.DictReader(f))

        self.rows = [
            row for row in rows
            if row["split"] == split
        ]

        # Use the complete dataset to establish the same
        # label-to-index mapping for every split.
        self.labels = sorted(
            set(row["label"] for row in rows)
        )

        self.label_to_idx = {
            label: index
            for index, label in enumerate(self.labels)
        }

        self.augment_data = augment_data

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        row = self.rows[index]

        xyz = np.load(row["npy_path"]).astype(
            np.float32
        )

        # Expected:
        # (100, 39, 3)
        if xyz.shape != (100, 39, 3):
            raise ValueError(
                f"Unexpected input shape {xyz.shape} "
                f"for {row['npy_path']}. "
                f"Expected (100, 39, 3)."
            )

        # Training augmentation:
        # small rotation around vertical axis + Gaussian noise.
        if self.augment_data:

            angle = np.random.uniform(
                -15,
                15
            ) * np.pi / 180.0

            c = np.cos(angle)
            s = np.sin(angle)

            rotation = np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            xyz = xyz @ rotation.T

            noise = np.random.normal(
                0,
                0.01,
                xyz.shape
            ).astype(np.float32)

            xyz = xyz + noise

        label = self.label_to_idx[row["label"]]

        return (
            torch.from_numpy(xyz).float(),
            label,
        )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        # Guard against numerical instability.
        if not torch.isfinite(loss):
            print(
                "Warning: non-finite loss encountered. "
                "Skipping batch."
            )
            continue

        loss.backward()

        # Gradient clipping for stability.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += (
            loss.item() * x.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)

    if total == 0:
        return float("nan"), 0.0

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    num_classes,
):

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_loss += (
            loss.item() * x.size(0)
        )

        total_samples += x.size(0)

        all_labels.extend(
            y.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        num_classes,
    )

    if total_samples > 0:
        metrics["loss"] = (
            total_loss / total_samples
        )
    else:
        metrics["loss"] = float("nan")

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ST-GCN Karate Technique Recognition")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print("Input shape: 100 x 39 x 3")
    print("Channels: 3 (X, Y, Z)")
    print("Velocity channels: disabled")
    print()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = GraphKarateDataset(
        split="train",
        augment_data=True,
    )

    val_dataset = GraphKarateDataset(
        split="val",
        augment_data=False,
    )

    test_dataset = GraphKarateDataset(
        split="test",
        augment_data=False,
    )

    print(
        f"Train samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print(
        f"Classes: {train_dataset.labels}"
    )

    print(
        f"Number of classes: "
        f"{len(train_dataset.labels)}"
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    
    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    # build_adjacency() returns a NumPy array.
    # STGCN converts it to a PyTorch tensor internally.
    A = build_adjacency()

    print()
    print(
        f"Graph shape: {A.shape}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = STGCN(
        A,
        in_channels=3,
        num_classes=len(train_dataset.labels),
    ).to(DEVICE)

    print()
    print("Model:")
    print(model)

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_val_f1 = -1.0
    best_epoch = 0

    best_checkpoint = (
        OUTPUT_DIR / "stgcn_best.pt"
    )

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            len(train_dataset.labels),
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        # Save the model with the best validation Macro-F1.
        if (
            np.isfinite(val_metrics["macro_f1"])
            and val_metrics["macro_f1"] > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "labels": train_dataset.labels,
                    "input_shape": (
                        100,
                        39,
                        3,
                    ),
                    "in_channels": 3,
                    "seed": SEED,
                },
                best_checkpoint,
            )

            print(
                f"  -> Saved best model "
                f"(Val Macro-F1: "
                f"{best_val_f1:.4f})"
            )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print()
    print(
        f"Loading best checkpoint from epoch "
        f"{best_epoch}..."
    )

    checkpoint = torch.load(
        best_checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    test_metrics = evaluate(
        model,
        test_loader,
        len(train_dataset.labels),
    )

    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro-F1 : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Loss     : "
        f"{test_metrics['loss']:.4f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # SAVE FINAL METRICS
    # --------------------------------------------------------

    import json

    metrics_path = (
        OUTPUT_DIR / "stgcn_metrics.json"
    )

    results = {
        "model": "ST-GCN",
        "input_shape": [
            100,
            39,
            3,
        ],
        "in_channels": 3,
        "velocity_channels": False,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "labels": train_dataset.labels,
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved checkpoint: "
        f"{best_checkpoint}"
    )

    print(
        f"Saved metrics: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()