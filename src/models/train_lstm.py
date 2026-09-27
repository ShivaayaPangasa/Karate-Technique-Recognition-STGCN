"""
Train and evaluate the LSTM sequence baseline
for karate technique recognition.

Primary research comparison:

    LSTM:
        Sequence-only temporal modelling

    ST-GCN:
        Explicit human-body topology + temporal modelling

Experimental protocol:

    Dataset:
        Optical motion capture karate dataset

    Input:
        100 frames × 39 markers × 3 XYZ coordinates

    LSTM representation:
        100 × 117

    Classes:
        S01-S05 karate techniques

    Split:
        Subject-independent train/validation/test

    Seed:
        42

    Primary metric:
        Macro-F1

    Model selection:
        Best validation Macro-F1

    Test:
        Evaluated only after selecting best checkpoint
"""

# ============================================================
# IMPORTS
# ============================================================

import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow imports from src/models when running:
#
# python src/models/train_lstm.py

sys.path.append(
    str(Path(__file__).parent)
)

from dataset import KarateDataset
from lstm_model import LSTMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = "data/processed_index.csv"

OUTPUT_DIR = Path("outputs")

SEED = 42

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-4

INPUT_SIZE = 117

NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    num_classes,
):
    """
    Calculate:

        Accuracy
        Macro-F1
        Macro-Precision
        Macro-Recall

    Implemented without scikit-learn so that the
    LSTM and ST-GCN use the same metric implementation.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )

    precisions = []

    recalls = []

    f1_scores = []

    # --------------------------------------------------------
    # PER-CLASS METRICS
    # --------------------------------------------------------

    for cls in range(
        num_classes
    ):

        true_positive = np.sum(
            (y_true == cls)
            &
            (y_pred == cls)
        )

        false_positive = np.sum(
            (y_true != cls)
            &
            (y_pred == cls)
        )

        false_negative = np.sum(
            (y_true == cls)
            &
            (y_pred != cls)
        )

        # Precision
        if (
            true_positive
            + false_positive
        ) > 0:

            precision = (
                true_positive
                /
                (
                    true_positive
                    + false_positive
                )
            )

        else:

            precision = 0.0

        # Recall
        if (
            true_positive
            + false_negative
        ) > 0:

            recall = (
                true_positive
                /
                (
                    true_positive
                    + false_negative
                )
            )

        else:

            recall = 0.0

        # F1
        if (
            precision + recall
        ) > 0:

            f1 = (
                2.0
                * precision
                * recall
                /
                (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )

    return {
        "accuracy": accuracy,

        "macro_f1": float(
            np.mean(f1_scores)
        ),

        "precision": float(
            np.mean(precisions)
        ),

        "recall": float(
            np.mean(recalls)
        ),
    }


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

    for x, y in loader:

        x = x.to(
            DEVICE
        )

        y = y.to(
            DEVICE
        )

        # ----------------------------------------------------
        # CONVERT
        #
        # B × 100 × 39 × 3
        #
        # TO
        #
        # B × 100 × 117
        # ----------------------------------------------------

        x = x.reshape(
            x.shape[0],
            x.shape[1],
            -1,
        )

        logits = model(
            x
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.extend(
            y.cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

    return calculate_metrics(
        all_labels,
        all_predictions,
        num_classes,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "LSTM Karate Technique Recognition"
    )

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        "Input representation: "
        "100 × 39 × 3 XYZ"
    )

    print(
        "LSTM input representation: "
        "100 × 117"
    )

    print(
        "Velocity channels: disabled"
    )

    print(
        "Training augmentation: enabled"
    )

    print(
        "Validation augmentation: disabled"
    )

    print(
        "Test augmentation: disabled"
    )

    print()


    # ========================================================
    # DATASETS
    # ========================================================

    train_ds = KarateDataset(
        split="train",
        index_path=INDEX_PATH,
        augment_data=True,
    )

    val_ds = KarateDataset(
        split="val",
        index_path=INDEX_PATH,
        augment_data=False,
    )

    test_ds = KarateDataset(
        split="test",
        index_path=INDEX_PATH,
        augment_data=False,
    )


    # ========================================================
    # DATASET VALIDATION
    # ========================================================

    print(
        f"Train samples: "
        f"{len(train_ds)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_ds)}"
    )

    print(
        f"Test samples: "
        f"{len(test_ds)}"
    )

    print(
        f"Classes: "
        f"{train_ds.labels}"
    )

    print(
        f"Number of classes: "
        f"{len(train_ds.labels)}"
    )


    # Make sure the experiment really has 5 classes.

    if len(train_ds.labels) != NUM_CLASSES:

        raise ValueError(
            "Expected exactly 5 technique "
            f"classes, but found "
            f"{len(train_ds.labels)}: "
            f"{train_ds.labels}"
        )


    # Make sure all splits contain the same
    # global label mapping.

    if train_ds.labels != val_ds.labels:

        raise ValueError(
            "Train and validation label "
            "mappings do not match."
        )

    if train_ds.labels != test_ds.labels:

        raise ValueError(
            "Train and test label "
            "mappings do not match."
        )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        num_classes=len(
            train_ds.labels
        ),
    ).to(
        DEVICE
    )


    print()

    print("Model:")

    print(model)


    # ========================================================
    # LOSS
    # ========================================================

    # Same unweighted CrossEntropyLoss formulation
    # used by the primary ST-GCN experiment.

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # ========================================================
    # LEARNING-RATE SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
    )


    # ========================================================
    # TRAINING
    # ========================================================

    best_val_f1 = -1.0

    best_epoch = 0

    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        total_samples = 0

        skipped_batches = 0


        # ----------------------------------------------------
        # TRAINING BATCHES
        # ----------------------------------------------------

        for x, y in train_loader:

            x = x.to(
                DEVICE
            )

            y = y.to(
                DEVICE
            )


            # ------------------------------------------------
            # B × 100 × 39 × 3
            #
            # ->
            #
            # B × 100 × 117
            # ------------------------------------------------

            x = x.reshape(
                x.shape[0],
                x.shape[1],
                -1,
            )


            optimizer.zero_grad()


            logits = model(
                x
            )


            loss = criterion(
                logits,
                y
            )


            # ------------------------------------------------
            # NUMERICAL STABILITY CHECK
            # ------------------------------------------------

            if not torch.isfinite(
                loss
            ):

                print(
                    "Warning: non-finite "
                    "loss encountered. "
                    "Skipping batch."
                )

                skipped_batches += 1

                continue


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            loss.backward()


            # Gradient clipping

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )


            optimizer.step()


            # ------------------------------------------------
            # LOSS ACCOUNTING
            # ------------------------------------------------

            total_loss += (
                loss.item()
                * x.size(0)
            )

            total_samples += (
                x.size(0)
            )


        # ----------------------------------------------------
        # LEARNING RATE UPDATE
        # ----------------------------------------------------

        scheduler.step()


        # ----------------------------------------------------
        # EPOCH LOSS
        # ----------------------------------------------------

        if total_samples > 0:

            epoch_loss = (
                total_loss
                /
                total_samples
            )

        else:

            epoch_loss = float(
                "nan"
            )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            len(train_ds.labels),
        )


        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if (
            np.isfinite(
                val_metrics["macro_f1"]
            )
            and
            val_metrics["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            # Clone to CPU so the saved state
            # does not depend on GPU memory.

            best_state = {
                key: value.detach()
                .cpu()
                .clone()

                for key, value
                in model.state_dict().items()
            }


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} | "
                f"Train Loss: "
                f"{epoch_loss:.4f} | "
                f"Val Acc: "
                f"{val_metrics['accuracy']:.4f} | "
                f"Val Macro-F1: "
                f"{val_metrics['macro_f1']:.4f} | "
                f"LR: "
                f"{current_lr:.6f}"
            )


    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    print()

    print(
        f"Best validation Macro-F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Skipped batches: "
        f"{skipped_batches}"
    )


    if best_state is None:

        raise RuntimeError(
            "No valid best model checkpoint "
            "was produced."
        )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_state
    )

    model.to(
        DEVICE
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    print()

    print("=" * 70)

    print(
        "FINAL TEST RESULTS "
        "(best validation checkpoint)"
    )

    print("=" * 70)


    test_metrics = evaluate(
        model,
        test_loader,
        len(train_ds.labels),
    )


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


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = (
        OUTPUT_DIR
        /
        "lstm_best.pt"
    )


    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "best_epoch":
                best_epoch,

            "best_validation_macro_f1":
                best_val_f1,

            "labels":
                train_ds.labels,

            "input_shape":
                [
                    100,
                    39,
                    3,
                ],

            "lstm_input_shape":
                [
                    100,
                    117,
                ],

            "input_size":
                INPUT_SIZE,

            "seed":
                SEED,

            "test_metrics":
                test_metrics,
        },
        checkpoint_path,
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = (
        OUTPUT_DIR
        /
        "lstm_metrics.json"
    )


    results = {

        "model":
            "LSTM",

        "research_role":
            "sequence_only_baseline",

        "dataset":
            "Kyokushin karate optical motion capture",

        "input_shape":
            [
                100,
                39,
                3,
            ],

        "lstm_input_shape":
            [
                100,
                117,
            ],

        "velocity_channels":
            False,

        "training_augmentation":
            True,

        "validation_augmentation":
            False,

        "test_augmentation":
            False,

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_val_f1,

        "train_samples":
            len(train_ds),

        "validation_samples":
            len(val_ds),

        "test_samples":
            len(test_ds),

        "labels":
            train_ds.labels,

        "test_metrics":
            test_metrics,
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


    # ========================================================
    # FINISH
    # ========================================================

    print()

    print(
        f"Saved checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Saved metrics: "
        f"{metrics_path}"
    )

    print()

    print("=" * 70)

    print(
        "LSTM EXPERIMENT COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()