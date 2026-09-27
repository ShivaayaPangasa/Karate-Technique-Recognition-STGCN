import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================================
# IMPORT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT_DIR / "src" / "models")
)

from dataset import KarateDataset
from lstm_model import LSTMClassifier
from stgcn_model import STGCN
from graph import build_adjacency


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = "data/processed_index.csv"

LSTM_CHECKPOINT = (
    "outputs/lstm_best.pt"
)

STGCN_CHECKPOINT = (
    "outputs/stgcn_best.pt"
)

OUTPUT_DIR = Path(
    "outputs/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_CLASSES = 5

CLASS_NAMES = [
    "Gyaku-Zuki",
    "Mae-Geri",
    "Mawashi-Geri gedan",
    "Mawashi-Geri jodan",
    "Ushiro-Mawashi-Geri",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    num_classes,
):
    """
    Calculate overall and per-class metrics.

    Metrics:
        Accuracy
        Macro Precision
        Macro Recall
        Macro F1
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )

    precisions = []
    recalls = []
    f1_scores = []

    per_class = {}

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

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RECALL
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

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

        per_class[
            CLASS_NAMES[cls]
        ] = {
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "support": int(
                np.sum(
                    y_true == cls
                )
            ),
        }

    return {
        "accuracy": accuracy,

        "macro_precision": float(
            np.mean(
                precisions
            )
        ),

        "macro_recall": float(
            np.mean(
                recalls
            )
        ),

        "macro_f1": float(
            np.mean(
                f1_scores
            )
        ),

        "per_class": per_class,
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def build_confusion_matrix(
    y_true,
    y_pred,
    num_classes,
):

    matrix = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    for true_label, predicted_label in zip(
        y_true,
        y_pred,
    ):

        matrix[
            true_label,
            predicted_label
        ] += 1

    return matrix


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

def normalize_confusion_matrix(
    matrix
):

    matrix = matrix.astype(
        np.float64
    )

    row_sums = matrix.sum(
        axis=1,
        keepdims=True
    )

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix
        ),
        where=row_sums != 0,
    )

    return normalized


# ============================================================
# MODEL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_models(
    lstm,
    stgcn,
    dataset,
):

    lstm.eval()
    stgcn.eval()

    all_labels = []
    lstm_predictions = []
    stgcn_predictions = []

    # --------------------------------------------------------
    # EVALUATE EVERY TEST SAMPLE
    # --------------------------------------------------------

    for index in range(len(dataset)):

        x, y = dataset[index]

        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        #
        # Dataset sample:
        #     [T, V, C]
        #     [100, 39, 3]
        #
        # After unsqueeze:
        #     [B, T, V, C]
        #     [1, 100, 39, 3]
        # ----------------------------------------------------

        x = x.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # LSTM
        #
        # [B, T, V, C]
        #     ↓
        # [B, T, V*C]
        #
        # [1, 100, 39, 3]
        #     ↓
        # [1, 100, 117]
        # ----------------------------------------------------

        lstm_input = x.reshape(
            x.shape[0],
            x.shape[1],
            -1,
        )

        lstm_logits = lstm(lstm_input)

        lstm_prediction = (
            torch.argmax(
                lstm_logits,
                dim=1,
            )
            .item()
        )

        # ----------------------------------------------------
        # ST-GCN
        #
        # IMPORTANT:
        #
        # x is already:
        #     [B, T, V, C]
        #     [1, 100, 39, 3]
        #
        # STGCN.forward() performs the permutation internally:
        #
        #     [B, T, V, C]
        #          ↓
        #     [B, C, T, V]
        #     [1, 3, 100, 39]
        #
        # Therefore DO NOT permute x here.
        # ----------------------------------------------------

        if x.ndim != 4:
            raise ValueError(
                "Unexpected input dimensions for ST-GCN: "
                f"{tuple(x.shape)}. "
                "Expected [B, T, V, C]."
            )

        expected_shape = (
            x.shape[0],
            100,
            39,
            3,
        )

        if tuple(x.shape) != expected_shape:
            raise ValueError(
                "ST-GCN input shape is incorrect. "
                f"Got {tuple(x.shape)}, "
                f"expected {expected_shape}."
            )

        stgcn_logits = stgcn(x)

        stgcn_prediction = (
            torch.argmax(
                stgcn_logits,
                dim=1,
            )
            .item()
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_labels.append(
            int(y)
        )

        lstm_predictions.append(
            lstm_prediction
        )

        stgcn_predictions.append(
            stgcn_prediction
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (index + 1) % 50 == 0 or index == 0:
            print(
                f"Evaluated samples: {index + 1}/{len(dataset)}"
            )

    # --------------------------------------------------------
    # FINAL SAFETY CHECK
    # --------------------------------------------------------

    if len(all_labels) != len(dataset):
        raise ValueError(
            "Unexpected number of test labels. "
            f"Got {len(all_labels)}, "
            f"expected {len(dataset)}."
        )

    if len(lstm_predictions) != len(dataset):
        raise ValueError(
            "Unexpected number of LSTM predictions. "
            f"Got {len(lstm_predictions)}, "
            f"expected {len(dataset)}."
        )

    if len(stgcn_predictions) != len(dataset):
        raise ValueError(
            "Unexpected number of ST-GCN predictions. "
            f"Got {len(stgcn_predictions)}, "
            f"expected {len(dataset)}."
        )

    print(
        f"Evaluated samples: {len(dataset)}/{len(dataset)}"
    )

    return (
        np.array(all_labels),
        np.array(lstm_predictions),
        np.array(stgcn_predictions),
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    matrix,
    filename,
    title,
):

    import matplotlib.pyplot as plt

    normalized = (
        normalize_confusion_matrix(
            matrix
        )
    )

    figure = plt.figure(
        figsize=(
            9,
            7,
        )
    )

    axis = figure.add_subplot(
        111
    )

    image = axis.imshow(
        normalized,
        interpolation="nearest",
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "Predicted technique"
    )

    axis.set_ylabel(
        "True technique"
    )

    axis.set_xticks(
        range(
            NUM_CLASSES
        )
    )

    axis.set_yticks(
        range(
            NUM_CLASSES
        )
    )

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        CLASS_NAMES
    )

    # --------------------------------------------------------
    # WRITE VALUES INTO CELLS
    # --------------------------------------------------------

    for row in range(
        NUM_CLASSES
    ):

        for column in range(
            NUM_CLASSES
        ):

            axis.text(
                column,
                row,
                f"{normalized[row, column]:.2f}",
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis
    )

    figure.tight_layout()

    path = (
        OUTPUT_DIR
        /
        filename
    )

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Saved confusion matrix: "
        f"{path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "LSTM vs ST-GCN TEST SET COMPARISON"
    )

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Input representation: "
        "100 × 39 × 3 XYZ"
    )

    print(
        "Velocity channels: disabled"
    )

    print()


    # ========================================================
    # TEST DATASET
    # ========================================================

    test_dataset = KarateDataset(
        split="test",
        index_path=INDEX_PATH,
        augment_data=False,
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    print(
        f"Classes: "
        f"{test_dataset.labels}"
    )

    print()


    # ========================================================
    # VERIFY DATASET
    # ========================================================

    if len(
        test_dataset
    ) != 264:

        raise ValueError(
            "Expected 264 test samples, "
            f"but found "
            f"{len(test_dataset)}."
        )

    if len(
        test_dataset.labels
    ) != NUM_CLASSES:

        raise ValueError(
            "Expected 5 classes, "
            f"but found "
            f"{test_dataset.labels}"
        )


    # ========================================================
    # LOAD LSTM
    # ========================================================

    print(
        "Loading LSTM checkpoint..."
    )

    lstm_checkpoint = torch.load(
        LSTM_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    lstm = LSTMClassifier(
        input_size=117,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
    ).to(
        DEVICE
    )

    lstm.load_state_dict(
        lstm_checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "LSTM checkpoint loaded."
    )


    # ========================================================
    # LOAD ST-GCN
    # ========================================================

    print(
        "Loading ST-GCN checkpoint..."
    )

    stgcn_checkpoint = torch.load(
        STGCN_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    adjacency = (
        build_adjacency()
    )

    stgcn = STGCN(
        adjacency,
        in_channels=3,
        num_classes=NUM_CLASSES,
    ).to(
        DEVICE
    )

    stgcn.load_state_dict(
        stgcn_checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "ST-GCN checkpoint loaded."
    )

    print()


    # ========================================================
    # EVALUATE
    # ========================================================

    print(
        "Evaluating both models on "
        "the same test set..."
    )

    (
        y_true,
        lstm_predictions,
        stgcn_predictions,
    ) = evaluate_models(
        lstm,
        stgcn,
        test_dataset,
    )

    print(
        f"Evaluated samples: "
        f"{len(y_true)}"
    )

    print()


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(
        y_true
    ) != 264:

        raise ValueError(
            "Unexpected number of "
            "test predictions."
        )


    # ========================================================
    # METRICS
    # ========================================================

    lstm_metrics = calculate_metrics(
        y_true,
        lstm_predictions,
        NUM_CLASSES,
    )

    stgcn_metrics = calculate_metrics(
        y_true,
        stgcn_predictions,
        NUM_CLASSES,
    )


    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    lstm_matrix = (
        build_confusion_matrix(
            y_true,
            lstm_predictions,
            NUM_CLASSES,
        )
    )

    stgcn_matrix = (
        build_confusion_matrix(
            y_true,
            stgcn_predictions,
            NUM_CLASSES,
        )
    )


    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "OVERALL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "LSTM"
    )

    print(
        f"  Accuracy : "
        f"{lstm_metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{lstm_metrics['macro_precision']:.4f}"
    )

    print(
        f"  Recall   : "
        f"{lstm_metrics['macro_recall']:.4f}"
    )

    print(
        f"  Macro-F1 : "
        f"{lstm_metrics['macro_f1']:.4f}"
    )

    print()

    print(
        "ST-GCN"
    )

    print(
        f"  Accuracy : "
        f"{stgcn_metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision: "
        f"{stgcn_metrics['macro_precision']:.4f}"
    )

    print(
        f"  Recall   : "
        f"{stgcn_metrics['macro_recall']:.4f}"
    )

    print(
        f"  Macro-F1 : "
        f"{stgcn_metrics['macro_f1']:.4f}"
    )

    print()
    
    
    # ========================================================
    # DIFFERENCE
    # ========================================================

    accuracy_difference = (
        stgcn_metrics["accuracy"]
        - lstm_metrics["accuracy"]
    )

    precision_difference = (
        stgcn_metrics["macro_precision"]
        - lstm_metrics["macro_precision"]
    )

    recall_difference = (
        stgcn_metrics["macro_recall"]
        - lstm_metrics["macro_recall"]
    )

    f1_difference = (
        stgcn_metrics["macro_f1"]
        - lstm_metrics["macro_f1"]
    )


    print(
        "ST-GCN minus LSTM"
    )

    print(
        f"  Accuracy : {accuracy_difference:.4f}"
    )

    print(
        f"  Precision: {precision_difference:.4f}"
    )

    print(
        f"  Recall   : {recall_difference:.4f}"
    )

    print(
        f"  Macro-F1 : {f1_difference:.4f}"
    )

    print()   

    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "PER-CLASS RESULTS"
    )

    print(
        "=" * 70
    )

    print()

    for class_name in CLASS_NAMES:

        lstm_class = (
            lstm_metrics[
                "per_class"
            ][class_name]
        )

        stgcn_class = (
            stgcn_metrics[
                "per_class"
            ][class_name]
        )

        print(
            class_name
        )

        print(
            f"  Support: "
            f"{lstm_class['support']}"
        )

        print(
            f"  LSTM   -> "
            f"Precision: "
            f"{lstm_class['precision']:.4f}, "
            f"Recall: "
            f"{lstm_class['recall']:.4f}, "
            f"F1: "
            f"{lstm_class['f1']:.4f}"
        )

        print(
            f"  ST-GCN -> "
            f"Precision: "
            f"{stgcn_class['precision']:.4f}, "
            f"Recall: "
            f"{stgcn_class['recall']:.4f}, "
            f"F1: "
            f"{stgcn_class['f1']:.4f}"
        )

        print()


    # ========================================================
    # CONFUSION MATRIX PRINT
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "LSTM CONFUSION MATRIX"
    )

    print(
        "=" * 70
    )

    print(
        lstm_matrix
    )

    print()

    print(
        "=" * 70
    )

    print(
        "ST-GCN CONFUSION MATRIX"
    )

    print(
        "=" * 70
    )

    print(
        stgcn_matrix
    )

    print()


    # ========================================================
    # SAVE FIGURES
    # ========================================================

    save_confusion_matrix(
        lstm_matrix,
        "lstm_confusion_matrix.png",
        "LSTM Normalized Confusion Matrix",
    )

    save_confusion_matrix(
        stgcn_matrix,
        "st-gcn_confusion_matrix.png",
        "ST-GCN Normalized Confusion Matrix",
    )


    # ========================================================
    # SAVE REPORTS
    # ========================================================

    with open(
        "outputs/lstm_classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            lstm_metrics,
            file,
            indent=2,
        )


    with open(
        "outputs/st-gcn_classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            stgcn_metrics,
            file,
            indent=2,
        )


    # ========================================================
    # SAVE COMPARISON
    # ========================================================

    comparison = {

        "experiment":
            "LSTM vs ST-GCN",

        "dataset":
            "Kyokushin karate optical motion capture",

        "test_samples":
            len(y_true),

        "input_shape":
            [
                100,
                39,
                3,
            ],

        "velocity_channels":
            False,

        "classes":
            CLASS_NAMES,

        "lstm": {
            "accuracy":
                lstm_metrics[
                    "accuracy"
                ],

            "macro_precision":
                lstm_metrics[
                    "macro_precision"
                ],

            "macro_recall":
                lstm_metrics[
                    "macro_recall"
                ],

            "macro_f1":
                lstm_metrics[
                    "macro_f1"
                ],

            "per_class":
                lstm_metrics[
                    "per_class"
                ],

            "confusion_matrix":
                lstm_matrix.tolist(),
        },

        "stgcn": {
            "accuracy":
                stgcn_metrics[
                    "accuracy"
                ],

            "macro_precision":
                stgcn_metrics[
                    "macro_precision"
                ],

            "macro_recall":
                stgcn_metrics[
                    "macro_recall"
                ],

            "macro_f1":
                stgcn_metrics[
                    "macro_f1"
                ],

            "per_class":
                stgcn_metrics[
                    "per_class"
                ],

            "confusion_matrix":
                stgcn_matrix.tolist(),
        },

        "stgcn_minus_lstm": {
            "accuracy":
                (
                    stgcn_metrics[
                        "accuracy"
                    ]
                    -
                    lstm_metrics[
                        "accuracy"
                    ]
                ),

            "macro_precision":
                (
                    stgcn_metrics[
                        "macro_precision"
                    ]
                    -
                    lstm_metrics[
                        "macro_precision"
                    ]
                ),

            "macro_recall":
                (
                    stgcn_metrics[
                        "macro_recall"
                    ]
                    -
                    lstm_metrics[
                        "macro_recall"
                    ]
                ),

            "macro_f1":
                (
                    stgcn_metrics[
                        "macro_f1"
                    ]
                    -
                    lstm_metrics[
                        "macro_f1"
                    ]
                ),
        },
    }


    comparison_path = (
        "outputs/"
        "lstm_vs_stgcn_comparison.json"
    )


    with open(
        comparison_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison,
            file,
            indent=2,
        )


    # ========================================================
    # FINISH
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "COMPARISON COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  outputs/lstm_classification_report.json"
    )

    print(
        "  outputs/st-gcn_classification_report.json"
    )

    print(
        "  outputs/lstm_vs_stgcn_comparison.json"
    )

    print(
        "  outputs/figures/lstm_confusion_matrix.png"
    )

    print(
        "  outputs/figures/"
        "st-gcn_confusion_matrix.png"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()