import csv
import numpy as np
import torch
from torch.utils.data import Dataset


INDEX_PATH = "data/processed_index.csv"


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment(xyz):
    """
    Apply small training-only spatial augmentation.

    Input:
        xyz shape = (100, 39, 3)

    Operations:
        1. Small rotation around vertical Z axis
        2. Small Gaussian coordinate noise

    This augmentation is used ONLY for the training split.
    Validation and test data remain unchanged.
    """

    # Random rotation between -15 and +15 degrees
    angle = np.random.uniform(
        -15.0,
        15.0
    ) * np.pi / 180.0

    c = np.cos(angle)
    s = np.sin(angle)

    rotation = np.array(
        [
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ],
        dtype=np.float32,
    )

    xyz = xyz @ rotation.T

    # Small Gaussian coordinate noise
    noise = np.random.normal(
        0.0,
        0.01,
        xyz.shape
    ).astype(np.float32)

    xyz = xyz + noise

    return xyz.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class KarateDataset(Dataset):

    def __init__(
        self,
        split,
        index_path=INDEX_PATH,
        augment_data=False,
    ):
        """
        Load one dataset split.

        split:
            "train"
            "val"
            "test"

        Expected sample shape:
            (100, 39, 3)
        """

        with open(
            index_path,
            newline="",
            encoding="utf-8",
        ) as f:

            rows = list(
                csv.DictReader(f)
            )

        # Keep only requested split
        self.rows = [
            row
            for row in rows
            if row["split"] == split
        ]

        # Establish one global label ordering
        # for all splits.
        self.labels = sorted(
            set(
                row["label"]
                for row in rows
            )
        )

        self.label_to_idx = {
            label: index
            for index, label
            in enumerate(self.labels)
        }

        self.augment_data = augment_data

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        row = self.rows[index]

        # ----------------------------------------------------
        # LOAD PROCESSED XYZ DATA
        # ----------------------------------------------------

        xyz = np.load(
            row["npy_path"]
        ).astype(np.float32)

        # ----------------------------------------------------
        # SHAPE VALIDATION
        # ----------------------------------------------------

        expected_shape = (
            100,
            39,
            3,
        )

        if xyz.shape != expected_shape:

            raise ValueError(
                f"Unexpected input shape "
                f"{xyz.shape} for "
                f"{row['npy_path']}. "
                f"Expected {expected_shape}."
            )

        # ----------------------------------------------------
        # FINITE VALUE VALIDATION
        # ----------------------------------------------------

        if not np.isfinite(xyz).all():

            raise ValueError(
                f"Non-finite values found in "
                f"{row['npy_path']}"
            )

        # ----------------------------------------------------
        # TRAINING AUGMENTATION
        # ----------------------------------------------------

        if self.augment_data:

            xyz = augment(xyz)

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = self.label_to_idx[
            row["label"]
        ]

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return (
            torch.from_numpy(
                xyz
            ).float(),

            label,
        )