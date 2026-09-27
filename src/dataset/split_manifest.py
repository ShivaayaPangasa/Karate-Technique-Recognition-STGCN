import random
from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path(
    "data/manifest.csv"
)

SPLIT_MANIFEST_PATH = Path(
    "data/manifest_split.csv"
)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42


def split_participants(
    participants
):

    participants = sorted(
        participants
    )

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        participants
    )

    n = len(
        participants
    )

    n_train = int(
        n * TRAIN_FRAC
    )

    n_val = int(
        n * VAL_FRAC
    )

    train_ids = set(
        participants[
            :n_train
        ]
    )

    val_ids = set(
        participants[
            n_train:n_train + n_val
        ]
    )

    test_ids = set(
        participants[
            n_train + n_val:
        ]
    )

    return (
        train_ids,
        val_ids,
        test_ids
    )


if __name__ == "__main__":

    df = pd.read_csv(
        MANIFEST_PATH
    )

    df = df[
        df["participant"].notna()
    ].copy()

    participants = (
        df["participant"]
        .unique()
        .tolist()
    )

    train_ids, val_ids, test_ids = (
        split_participants(
            participants
        )
    )

    assert train_ids.isdisjoint(
        val_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert val_ids.isdisjoint(
        test_ids
    )

    print(
        f"Total participants: "
        f"{len(participants)}"
    )

    print(
        f"Train participants: "
        f"{len(train_ids)}"
    )

    print(
        f"Validation participants: "
        f"{len(val_ids)}"
    )

    print(
        f"Test participants: "
        f"{len(test_ids)}"
    )

    def assign_split(
        participant
    ):

        if participant in train_ids:
            return "train"

        if participant in val_ids:
            return "val"

        if participant in test_ids:
            return "test"

        return "unknown"

    df["split"] = (
        df["participant"]
        .apply(assign_split)
    )

    print(
        "\nConfirmed:"
    )

    print(
        "Train ∩ Val =",
        len(
            train_ids & val_ids
        )
    )

    print(
        "Train ∩ Test =",
        len(
            train_ids & test_ids
        )
    )

    print(
        "Val ∩ Test =",
        len(
            val_ids & test_ids
        )
    )

    print(
        "\nFiles per split:"
    )

    print(
        df["split"]
        .value_counts()
    )

    print(
        "\nTechnique distribution:"
    )

    print(
        pd.crosstab(
            df["split"],
            df["technique"]
        ).sort_index()
    )

    df.to_csv(
        SPLIT_MANIFEST_PATH,
        index=False
    )

    print(
        f"\nSaved: "
        f"{SPLIT_MANIFEST_PATH}"
    )