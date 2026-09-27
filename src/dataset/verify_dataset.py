import csv
from pathlib import Path
from collections import defaultdict

import numpy as np


MANIFEST_PATH = Path(
    "data/manifest_split.csv"
)

INDEX_PATH = Path(
    "data/processed_index.csv"
)


TECHNIQUE_NAMES = {

    "1":
        "Gyaku-Zuki",

    "2":
        "Mae-Geri",

    "3":
        "Mawashi-Geri gedan",

    "4":
        "Mawashi-Geri jodan",

    "5":
        "Ushiro-Mawashi-Geri",
}


CONDITION_NAMES = {

    "1":
        "air",

    "2":
        "shield",

    "3":
        "attacker",

    "4":
        "defender",
}


if __name__ == "__main__":

    with open(
        MANIFEST_PATH,
        newline="",
        encoding="utf-8"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    techniques = sorted(
        {
            r["technique"]
            for r in rows
        }
    )

    conditions = sorted(
        {
            r["condition"]
            for r in rows
        }
    )

    print(
        "=== DATASET VERIFICATION ==="
    )

    print(
        "Technique codes:",
        techniques
    )

    print(
        "Condition codes:",
        conditions
    )

    assert techniques == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]

    assert conditions == [
        "1",
        "2",
        "3",
        "4",
    ]

    print(
        "\nTECHNIQUES"
    )

    for code in techniques:

        count = sum(
            r["technique"] == code
            for r in rows
        )

        print(
            f"S{int(code):02d} = "
            f"{TECHNIQUE_NAMES[code]} "
            f"-> {count} files"
        )

    print(
        "\nCONDITIONS"
    )

    for code in conditions:

        count = sum(
            r["condition"] == code
            for r in rows
        )

        print(
            f"E{int(code):02d} = "
            f"{CONDITION_NAMES[code]} "
            f"-> {count} files"
        )

    # ------------------------------------------------
    # PARTICIPANT SPLIT CHECK
    # ------------------------------------------------

    participants = defaultdict(
        set
    )

    for row in rows:

        participants[
            row["split"]
        ].add(
            row["participant"]
        )

    train = participants["train"]
    val = participants["val"]
    test = participants["test"]

    print(
        "\nPARTICIPANT OVERLAP"
    )

    print(
        "Train ∩ Val:",
        len(train & val)
    )

    print(
        "Train ∩ Test:",
        len(train & test)
    )

    print(
        "Val ∩ Test:",
        len(val & test)
    )

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    # ------------------------------------------------
    # PROCESSED DATA CHECK
    # ------------------------------------------------

    if INDEX_PATH.exists():

        with open(
            INDEX_PATH,
            newline="",
            encoding="utf-8"
        ) as f:

            processed = list(
                csv.DictReader(f)
            )

        labels = sorted(
            {
                r["label"]
                for r in processed
            }
        )

        print(
            "\nProcessed labels:",
            labels
        )

        assert labels == [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]

        for row in processed[
            :min(50, len(processed))
        ]:

            x = np.load(
                row["npy_path"]
            )

            assert x.shape == (
                100,
                39,
                3
            )

            assert np.isfinite(
                x
            ).all()

        print(
            "Tensor shape verified: "
            "(100, 39, 3)"
        )

    print(
        "\nPASS."
    )

    print(
        "The dataset is now configured "
        "for 5-class karate technique "
        "recognition."
    )