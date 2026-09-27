import csv
from pathlib import Path


MANIFEST_PATH = Path(
    "data/manifest.csv"
)


# Official dataset correction mapping.
#
# These corrections affect the execution condition
# and trial metadata.
#
# The S-code / technique does NOT change.

FILENAME_CORRECTIONS = {

    "2017-03-03-B0388-S05-E01-T02.c3d":
        "2017-03-03-B0388-S05-E02-T01.c3d",

    "2017-03-03-B0388-S05-E02-T01.c3d":
        "2017-03-03-B0388-S05-E01-T02.c3d",

    "2017-03-07-B0396-S03-E02-T01.c3d":
        "2017-03-07-B0396-S03-E01-T03.c3d",
}


def corrected_condition_trial(
    corrected_filename
):

    stem = Path(
        corrected_filename
    ).stem

    parts = stem.split("-")

    condition = str(
        int(
            parts[-2]
            .lstrip("Ee")
        )
    )

    trial = str(
        int(
            parts[-1]
            .lstrip("Tt")
        )
    )

    return condition, trial


if __name__ == "__main__":

    with open(
        MANIFEST_PATH,
        newline="",
        encoding="utf-8"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    fixed_count = 0

    for row in rows:

        filename = row["filename"]

        if filename not in FILENAME_CORRECTIONS:
            continue

        corrected_filename = (
            FILENAME_CORRECTIONS[
                filename
            ]
        )

        condition, trial = (
            corrected_condition_trial(
                corrected_filename
            )
        )

        print(
            f"{filename}: "
            f"condition {row['condition']} "
            f"-> {condition}, "
            f"trial {row['trial']} "
            f"-> {trial}"
        )

        row["condition"] = condition
        row["trial"] = trial

        fixed_count += 1

    if rows:

        with open(
            MANIFEST_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

    print(
        f"\nCorrected {fixed_count} row(s)."
    )

    print(
        f"Manifest updated: {MANIFEST_PATH}"
    )