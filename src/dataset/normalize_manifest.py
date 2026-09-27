import csv
from pathlib import Path


MANIFEST_PATH = Path(
    "data/manifest.csv"
)


def normalize_code(value):

    if value is None or value == "":
        return value

    return str(
        int(value)
    )


if __name__ == "__main__":

    with open(
        MANIFEST_PATH,
        newline="",
        encoding="utf-8"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    for row in rows:

        for field in (
            "technique",
            "condition",
            "trial"
        ):

            row[field] = normalize_code(
                row[field]
            )

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
        f"Normalized technique/condition/trial "
        f"in {MANIFEST_PATH}"
    )