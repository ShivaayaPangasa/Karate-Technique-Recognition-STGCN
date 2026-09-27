import csv
from pathlib import Path

import numpy as np

from preprocess import preprocess_file


MANIFEST_PATH = Path(
    "data/manifest_split.csv"
)

OUT_DIR = Path(
    "data/processed"
)

INDEX_PATH = Path(
    "data/processed_index.csv"
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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index_rows = []

    failed = 0

    for i, row in enumerate(
        rows,
        start=1
    ):

        technique = int(
            row["technique"]
        )

        condition = int(
            row["condition"]
        )

        trial = int(
            row["trial"]
        )

        output_path = (
            OUT_DIR
            / f"{row['participant']}_"
              f"S{technique:02d}_"
              f"E{condition:02d}_"
              f"T{trial:02d}_"
              f"{i}.npy"
        )

        try:

            tensor = preprocess_file(
                Path(
                    row["filepath"]
                )
            )

            if tensor.shape != (
                100,
                39,
                3
            ):

                raise ValueError(
                    f"Unexpected tensor shape: "
                    f"{tensor.shape}"
                )

            if not np.isfinite(
                tensor
            ).all():

                raise ValueError(
                    "Tensor contains NaN/Inf"
                )

            np.save(
                output_path,
                tensor
            )

            index_rows.append(
                {
                    "npy_path":
                        output_path.as_posix(),

                    "participant":
                        row["participant"],

                    "technique":
                        row["technique"],

                    "condition":
                        row["condition"],

                    "trial":
                        row["trial"],

                    # IMPORTANT:
                    # technique is the prediction target.
                    "label":
                        row["technique"],

                    "split":
                        row["split"],
                }
            )

        except Exception as exc:

            failed += 1

            print(
                f"FAILED "
                f"{row['filename']}: "
                f"{exc}"
            )

        if i % 100 == 0:

            print(
                f"Processed "
                f"{i}/{len(rows)}"
            )

    with open(
        INDEX_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fields = [
            "npy_path",
            "participant",
            "technique",
            "condition",
            "trial",
            "label",
            "split",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(
            index_rows
        )

    print(
        f"\nSucceeded: "
        f"{len(index_rows)}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Saved: {INDEX_PATH}"
    )