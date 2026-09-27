from pathlib import Path
import re
import pandas as pd
import ezc3d


RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.csv")

FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})-"
    r"(?P<participant>[A-Za-z0-9]+)-"
    r"S(?P<technique>\d+)-"
    r"E(?P<condition>\d+)-"
    r"T(?P<trial>\d+)\.c3d$"
)

VALID_TECHNIQUES = {"01", "02", "03", "04", "05"}
VALID_CONDITIONS = {"01", "02", "03", "04"}


def inspect_c3d(path):
    """
    Read basic metadata from a C3D file.
    """
    try:
        c3d = ezc3d.c3d(str(path))

        points = c3d["data"]["points"]

        n_points = points.shape[1]
        n_frames = points.shape[2]

        sampling_rate = float(
            c3d["parameters"]["POINT"]["RATE"]["value"][0]
        )

        duration_sec = (
            n_frames / sampling_rate
            if sampling_rate > 0
            else None
        )

        return {
            "sampling_rate": sampling_rate,
            "n_frames": n_frames,
            "n_points": n_points,
            "duration_sec": duration_sec,
            "load_error": None,
        }

    except Exception as exc:
        return {
            "sampling_rate": None,
            "n_frames": None,
            "n_points": None,
            "duration_sec": None,
            "load_error": str(exc),
        }


def parse_filename(filename):
    match = FILENAME_PATTERN.match(filename)

    if not match:
        return None

    return match.groupdict()


def main():

    files = sorted(RAW_DIR.rglob("*.c3d"))

    print(f"Scanning {len(files)} C3D files...")

    records = []
    unparsed = []
    load_errors = []

    for i, path in enumerate(files, start=1):

        filename = path.name

        parsed = parse_filename(filename)

        if parsed is None:
            unparsed.append(filename)
            continue

        metadata = inspect_c3d(path)

        if metadata["load_error"] is not None:
            load_errors.append(filename)

        record = {
            "filepath": str(path),
            "filename": filename,
            **parsed,
            **metadata,
        }

        records.append(record)

        if i % 100 == 0:
            print(f"  processed {i}/{len(files)}")

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No valid C3D files were parsed.")

    # Normalize code columns as strings.
    df["technique"] = df["technique"].astype(str)
    df["condition"] = df["condition"].astype(str)
    df["trial"] = df["trial"].astype(str)

    print()
    print(f"Total files: {len(df)}")
    print(f"Unique participants: {df['participant'].nunique()}")
    print(f"Unparsed filenames: {len(unparsed)}")
    print(f"Files with load errors: {len(load_errors)}")

    print()
    print("Technique codes BEFORE validation:")
    print(df["technique"].value_counts().sort_index())

    print()
    print("Condition codes:")
    print(df["condition"].value_counts().sort_index())

    # ---------------------------------------------------------
    # Validate technique labels.
    # ---------------------------------------------------------

    invalid_technique_mask = ~df["technique"].isin(VALID_TECHNIQUES)

    invalid_techniques = df.loc[
        invalid_technique_mask,
        ["filepath", "filename", "participant", "technique",
         "condition", "trial"]
    ]

    if not invalid_techniques.empty:

        print()
        print("WARNING: Invalid technique codes detected.")
        print(invalid_techniques.to_string(index=False))

        print()
        print(
            f"Excluding {len(invalid_techniques)} file(s) "
            "with invalid technique codes from the research manifest."
        )

        # IMPORTANT:
        # We do NOT relabel these files.
        # They remain in data/raw/ but are excluded from the
        # technique-recognition experiment.
        df = df.loc[~invalid_technique_mask].copy()

    # ---------------------------------------------------------
    # Validate conditions.
    # ---------------------------------------------------------

    invalid_condition_mask = ~df["condition"].isin(VALID_CONDITIONS)

    invalid_conditions = df.loc[
        invalid_condition_mask,
        ["filepath", "filename", "participant", "technique",
         "condition", "trial"]
    ]

    if not invalid_conditions.empty:
        print()
        print("ERROR: Invalid condition codes detected:")
        print(invalid_conditions.to_string(index=False))

        raise ValueError(
            "Invalid condition code(s) found. "
            "Fix the dataset before continuing."
        )

    # ---------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------

    final_techniques = sorted(df["technique"].unique(), key=int)
    final_conditions = sorted(df["condition"].unique(), key=int)

    print()
    print("Technique codes AFTER validation:")
    print(df["technique"].value_counts().sort_index())

    print()
    print("Final technique codes:", final_techniques)
    print("Final condition codes:", final_conditions)
    
    if final_techniques != ["01", "02", "03", "04", "05"]:
        raise ValueError(
            f"Expected techniques ['01', '02', '03', '04', '05'], "
            f"but found {final_techniques}"
        )

    if final_conditions != ["01", "02", "03", "04"]:
        raise ValueError(
            f"Expected conditions ['01', '02', '03', '04'], "
            f"but found {final_conditions}"
        )
    
    # Save final research manifest.
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(MANIFEST_PATH, index=False)

    print()
    print(f"Final research manifest contains {len(df)} files.")
    print(f"Manifest saved to {MANIFEST_PATH}")

    if unparsed:
        print()
        print("Unparsed filenames:")
        for filename in unparsed:
            print(" ", filename)

    if load_errors:
        print()
        print("Files with load errors:")
        for filename in load_errors:
            print(" ", filename)


if __name__ == "__main__":
    main()