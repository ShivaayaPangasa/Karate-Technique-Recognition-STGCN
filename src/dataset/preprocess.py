import numpy as np
import ezc3d

from marker_filter import RAW_MARKERS


PELVIS_MARKERS = ["LASI", "RASI", "LPSI", "RPSI"]

SCALE_MARKERS = ("C7", "LASI")

FIXED_LENGTH = 100

SPEED_PERCENTILE = 60

PAD_FRAMES = 5


def _strip_prefix(label):
    """
    Vicon sometimes prefixes marker labels with a subject name,
    for example:

        B0367:LFHD

    This removes the prefix so that marker matching works.
    """

    label = str(label).strip()

    return label.split(":")[-1] if ":" in label else label


def load_raw_markers(path):
    """
    Load the required 39 markers from a C3D file.

    Returns
    -------
    xyz:
        Shape (frames, 39, 3)

    residual:
        Shape (frames, 39)
    """

    c3d = ezc3d.c3d(str(path))

    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]

    points = np.asarray(
        c3d["data"]["points"],
        dtype=np.float64
    )

    clean_labels = [
        _strip_prefix(label)
        for label in labels
    ]

    idx = []

    missing = []

    for marker in RAW_MARKERS:

        if marker in clean_labels:

            idx.append(
                clean_labels.index(marker)
            )

        else:

            missing.append(marker)

    if missing:

        raise ValueError(
            f"Missing markers: {missing}\n"
            f"  Actual labels in file (first 20): "
            f"{labels[:20]}"
        )

    xyz = (
        points[:3, idx, :]
        .transpose(2, 1, 0)
    )

    residual = (
        points[3, idx, :].T
    )

    return xyz, residual


def interpolate_missing(xyz, residual):
    """
    Repair missing marker observations.

    A point is considered missing when either:

    1. Its C3D residual is negative, or
    2. Any of its XYZ coordinates is NaN/Inf.

    Missing values are linearly interpolated over time.

    Missing values at the beginning or end of a sequence
    are filled using the nearest valid observation.

    A marker with no valid observation at all is rejected.
    """

    xyz = np.asarray(
        xyz,
        dtype=np.float64
    )

    residual = np.asarray(
        residual,
        dtype=np.float64
    )

    n_frames, n_markers, _ = xyz.shape

    for m in range(n_markers):

        # Residual-based missing values.
        residual_missing = (
            residual[:, m] < 0
        )

        # Coordinate-based missing values.
        coordinate_missing = (
            ~np.isfinite(
                xyz[:, m, :]
            ).all(axis=1)
        )

        missing = (
            residual_missing
            | coordinate_missing
        )

        valid = ~missing

        # If the marker has no usable observations,
        # we cannot reconstruct it safely.
        if not valid.any():

            raise ValueError(
                f"Marker {RAW_MARKERS[m]} "
                f"has no valid observations"
            )

        # Nothing to repair.
        if not missing.any():

            continue

        valid_indices = np.where(valid)[0]

        missing_indices = np.where(missing)[0]

        for axis in range(3):

            values = xyz[
                valid_indices,
                m,
                axis
            ]

            xyz[
                missing_indices,
                m,
                axis
            ] = np.interp(
                missing_indices,
                valid_indices,
                values
            )

    if not np.isfinite(xyz).all():

        raise ValueError(
            "Non-finite values remain after "
            "marker interpolation"
        )

    return xyz


def center_on_pelvis(xyz):
    """
    Translate the skeleton so that the pelvis
    is approximately centered at the origin.
    """

    idx = [
        RAW_MARKERS.index(marker)
        for marker in PELVIS_MARKERS
    ]

    pelvis = xyz[
        :,
        idx,
        :
    ]

    if not np.isfinite(pelvis).all():

        raise ValueError(
            "Non-finite pelvis coordinates"
        )

    pelvis_center = pelvis.mean(
        axis=1,
        keepdims=True
    )

    return xyz - pelvis_center


def normalize_scale(xyz):
    """
    Normalize body size using the mean C7-LASI
    distance across the sequence.
    """

    i = RAW_MARKERS.index(
        SCALE_MARKERS[0]
    )

    j = RAW_MARKERS.index(
        SCALE_MARKERS[1]
    )

    distances = np.linalg.norm(
        xyz[:, i, :] - xyz[:, j, :],
        axis=1
    )

    distances = distances[
        np.isfinite(distances)
    ]

    if len(distances) == 0:

        raise ValueError(
            "No valid scale measurements"
        )

    ref = distances.mean()

    if not np.isfinite(ref):

        raise ValueError(
            "Invalid scale reference"
        )

    if ref <= 1e-6:

        raise ValueError(
            "Scale reference is too small"
        )

    xyz = xyz / ref

    if not np.isfinite(xyz).all():

        raise ValueError(
            "Non-finite values after scale normalization"
        )

    return xyz


def trim_idle_frames(xyz):
    """
    Keep the active-motion region while retaining
    a small amount of context before and after.
    """

    speed = np.linalg.norm(
        np.diff(xyz, axis=0),
        axis=2
    ).sum(axis=1)

    speed = np.nan_to_num(
        speed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    threshold = np.percentile(
        speed,
        SPEED_PERCENTILE
    )

    active = np.where(
        speed > threshold
    )[0]

    if len(active) == 0:

        return xyz

    start = max(
        0,
        active[0] - PAD_FRAMES
    )

    end = min(
        len(xyz),
        active[-1] + PAD_FRAMES + 2
    )

    return xyz[start:end]


def resample_time(
    xyz,
    target_len=FIXED_LENGTH
):
    """
    Resample every sequence to a fixed
    number of frames.
    """

    old_t = np.linspace(
        0,
        1,
        xyz.shape[0]
    )

    new_t = np.linspace(
        0,
        1,
        target_len
    )

    out = np.zeros(
        (
            target_len,
            xyz.shape[1],
            xyz.shape[2]
        ),
        dtype=np.float64
    )

    for marker in range(
        xyz.shape[1]
    ):

        for axis in range(3):

            out[
                :,
                marker,
                axis
            ] = np.interp(
                new_t,
                old_t,
                xyz[
                    :,
                    marker,
                    axis
                ]
            )

    if not np.isfinite(out).all():

        raise ValueError(
            "Non-finite values after temporal resampling"
        )

    return out


def preprocess_file(path):
    """
    Complete preprocessing pipeline.

    Output shape:

        (100, 39, 3)

    Channels:

        X, Y, Z marker coordinates.
    """

    xyz, residual = load_raw_markers(path)

    xyz = interpolate_missing(
        xyz,
        residual
    )

    xyz = center_on_pelvis(
        xyz
    )

    xyz = normalize_scale(
        xyz
    )

    xyz = trim_idle_frames(
        xyz
    )

    xyz = resample_time(
        xyz
    )

    if xyz.shape != (
        FIXED_LENGTH,
        len(RAW_MARKERS),
        3
    ):

        raise ValueError(
            f"Unexpected final shape: "
            f"{xyz.shape}"
        )

    if not np.isfinite(xyz).all():

        raise ValueError(
            "Final tensor contains NaN/Inf"
        )

    return xyz.astype(
        np.float32
    )