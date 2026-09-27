import numpy as np

# The 39 Plug-in Gait markers used by the ST-GCN.
# Kept here so graph construction does not depend on ezc3d.
RAW_MARKERS = [
    "LFHD", "RFHD", "LBHD", "RBHD",
    "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI",
    "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
]


# Anatomical skeleton connectivity for the 39 Plug-in Gait markers.
EDGES = [
    ("LFHD", "RFHD"), ("LFHD", "LBHD"), ("RFHD", "RBHD"), ("LBHD", "RBHD"),
    ("LFHD", "C7"), ("RFHD", "C7"),
    ("C7", "CLAV"), ("C7", "T10"), ("CLAV", "STRN"), ("C7", "RBAK"), ("T10", "RBAK"),
    ("C7", "LSHO"), ("C7", "RSHO"), ("CLAV", "LSHO"), ("CLAV", "RSHO"),
    ("LSHO", "LUPA"), ("LUPA", "LELB"), ("LELB", "LFRM"),
    ("LFRM", "LWRA"), ("LFRM", "LWRB"), ("LWRA", "LFIN"), ("LWRB", "LFIN"),
    ("RSHO", "RUPA"), ("RUPA", "RELB"), ("RELB", "RFRM"),
    ("RFRM", "RWRA"), ("RFRM", "RWRB"), ("RWRA", "RFIN"), ("RWRB", "RFIN"),
    ("LASI", "RASI"), ("LASI", "LPSI"), ("RASI", "RPSI"), ("LPSI", "RPSI"),
    ("T10", "LPSI"), ("T10", "RPSI"),
    ("LASI", "LTHI"), ("LPSI", "LTHI"), ("LTHI", "LKNE"), ("LKNE", "LTIB"),
    ("LTIB", "LANK"), ("LANK", "LHEE"), ("LANK", "LTOE"), ("LHEE", "LTOE"),
    ("RASI", "RTHI"), ("RPSI", "RTHI"), ("RTHI", "RKNE"), ("RKNE", "RTIB"),
    ("RTIB", "RANK"), ("RANK", "RHEE"), ("RANK", "RTOE"), ("RHEE", "RTOE"),
]


def build_adjacency():
    n = len(RAW_MARKERS)
    idx = {m: i for i, m in enumerate(RAW_MARKERS)}

    # Start with self-loops.
    A = np.eye(n, dtype=np.float32)

    # Add anatomical connections.
    for a, b in EDGES:
        i, j = idx[a], idx[b]
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Symmetric normalization:
    # D^(-1/2) A D^(-1/2)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))

    return D_inv_sqrt @ A @ D_inv_sqrt