# ST-GCN-Based Karate Technique Recognition

## Spatio-Temporal Graph Convolutional Network for Karate Technique Recognition: A Comparison with LSTM

The project investigates whether explicitly modelling human-body topology with a Spatio-Temporal Graph Convolutional Network (ST-GCN) improves karate technique recognition compared with a sequence-only Long Short-Term Memory (LSTM) baseline under a subject-independent evaluation protocol.

---

## Research Question

> Does explicit modelling of human-body topology with an ST-GCN improve karate technique recognition compared with sequence-only LSTM modelling under a subject-independent evaluation protocol?

---

## Objective

The objective is to classify five Kyokushin karate techniques from 3D human motion-capture sequences.

Two approaches are compared:

- **LSTM:** sequence-based baseline using flattened body-marker coordinates at each time step.
- **ST-GCN:** graph-based model that explicitly represents anatomical relationships between body markers while modelling temporal movement.

The comparison is performed using the same processed dataset, subject-independent train/validation/test split, input representation, training augmentation strategy, and evaluation protocol.

---

## Dataset

The project uses the following publicly available dataset:

**Optical motion capture dataset of selected techniques in beginner and advanced Kyokushin karate athletes.**

The original dataset contains optical motion-capture recordings collected from Kyokushin karate athletes using a Vicon motion-capture system.

The dataset contains:

- 37 athletes
- 39 reflective body markers
- 1,411 C3D recordings in the original dataset
- Multiple karate techniques
- Multiple execution conditions

This project focuses on five techniques:

| Code | Technique |
|---|---|
| S01 | Gyaku-Zuki |
| S02 | Mae-Geri |
| S03 | Mawashi-Geri gedan |
| S04 | Mawashi-Geri jodan |
| S05 | Ushiro-Mawashi-Geri |

The original recordings also contain four execution conditions:

| Code | Condition |
|---|---|
| E01 | Air |
| E02 | Shield |
| E03 | Attacker |
| E04 | Defender |

### Dataset Sources

Scientific Data article:

https://doi.org/10.1038/s41597-021-00801-5

Figshare dataset:

https://doi.org/10.6084/m9.figshare.13164848

The original dataset publication and associated licensing/attribution requirements should be consulted when redistributing or using the dataset outside this academic project.

---

## Dataset Filtering and Final Research Set

The original dataset contains one recording outside the five selected technique classes used in this experiment.

The research pipeline therefore excludes the out-of-scope S06 recording.

During preprocessing, four additional S05 recordings from participant B0400 could not be processed because the required `LTHI` marker was absent from those C3D files.

Rather than fabricate or estimate the missing marker, those four recordings were excluded.

Therefore, the final processed research set contains:

**1,406 successfully processed recordings**

with five target technique classes.

---

## Preprocessing

Each C3D recording is processed using the following pipeline:

1. Load the C3D motion-capture recording.
2. Select the 39 required Plug-in Gait body markers.
3. Remove marker-name prefixes where necessary.
4. Detect invalid or missing coordinate values.
5. Interpolate valid marker trajectories where possible.
6. Fill valid endpoint gaps using the nearest valid observation.
7. Center the skeleton using pelvis landmarks.
8. Normalize body scale using the C7-LASI distance.
9. Remove idle frames.
10. Resample each sequence to a fixed length of 100 frames.
11. Validate that the final tensor contains only finite values.
12. Save the processed sequence as a NumPy array.

The final representation for every sample is:

```text
100 × 39 × 3
```

where:

- `100` = temporal frames
- `39` = body markers
- `3` = X, Y, Z coordinates

Velocity channels are not used in the primary experiment.

---

## Subject-Independent Evaluation

To reduce participant-specific leakage, the dataset is split at the participant level rather than randomly splitting individual recordings.

The final split contains:

| Split | Participants | Samples |
|---|---:|---:|
| Train | 25 | 968 |
| Validation | 5 | 174 |
| Test | 7 | 264 |
| **Total** | **37** | **1,406** |

No participant occurs in more than one split.

The random seed is:

```text
42
```

This makes the experimental split deterministic.

---

## Class Distribution

The final processed dataset contains the following technique distribution:

| Technique | Train | Validation | Test |
|---|---:|---:|---:|
| Gyaku-Zuki | 199 | 36 | 54 |
| Mae-Geri | 192 | 36 | 53 |
| Mawashi-Geri gedan | 194 | 36 | 52 |
| Mawashi-Geri jodan | 198 | 37 | 53 |
| Ushiro-Mawashi-Geri | 185 | 29 | 52 |
| **Total** | **968** | **174** | **264** |

---

# Models

## 1. LSTM Baseline

The LSTM is used as the sequence-only baseline.

Each frame originally contains:

```text
39 markers × 3 coordinates = 117 values
```

Therefore, the LSTM receives:

```text
100 × 117
```

per sequence.

Architecture:

```text
Input: 100 × 117
        ↓
2-layer LSTM
        ↓
Hidden size: 128
        ↓
Classification layer
        ↓
5 technique classes
```

The LSTM models temporal dependencies without explicitly representing anatomical graph connectivity.

---

## 2. ST-GCN

The proposed model represents the human body as a graph.

Each of the 39 body markers is treated as a graph node.

Anatomically related markers are connected using predefined body-structure edges.

The model therefore operates on:

```text
100 × 39 × 3
```

where the graph represents spatial relationships between body markers and the temporal dimension represents movement over time.

Architecture:

```text
Input
100 × 39 × 3
      ↓
Graph convolution
      ↓
Temporal convolution
      ↓
Graph-temporal block
      ↓
Graph-temporal block
      ↓
Graph-temporal block
      ↓
Global average pooling
      ↓
Fully connected layer
      ↓
5 technique classes
```

The ST-GCN explicitly models both:

- spatial relationships between body markers
- temporal movement patterns

---

# Training Configuration

The primary experiments use:

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Epochs | 100 |
| Batch size | 16 |
| Primary input | 100 × 39 × 3 |
| Classes | 5 |
| Loss | Cross-Entropy Loss |
| Optimizer | Adam |
| LSTM learning rate | 5 × 10⁻⁴ |
| LSTM weight decay | 1 × 10⁻⁴ |
| ST-GCN learning rate | 1 × 10⁻³ |
| ST-GCN weight decay | Not used |
| Gradient clipping | 1.0 |
| Training augmentation | Enabled |
| Validation augmentation | Disabled |
| Test augmentation | Disabled |
| Velocity channels | Disabled |
| Model selection | Best validation Macro-F1 |

The final test set is evaluated only after selecting the checkpoint with the best validation Macro-F1.

---

# Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1

Macro-F1 is used as the primary metric because the experiment contains multiple technique classes and the metric gives equal importance to each class.

---

# Final Results

Both models were evaluated on the same 264-sample subject-independent test set.

| Metric | LSTM | ST-GCN |
|---|---:|---:|
| Accuracy | 87.50% | **95.08%** |
| Macro Precision | 88.07% | **95.36%** |
| Macro Recall | 87.45% | **95.08%** |
| Macro-F1 | 87.43% | **95.08%** |

The difference between the two models was:

| Metric | ST-GCN minus LSTM |
|---|---:|
| Accuracy | +7.58 percentage points |
| Macro Precision | +7.29 percentage points |
| Macro Recall | +7.63 percentage points |
| Macro-F1 | +7.65 percentage points |

Under the evaluated subject-independent split and configuration, the ST-GCN achieved higher test performance than the LSTM baseline.

These results provide experimental evidence supporting the research question for this dataset and evaluation protocol.

They should not be interpreted as proof that ST-GCN will outperform LSTM on every karate movement dataset or every subject-independent evaluation setting.

---

# Per-Class Results

## Gyaku-Zuki

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| LSTM | 79.10% | 98.15% | 87.60% |
| ST-GCN | 100.00% | 96.30% | 98.11% |

## Mae-Geri

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| LSTM | 88.89% | 90.57% | 89.72% |
| ST-GCN | 89.66% | 98.11% | 93.69% |

## Mawashi-Geri gedan

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| LSTM | 86.96% | 76.92% | 81.63% |
| ST-GCN | 100.00% | 92.31% | 96.00% |

## Mawashi-Geri jodan

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| LSTM | 85.42% | 77.36% | 81.19% |
| ST-GCN | 95.92% | 88.68% | 92.16% |

## Ushiro-Mawashi-Geri

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| LSTM | 100.00% | 94.23% | 97.03% |
| ST-GCN | 91.23% | 100.00% | 95.41% |

---

# Confusion Matrices

The repository contains the generated confusion matrices:

```text
outputs/figures/lstm_confusion_matrix.png
outputs/figures/st-gcn_confusion_matrix.png
```

The numerical classification reports are also stored as JSON files.

---

# Reproducibility

The repository contains the source code required to reproduce the preprocessing and model experiments.

## 1. Clone the repository

```bash
git clone https://github.com/ShivaayaPangasa/Karate-Technique-Recognition-STGCN.git
cd Karate-Technique-Recognition-STGCN
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scriptsctivate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Obtain the dataset

Download the original C3D dataset from the Figshare source and place the raw data in:

```text
data/raw/
```

The raw dataset is intentionally not included in this repository.

## 5. Build the dataset manifest

```bash
python -m src.dataset.build_manifest
```

## 6. Apply dataset corrections

```bash
python -m src.dataset.apply_corrections
```

## 7. Normalize the manifest

```bash
python -m src.dataset.normalize_manifest
```

## 8. Create the subject-independent split

```bash
python -m src.dataset.split_manifest
```

## 9. Build the processed dataset

```bash
python -m src.dataset.build_dataset
```

## 10. Verify the processed dataset

```bash
python -m src.dataset.verify_dataset
```

The verification script checks:

- five technique classes
- four execution conditions
- subject-disjoint train/validation/test splits
- processed labels
- expected tensor shape
- finite processed values

## 11. Train the LSTM baseline

```bash
python -m src.models.train_lstm
```

## 12. Train the ST-GCN

```bash
python -m src.models.train_stgcn
```

## 13. Compare both models

```bash
python -m src.evaluation.compare_models
```

The comparison script generates:

```text
outputs/lstm_classification_report.json
outputs/st-gcn_classification_report.json
outputs/lstm_vs_stgcn_comparison.json
outputs/figures/lstm_confusion_matrix.png
outputs/figures/st-gcn_confusion_matrix.png
```

---

# Repository Structure

```text
Karate-Technique-Recognition-STGCN/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── dataset/
│   │   ├── apply_corrections.py
│   │   ├── build_dataset.py
│   │   ├── build_manifest.py
│   │   ├── marker_filter.py
│   │   ├── normalize_manifest.py
│   │   ├── preprocess.py
│   │   ├── split_manifest.py
│   │   └── verify_dataset.py
│   │
│   ├── evaluation/
│   │   └── compare_models.py
│   │
│   └── models/
│       ├── dataset.py
│       ├── graph.py
│       ├── lstm_model.py
│       ├── stgcn_model.py
│       ├── train_lstm.py
│       └── train_stgcn.py
│
└── outputs/
    ├── figures/
    │   ├── lstm_confusion_matrix.png
    │   └── st-gcn_confusion_matrix.png
    │
    ├── lstm_classification_report.json
    ├── lstm_metrics.json
    ├── lstm_vs_stgcn_comparison.json
    ├── st-gcn_classification_report.json
    └── stgcn_metrics.json
```

---

# Limitations

The experiment has several limitations:

1. The evaluation uses one deterministic subject-independent train/validation/test split.
2. The dataset contains a limited set of five selected karate techniques.
3. The dataset comes from controlled optical motion-capture recordings rather than unconstrained real-world video.
4. The ST-GCN graph topology is manually defined using anatomical marker relationships.
5. The experiment does not establish generalization to other martial arts, datasets, sensor configurations, or real-time camera input.
6. The dataset is structured compared with real-world sports environments.
7. The reported performance should therefore be interpreted within the scope of the dataset and experimental protocol.

---

# Future Work

Potential future research directions include:

- evaluation using additional independent datasets
- cross-dataset generalization
- larger and more diverse athlete populations
- real-time pose-estimation input
- adaptive graph topology
- velocity and acceleration representations
- explainability and body-region analysis
- robustness to missing or occluded body markers
- deployment for real-time technique feedback

These extensions are outside the scope of the primary experiment reported in this repository.

---

# Academic Scope

This repository is developed as part of the AIML305 Advanced Deep Neural Networks academic project.

The primary research contribution investigated here is the comparison between:

```text
Sequence-only LSTM
        vs
Topology-aware ST-GCN
```

for subject-independent five-class karate technique recognition.

---

# Citation and Attribution

If using the underlying dataset, please cite the original publication and follow its licensing and attribution requirements.

Dataset publication:

```text
Optical motion capture dataset of selected techniques
in beginner and advanced Kyokushin karate athletes.
Scientific Data, 2021.
https://doi.org/10.1038/s41597-021-00801-5
```

Dataset:

```text
https://doi.org/10.6084/m9.figshare.13164848
```

---

# License

The source code in this repository is provided for academic and research purposes.

The underlying motion-capture dataset is a separate third-party resource and is subject to the license and attribution requirements of its original publication and distribution record.

Users should review the original dataset terms before redistribution or commercial use.
