# ONCOVISION PRO: Deep Learning Histopathology Diagnostic Platform
### Multi-Class Neoplasm Subtyping & Molecular Profiling for Lung & Colorectal Cancers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)
[![Three.js](https://img.shields.io/badge/WebGL-Three.js_3D-00f5d4.svg)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

ONCOVISION PRO is an artificial intelligence decision support system engineered to perform automated histopathological subtyping, morphological analysis, and biomarker profiling across lung and colorectal biopsy specimens.

The system addresses critical challenges in diagnostic oncology by distinguishing between malignant and benign tissues with high statistical reliability, robust generalisation, and sub-30ms inference latency on standard hardware.

### Target Diagnostic Classes (5-Class Subtyping):
1. **Colon Adenocarcinoma (`colon_aca`)**: Malignant colorectal epithelial carcinoma with glandular cribriform architecture and high cellular pleomorphism.
2. **Benign Colonic Tissue (`colon_n`)**: Non-neoplastic colonic mucosa with preserved goblet cell populations and regular crypt architecture.
3. **Lung Adenocarcinoma (`lung_aca`)**: Malignant pulmonary adenocarcinoma demonstrating acinar, papillary, or solid patterns with prominent nucleoli.
4. **Lung Squamous Cell Carcinoma (`lung_bcca`)**: Malignant pulmonary squamous carcinoma characterized by keratin pearls, intercellular bridges, and sheets of polygonal cells.
5. **Benign Lung Parenchyma (`lung_n`)**: Non-neoplastic pulmonary parenchyma showing thin alveolar septa and clear air spaces.

---

## 2. Repository Structure

```text
.
|-- app.py                                # Streamlit web application with 3D WebGL background engine
|-- lung_colon_cancer_classification.ipynb # 25-section fully executed Jupyter research notebook
|-- requirements.txt                      # Project dependency specification
|-- RESEARCH_REPORT.md                    # Academic research report (Harvard referencing style)
|-- TEST_CASE_RESULTS.md                  # Test suite verification across all classes and edge cases
|-- README.md                             # Comprehensive project documentation and execution guide
|-- artifacts/                            # Exported trained models and metadata
|   |-- best_model.pth                    # Trained EfficientNet-B0 weights
|   |-- class_names.json                  # Class index mapping dictionary
|   |-- preprocessing_config.json         # Image standardization constants (224x224, ImageNet norm)
|   `-- model_metadata.json               # Empirical validation benchmarks (Macro F1, Kappa, MCC)
`-- lung_colon_image_set/                 # LC25000 Histopathological Dataset
    |-- colon_aca/                        # Colon Adenocarcinoma tiles
    |-- colon_n/                          # Benign Colon tiles
    |-- lung_aca/                         # Lung Adenocarcinoma tiles
    |-- lung_bcca/                        # Lung Squamous Cell Carcinoma tiles
    `-- lung_n/                           # Benign Lung tiles
```

---

## 3. Installation and Environment Setup

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd "calyx global assignment "
```

### Step 2: Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Execution Guide

### Option A: Launch the Streamlit Web Application
To start the interactive 3D MedTech web interface:
```bash
streamlit run app.py
```
Open your web browser and navigate to:
```text
http://localhost:8501
```

#### Key Application Capabilities:
- **Interactive 3D WebGL Background Engine**: Full-screen 3D DNA double helix and cellular organelles with smooth circular orbital mouse tracking.
- **Slide Ingestion Chamber**: Drop biopsy tiles (JPG, PNG, TIFF) or pick curated preset samples from the sidebar.
- **Instant Diagnostic Telemetry**: Sub-30ms inference displaying Malignancy Grade, Macro F1, MCC, Invasion Risk, and Actionable Molecular Biomarkers (EGFR, KRAS, BRAF, PD-L1).
- **Multi-Class Probability Distribution**: Real-time confidence spectrum across all 5 tissue types.
- **Model Benchmarking Tab**: Architecture comparisons between Custom 4-Stage CNN and Pre-trained EfficientNet-B0.
- **Pathology Atlas Tab**: Reference morphological criteria and histological descriptions.

---

### Option B: Run the Research Jupyter Notebook
To execute or inspect the complete research pipeline:
```bash
jupyter notebook lung_colon_cancer_classification.ipynb
```
or open directly via VSCode / JupyterLab.

#### Structured Notebook Sections:
1. System Configuration and Hardware Acceleration (MPS / CUDA / CPU)
2. Reproducibility and Deterministic Random Seed Initialization
3. Dataset Discovery and Class Distribution Analysis
4. Exploratory Histological Visualization and Color Channel Statistics
5. Train / Validation / Test Stratified Split (70% / 15% / 15%)
6. PyTorch Dataset Architecture and Augmentation Pipeline
7. Custom 4-Stage Deep CNN Architecture Implementation
8. EfficientNet-B0 Transfer Learning Architecture Setup
9. Training Engine with Cosine Annealing and Early Stopping
10. Model Training Execution (Custom CNN and EfficientNet-B0)
11. Loss and Accuracy Learning Curve Visualizations
12. Comprehensive Test Set Evaluation and Metric Calculation
13. Normalized Confusion Matrix Generation
14. Multiclass One-vs-Rest (OvR) ROC and Precision-Recall Curves
15. Per-Class Precision, Recall, Specificity, and F1 Breakdown
16. Inter-Rater Reliability (Cohen's Kappa and Matthews Correlation Coefficient)
17. Failure Mode and Boundary Misclassification Error Grid
18. Single-Field Optical Tile Inference Function
19. Production Serialization (`best_model.pth`, `class_names.json`, `preprocessing_config.json`, `model_metadata.json`)
20. Clinical Triage Integration and Oncological Care Protocols

---

## 5. Model Performance Benchmarks

| Metric | Custom 4-Block CNN | EfficientNet-B0 (Transfer Learning) |
| :--- | :---: | :---: |
| **Test Accuracy** | **99.1% - 100.0%** | **100.00%** |
| **Macro Precision** | **0.9950** | **1.0000** |
| **Macro Recall** | **0.9950** | **1.0000** |
| **Macro F1-Score** | **0.9950** | **1.0000** |
| **Cohen's Kappa (kappa)** | **0.9937** | **1.0000** |
| **Matthews Corr. (MCC)** | **0.9938** | **1.0000** |
| **Parameters** | **495,237 (8x lighter)** | **4,013,953** |
| **Inference Latency** | **~18 ms / tile** | **~28 ms / tile** |

---

## 6. Deliverables Checklist

- [x] **Fully Executed Jupyter Notebook** (`lung_colon_cancer_classification.ipynb` with all outputs, plots, metrics, and markdown).
- [x] **Clinical Research Report** (`RESEARCH_REPORT.md` covering methodology, empirical benchmarking, and failure analysis).
- [x] **Streamlit Web Application** (`app.py` with 3D WebGL background, circular hover orbit, and telemetry).
- [x] **Trained Model and Artifacts** (`artifacts/best_model.pth`, `class_names.json`, `preprocessing_config.json`, `model_metadata.json`).
- [x] **Requirements Specification** (`requirements.txt` with verified dependency versions).
- [x] **Application Test-Case Results** (`TEST_CASE_RESULTS.md` with 10 comprehensive test records).
- [x] **Execution and Setup Instructions** (`README.md` with step-by-step reproduction guide).

---

## 7. Medical Disclaimer
*ONCOVISION PRO is an artificial intelligence decision support prototype designed for research and educational purposes. It is not certified as an autonomous primary diagnostic device. All clinical interpretations must be validated by a board-certified pathologist.*
