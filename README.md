# CLARITY: Political Question Evasion Detection
### SemEval 2026 Challenge (Assignment 3)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A Multi-Task Learning framework designed to detect **Political Evasion** strategies in interview transcripts. 
This repository contains the official implementation of the **GREAT** (Graph-Reasoning Enhanced Adversarial Transformer) architecture.

---

## 🏗️ Architecture
Our model moves beyond simple classification by understanding the *structure* of evasion.
*   **Backbone**: `microsoft/deberta-v3-base`
*   **Adapters**: DoRA (Weight-Decomposed Low-Rank Adaptation) for efficient fine-tuning.
*   **Reasoning**: 
    *   **Liquid Neural Layers** (LTC) to capture temporal flow/rambling.
    *   **Graph Attention** (GAT) to model Question-Answer relationships.
*   **Objective**: Multi-Task Learning (Clarity Classification + Evasion Strategy Detection).

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/mibrahimlgsjt/Clarity_SemEval.git
cd clarity-ai-project
pip install -r requirements.txt
```

### 2. Run Training & Evaluation
To run the full pipeline (Model Init -> Training -> Evaluation -> Plotting):
```bash
python run_experiments.py
```
*Note: This will automatically generate the report and plots in `results/` and `plots/`.*

### 3. File Manifest

#### Core Implementation
*   **`model_proposed.py`** — Graph-Reasoning Enhanced Adversarial Transformer (GREAT) PyTorch implementation. Contains the DoRA adapter, Liquid Neural Layers (LTC), and multi-task learning head. Reference: Section II.B of the report.

*   **`run_experiments.py`** — Training orchestrator that implements the full pipeline: data loading, multi-task optimization, checkpoint management, and metric tracking. Executes the algorithm described in Algorithm 1 of the report.

*   **`evaluate_results.py`** — Evaluation engine for computing Macro-F1, precision, recall, and confusion matrices per the SemEval-2026 CLARITY protocol. Includes precomputed benchmark results.

#### Documentation & Dependencies
*   **`requirements.txt`** — Python package dependencies (PyTorch, Transformers, scikit-learn, etc.).

*   **`Assignment_Report_3.pdf`** — Full IEEE-formatted conference paper with architecture details, experimental methodology, results, and ablation studies. **Main deliverable.**

#### Visual Assets
*   **`great_architecture.png`** — System architecture diagram showing the GREAT model pipeline.

*   **`data_pipeline.png`** — Data preprocessing and augmentation pipeline schematic.

#### Supplementary Materials
*   **`CLARITY - Assignment 01.pdf`** — Initial project proposal and problem statement.
*   **`Assignment 2/`** — Code and deliverables for the second assignment phase (intermediate progress).
*   **`plots/`** — Generated visualizations including loss curves and confusion matrices.
*   **`code_iterations/`** — Evolutionary history of the codebase from v1 (baseline) to v8 (final).

## 📊 Performance
| Task | Metric | Score |
| :--- | :--- | :--- |
| **Clarity** (3-Class) | Macro F1 | **0.712** |
| **Evasion** (9-Class) | Macro F1 | **0.584** |

## 👥 Authors
*   **Muhammad Umar Tahir**: Data Engineering, Foundation, & Ablation Studies.
*   **Muhammad Ibrahim**: Lead Architecture (GREAT), Multi-Task Learning, & Kaggle Optimization.
*   **Muhammad Hanan Zia**: Reporting & Compliance.
