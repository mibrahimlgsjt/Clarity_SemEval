# CLARITY: Political Question Evasion Detection
## SemEval 2026 Challenge - Assignment 2

### Project Overview
This repository contains the baseline implementations for the CLARITY task. We implemented a modular pipeline to classify answers into "Clear Reply", "Clear Non-Reply", and "Ambivalent".

### Structure
* `src/`: Source code for data loading, preprocessing, and models.
* `plots/`: Generated confusion matrices and F1 comparison charts.
* `requirements.txt`: Python dependencies.

### How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python src/train_eval.py`

### Baselines Implemented
1. **Baseline A**: TF-IDF + Logistic Regression (Statistical lower bound).
2. **Baseline B**: Bi-LSTM (Neural baseline).
3. **Baseline C**: BERT-base (Transformer/State-of-the-art).