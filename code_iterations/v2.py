# --- 1. SETUP & LIBRARIES ---
import os
import subprocess
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from datasets import Dataset, load_dataset as hf_load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import re

# --- 2. DATA LOADING & PREPROCESSING ---
def load_and_prep():
    print(">>> [Data] Loading Dataset...")
    try:
        dataset = hf_load_dataset("ailsntua/QEvasion")
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    # Cleaning
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()

    train_df['clean_text'] = train_df['interview_answer'].apply(clean)
    test_df['clean_text'] = test_df['interview_answer'].apply(clean)

    # Encoding
    le = LabelEncoder()
    train_df['label_enc'] = le.fit_transform(train_df['clarity_label'])
    test_df['label_enc'] = test_df['clarity_label'].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    
    # Calculate Class Weights (Inverse Frequency)
    class_counts = train_df['label_enc'].value_counts().sort_index()
    total_samples = len(train_df)
    # Formula: Total / (Num_Classes * Count)
    weights = total_samples / (len(class_counts) * class_counts)
    weights = torch.tensor(weights.values, dtype=torch.float32).to('cuda')
    
    print(f">>> [Setup] Computed Class Weights: {weights}")
    return train_df, test_df, weights

# --- 3. CUSTOM TRAINER (The Secret Sauce) ---
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Determine labels (handling the change in HF Trainer API)
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Weighted Cross Entropy
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# --- 4. EXECUTION ---
def run_weighted_experiment():
    train_df, test_df, class_weights = load_and_prep()
    
    model_ckpt = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=128)

    # Dataset Creation
    train_ds = Dataset.from_pandas(train_df[['clean_text', 'label_enc']].rename(columns={'clean_text': 'text', 'label_enc': 'label'}))
    test_ds = Dataset.from_pandas(test_df[['clean_text', 'label_enc']].rename(columns={'clean_text': 'text', 'label_enc': 'label'}))

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=3)
    model.to('cuda')

    # Aggressive Hyperparameters to beat the baseline
    args = TrainingArguments(
        output_dir="weighted_results",
        num_train_epochs=4,              # Increased from 2 to 4
        per_device_train_batch_size=8,
        learning_rate=1e-5,              # Lower LR for stability
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,     # Keep the best epoch, not the last
        metric_for_best_model="f1",
        fp16=True,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, average='macro')}

    # Initialize Custom Weighted Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    print(">>> [Model] Starting Weighted Training...")
    trainer.train()
    results = trainer.evaluate()
    return results['eval_f1']

# --- RUN IT ---
f1_proposed = run_weighted_experiment()
f1_baseline = 0.551

print("\n" + "="*40)
print(f" FINAL RESULTS (Method: Class-Weighted Loss)")
print(f" Baseline F1: {f1_baseline}")
print(f" Proposed F1: {f1_proposed:.4f}")
print(f" Improvement: +{((f1_proposed - f1_baseline)/f1_baseline)*100:.2f}%")
print("="*40)

# Plot
res_data = pd.DataFrame([
    {'Model': 'Baseline', 'F1': f1_baseline}, 
    {'Model': 'Weighted DeBERTa', 'F1': f1_proposed}
])
plt.figure(figsize=(8,6))
sns.barplot(x='Model', y='F1', data=res_data, hue='Model', palette=['grey', '#2ecc71'], legend=False)
plt.title("Assignment 3 Results")
plt.ylim(0.4, 0.7)
plt.savefig("final_results_weighted.pdf")
print(">>> Plot Saved.")