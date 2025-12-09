# --- 1. SETUP & DEFENSIVE IMPORTS ---
import os
import sys
import shutil
import re
import time
import requests
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, get_scheduler
from datasets import Dataset, load_dataset as hf_load_dataset

# SETTINGS & KEYS
GROQ_API_KEY = "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"
MODEL_CKPT = "microsoft/deberta-v3-base"

# --- 2. PhD FEATURE: FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

class MasterTrainer(Trainer):
    def __init__(self, focal_alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = FocalLoss(alpha=focal_alpha)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# --- 3. DATA SCRUBBER ---
def get_calibrated_data():
    ds = hf_load_dataset("ailsntua/QEvasion")
    train_df = pd.DataFrame(ds['train'])
    test_df = pd.DataFrame(ds['test'])

    def clean(t): 
        return re.sub(r'\s+', ' ', str(t).lower().replace('[inaudible]', '')).strip()

    for df in [train_df, test_df]:
        df['text'] = df['interview_answer'].apply(clean)
        df['evasion_label'] = df['evasion_label'].fillna("None").astype(str).str.strip()

    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([train_df['evasion_label'], test_df['evasion_label']]))
    le.fit(all_labels)
    train_df['label'] = le.transform(train_df['evasion_label'])
    test_df['label'] = le.transform(test_df['evasion_label'])
    return train_df, test_df, le

# --- 4. THE EXECUTION PIPELINE ---
def run_hybrid_engine(n_samples=5):
    print(f"\n>>> [System] Starting Pipeline (N={n_samples})...")
    train_df, test_df, le = get_calibrated_data()
    eval_df = test_df.head(n_samples).copy()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> [System] Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    train_ds = Dataset.from_pandas(train_df).map(lambda x: tokenizer(x['text'], truncation=True, max_length=128), batched=True).rename_column("label", "labels")
    test_ds = Dataset.from_pandas(eval_df).map(lambda x: tokenizer(x['text'], truncation=True, max_length=128), batched=True).rename_column("label", "labels")
    
    # Compute class weights, handling missing classes
    unique_train_labels = np.unique(train_df['label'])
    present_weights = compute_class_weight('balanced', classes=unique_train_labels, y=train_df['label'])
    full_weights = torch.zeros(len(le.classes_), dtype=torch.float)
    for idx, cls in enumerate(unique_train_labels):
        full_weights[cls] = present_weights[idx]
    weights = full_weights.to(device)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=len(le.classes_)).to(device)
    
    # Differential Learning Rates
    params = [
        {"params": model.deberta.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4}
    ]
    optimizer = torch.optim.AdamW(params)
    
    args = TrainingArguments(
        output_dir="master_out",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy="no",
        fp16=(device == "cuda"),  # Enable mixed precision if CUDA
        report_to="none"
    )
    
    num_training_steps = args.num_train_epochs * (len(train_ds) // args.per_device_train_batch_size + 1)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    trainer = MasterTrainer(
        focal_alpha=weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        optimizers=(optimizer, scheduler),
        data_collator=DataCollatorWithPadding(tokenizer)
    )
    
    trainer.train()
    
    # BERT Inference
    bert_outputs = trainer.predict(test_ds)
    bert_probs = torch.softmax(torch.tensor(bert_outputs.predictions), dim=-1)
    final_9_labels = []
    print(f">>> [Hybrid] Running Groq judicial review for low-confidence cases...")
    for i, (idx, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df))):
        conf, pred = torch.max(bert_probs[i], dim=0)
        if conf < 0.75:
            try:
                prompt = f"Classify the answer to the question as one of the following evasion labels: {', '.join(le.classes_)}. Output only the label.\n\nQ: {row['interview_question']}\nA: {row['interview_answer']}"
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "temperature": 0},
                    timeout=10
                )
                out = r.json()['choices'][0]['message']['content'].strip()
                if out in le.classes_:
                    final_9_labels.append(out)
                else:
                    final_9_labels.append(le.inverse_transform([pred.item()])[0])
            except Exception as e:
                print(f">>> [Error] Groq call failed for sample {i}: {e}")
                final_9_labels.append(le.inverse_transform([pred.item()])[0])
        else:
            final_9_labels.append(le.inverse_transform([pred.item()])[0])
    
    # 3-CLASS MAPPING
    def map_3(name):
        if name == 'Explicit': return "Clear Reply"
        if name in ['Declining to answer', 'Claims ignorance', 'Clarification']: return "Clear Non-Reply"
        return "Ambivalent"
    
    eval_df['Pred_9_Category'] = final_9_labels
    eval_df['Pred_3_Clarity'] = [map_3(n) for n in final_9_labels]
    
    # Evaluation Metrics
    true_9_labels = eval_df['evasion_label']
    print("\n>>> [Metrics] 9-Class Classification Report:")
    print(classification_report(true_9_labels, eval_df['Pred_9_Category'], zero_division=0))
    
    true_3_labels = [map_3(n) for n in true_9_labels]
    print(">>> [Metrics] 3-Class Classification Report:")
    print(classification_report(true_3_labels, eval_df['Pred_3_Clarity'], zero_division=0))
    
    # CSV EXPORT
    output_file = f"hybrid_predictions_n{n_samples}.csv"
    eval_df[['interview_question', 'interview_answer', 'Pred_9_Category', 'Pred_3_Clarity']].to_csv(output_file, index=False)
    print(f">>> [Export] Saved {n_samples} results to '{output_file}'")
    return True

# --- AUTOMATIC PROCEED ---
if run_hybrid_engine(n_samples=5):
    print("\n" + "="*50 + "\nPROBE SUCCESSFUL. RUNNING MASTER N=100 DATASET\n" + "="*50)
    run_hybrid_engine(n_samples=100)