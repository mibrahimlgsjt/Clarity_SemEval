# --- 1. CONFIGURATION & SETUP ---
import os, sys, shutil, subprocess, re, time, requests, torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_dataset as hf_load_dataset

# --- TOGGLE THIS FOR FULL RUN ---
USE_FULL_DATASET = True # Set to True to run on the entire dataset
# -------------------------------

GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# CLEANUP DISK
for folder in ["hybrid_master", "logs"]:
    if os.path.exists(folder): shutil.rmtree(folder)

# --- 2. DATA PREP & CLEANING ---
print(">>> [Data] Loading and sanitizing...")
dataset = hf_load_dataset("ailsntua/QEvasion")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

def clean_text(text):
    return re.sub(r'\s+', ' ', str(text).lower().replace('[inaudible]', '')).strip()

for df in [train_df, test_df]:
    df['text'] = df['interview_answer'].apply(clean_text)
    df['evasion_label'] = df['evasion_label'].fillna("None").astype(str).str.strip()
    df = df[df['evasion_label'] != ""].copy()

le = LabelEncoder()
le.fit(np.unique(np.concatenate([train_df['evasion_label'], test_df['evasion_label']])))
train_df['label'] = le.transform(train_df['evasion_label'])
test_df['label'] = le.transform(test_df['evasion_label'])
num_labels = len(le.classes_)

# --- 3. DEBERTA INTUITION LAYER ---
model_ckpt = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True).rename_column("label", "labels")
test_ds = Dataset.from_pandas(test_df[['text', 'label']]).map(tokenize, batched=True).rename_column("label", "labels")

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)

args = TrainingArguments(
    output_dir="hybrid_master", num_train_epochs=3, per_device_train_batch_size=8,
    save_strategy="no", fp16=False, report_to="none", logging_steps=100
)

trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, 
    data_collator=DataCollatorWithPadding(tokenizer)
)

print(">>> [Phase 1] Training DeBERTa backbone...")
trainer.train()

# --- 4. GROQ JUDICIAL LAYER ---
def call_groq_judge(question, answer):
    prompt = f"""
    Analyze if this politician is evading the question.
    QEvasion Categories: {list(le.classes_)}
    
    Q: {question}
    A: {answer}
    
    Reasoning: [1 sentence analysis]
    Label: [One category from the list above]
    """
    try:
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                         json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "temperature": 0}, timeout=10)
        output = r.json()['choices'][0]['message']['content']
        for c in le.classes_:
            if c.lower() in output.lower(): return le.transform([c])[0]
    except: return None
    return None

# --- 5. HYBRID EXECUTION ---
print(f">>> [Phase 2] Running Hybrid Stack (Full Dataset: {USE_FULL_DATASET})")
eval_df = test_df if USE_FULL_DATASET else test_df.head(5)

# Get BERT Probs
bert_outputs = trainer.predict(Dataset.from_pandas(eval_df[['text']]).map(tokenize, batched=True))
bert_probs = torch.softmax(torch.tensor(bert_outputs.predictions), dim=-1)

final_preds = []
for i, (idx, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df))):
    conf, pred = torch.max(bert_probs[i], dim=0)
    
    # TRIGGER LLM if BERT is uncertain (< 75% confidence)
    if conf < 0.75:
        llm_label = call_groq_judge(row['interview_question'], row['interview_answer'])
        final_preds.append(llm_label if llm_label is not None else pred.item())
    else:
        final_preds.append(pred.item())

# --- 6. HIERARCHY MAPPING (9 to 3 Class) ---
def map_to_3(idx):
    name = le.inverse_transform([idx])[0]
    if name == 'Explicit': return 0 # Clear Reply
    if name in ['Declining to answer', 'Claims ignorance', 'Clarification']: return 2 # Non-Reply
    return 1 # Ambivalent

eval_df['pred_3class'] = [map_to_3(p) for p in final_preds]
eval_df['true_3class'] = eval_df['clarity_label'].apply(lambda x: 0 if 'Clear Reply' in x else (1 if 'Ambivalent' in x else 2))

# --- 7. RESULTS ---
f1 = f1_score(eval_df['true_3class'], eval_df['pred_3class'], average='macro')
print(f"\n======== HYBRID F1 SCORE: {f1:.4f} ========")
print(classification_report(eval_df['true_3class'], eval_df['pred_3class'], target_names=['Clear', 'Ambivalent', 'Non-Reply']))

# Plot
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(eval_df['true_3class'], eval_df['pred_3class']), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Clear', 'Ambivalent', 'Non-Reply'], yticklabels=['Clear', 'Ambivalent', 'Non-Reply'])
plt.title(f"Hybrid Solution (N={len(eval_df)})")
plt.tight_layout()
plt.savefig("final_hybrid_check.pdf")
print(">>> Saved 'final_hybrid_check.pdf'")