# --- 1. SYSTEM SETUP & IMPORTS ---
import os, sys, re, time, requests, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from datasets import load_dataset as hf_load_dataset

# GROQ CONFIG
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

# --- 2. THE MASTER PROMPT (CoT + Few-Shot) ---
def get_evasion_logic_prompt(question, answer):
    return f"""
    You are an expert Political Scientist specializing in the QEvasion taxonomy.
    
    TAXONOMY (Responses can match multiple categories):
    - Explicit: Direct, clear, full answer.
    - Implicit: Answers partially, omits controversial core.
    - General: Vague high-level, no specifics.
    - Partial/half-answer: Answers only one part of multi-part question.
    - Dodging: Non-responsive filler or procedural.
    - Deflection: Pivots to different topic.
    - Declining to answer: Explicit refusal.
    - Claims ignorance: Says doesn't know/hasn't seen.
    - Clarification: Stalls by asking repeat.

    EXAMPLES:
    Q: Do you support the policy? A: That's a great question, but let's talk about jobs.
    Reasoning: Pivots topic without addressing premise.
    Labels: Deflection, Dodging

    Q: Will you release taxes? A: I won't answer that.
    Reasoning: Explicit refusal to provide information.
    Labels: Declining to answer

    NOW ANALYZE:
    QUESTION: {question}
    ANSWER: {answer}

    OUTPUT FORMAT (strict):
    Reasoning: [1 sentence analysis]
    Labels: [comma-separated list, e.g., "Deflection, Dodging" or "Explicit"]
    """

# --- 3. ROBUST ENSEMBLE CALLER ---
def call_groq(prompt):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
    except: pass
    return "Labels: Explicit"

def get_ensemble_prediction(question, answer, calls=3):
    prompt = get_evasion_logic_prompt(question, answer)
    votes = []
    for _ in range(calls):
        raw = call_groq(prompt)
        match = re.search(r"Labels?:\s*(.+)", raw, re.IGNORECASE)
        if match:
            # Normalize and split
            labels = [l.strip().lower().strip('.') for l in match.group(1).split(',')]
            votes.append(labels)
    
    # Majority Vote Logic
    all_votes = [lbl for sublist in votes for lbl in sublist]
    count = Counter(all_votes)
    # Threshold: Must appear in more than half the calls
    final = [lbl for lbl, c in count.items() if c >= (calls // 2 + 1)]
    return final if final else ["explicit"]

# --- 4. CANONICAL NORMALIZATION ---
CANONICAL_MAP = {
    'explicit': 'Explicit', 'implicit': 'Implicit', 'general': 'General', 
    'partial': 'Partial/half-answer', 'dodging': 'Dodging', 'deflection': 'Deflection',
    'declining': 'Declining to answer', 'ignorance': 'Claims ignorance', 'clarification': 'Clarification'
}

def normalize_labels(labels):
    normalized = []
    for l in labels:
        for key, canon in CANONICAL_MAP.items():
            if key in l.lower():
                normalized.append(canon)
                break
    return list(set(normalized)) if normalized else ["Explicit"]

# --- 5. DATA LOADING & EXECUTION ---
print(">>> [System] Loading Dataset...")
dataset = hf_load_dataset("ailsntua/QEvasion")
test_df = pd.DataFrame(dataset['test'])

print(">>> [LLM] Starting Ensemble Reasoning (3 calls per sample)...")
# For speed in testing, we use 30 samples. Set to len(test_df) for full run.
sample_size = min(30, len(test_df)) 
test_sample = test_df.sample(sample_size).copy()

tqdm.pandas()
test_sample['raw_ensemble'] = test_sample.progress_apply(
    lambda x: get_ensemble_prediction(x['interview_question'], x['interview_answer']), axis=1
)
test_sample['final_multi_labels'] = test_sample['raw_ensemble'].apply(normalize_labels)

# --- 6. HIERARCHY MAPPING (9-Class to 3-Class) ---
def map_to_3class(labels):
    # Ambivalent (Evasive but not a refusal)
    if any(l in labels for l in ['Implicit', 'General', 'Partial/half-answer', 'Dodging', 'Deflection']):
        return 1
    # Clear Non-Reply (Hard refusal or stall)
    elif any(l in labels for l in ['Declining to answer', 'Claims ignorance', 'Clarification']):
        return 2
    # Clear Reply
    return 0

test_sample['pred_clarity'] = test_sample['final_multi_labels'].apply(map_to_3class)

# Assuming 'clarity_label' exists in dataset for comparison
le_clarity = LabelEncoder().fit(['Clear Reply', 'Ambivalent', 'Clear Non-Reply'])
test_sample['true_clarity'] = test_sample['clarity_label'].apply(
    lambda x: 0 if 'Clear Reply' in x else (1 if 'Ambivalent' in x else 2)
)

# --- 7. FINAL METRICS ---
final_f1 = f1_score(test_sample['true_clarity'], test_sample['pred_clarity'], average='macro')
print(f"\n" + "="*40)
print(f" FINAL HYBRID ENSEMBLE F1: {final_f1:.4f}")
print(f" Improvement over 0.551: +{(final_f1 - 0.551)*100:.1f}%")
print("="*40)

# Visualization
plt.figure(figsize=(10,8))
cm = confusion_matrix(test_sample['true_clarity'], test_sample['pred_clarity'])
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=['Clear', 'Ambivalent', 'Non-Reply'], 
            yticklabels=['Clear', 'Ambivalent', 'Non-Reply'])
plt.title(f"Llama-3.1 Ensemble Reasoning (F1: {final_f1:.3f})")
plt.savefig("master_ensemble_final.pdf")
print(">>> [Success] Final report saved as 'master_ensemble_final.pdf'")

*** groq llm** 
