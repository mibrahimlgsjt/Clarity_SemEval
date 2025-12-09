# --- 1. INSTALLATION & SETUP (Auto-Runs First) ---
import os
import subprocess
import sys

# Force install nlpaug inside the script to prevent "ModuleNotFound"
print(">>> [Setup] Installing libraries... (This takes ~1 min)")
subprocess.check_call([sys.executable, "-m", "pip", "install", "nlpaug", "transformers", "datasets", "scikit-learn", "seaborn", "torch", "nltk"])

# Download NLTK data required by nlpaug
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

print(">>> [Setup] Libraries installed successfully.")

# --- 2. IMPORTS & DATA LOADING ---
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from datasets import Dataset, load_dataset as hf_load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import nlpaug.augmenter.word as naw
import re
from sklearn.preprocessing import LabelEncoder

# --- 3. HELPER FUNCTIONS (Pre-written for you) ---

def load_data():
    """Loads the QEvasion dataset from Hugging Face."""
    print(">>> [Data] Downloading dataset...")
    try:
        dataset = hf_load_dataset("ailsntua/QEvasion")
        return pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

class TextPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
    def clean_text_func(self, text):
        text = str(text).lower()
        # Fix: Raw string for regex to avoid SyntaxWarning
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'http\S+', '', text)
        return text.strip()
    def process_pipeline(self, train_df, test_df):
        print(">>> [Data] Cleaning and Encoding...")
        train_df['clean_text'] = train_df['interview_answer'].apply(self.clean_text_func)
        test_df['clean_text'] = test_df['interview_answer'].apply(self.clean_text_func)
        
        # Target: Clarity Label (3 classes)
        train_df['label_enc'] = self.label_encoder.fit_transform(train_df['clarity_label'])
        test_df['label_enc'] = test_df['clarity_label'].apply(
            lambda x: self.label_encoder.transform([x])[0] if x in self.label_encoder.classes_ else -1
        )
        return train_df, test_df, len(self.label_encoder.classes_)

# --- 4. THE PROPOSED MODEL (Augmented DeBERTa) ---

def run_experiment_full(train_df, test_df, num_classes):
    print("\n" + "="*50)
    print(" RUNNING ASSIGNMENT 3: Augmented DeBERTa (GPU)")
    print("="*50)

    # A. Augmentation Step
    print(f"   [Augmenter] Identifying minority classes...")
    class_counts = train_df['label_enc'].value_counts()
    minority_label = class_counts.idxmin() # Usually 'Clear Non-Reply'
    minority_df = train_df[train_df['label_enc'] == minority_label].copy()
    
    print(f"   [Augmenter] Loading Synonym Replacer...")
    # Using 'distilbert' for speed on Kaggle
    aug = naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased', 
        action="substitute", 
        aug_p=0.3, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"   [Augmenter] Synthesizing {len(minority_df)} new examples...")
    new_rows = []
    for text in minority_df['clean_text']:
        try:
            aug_text = aug.augment(text)
            # Handle list output from nlpaug
            aug_text = aug_text[0] if isinstance(aug_text, list) else aug_text
            new_rows.append(aug_text)
        except:
            new_rows.append(text)
            
    aug_data = pd.DataFrame({
        'clean_text': new_rows,
        'label_enc': [minority_label] * len(new_rows)
    })
    
    # Combine original + augmented data
    train_df_aug = pd.concat([train_df, aug_data]).sample(frac=1).reset_index(drop=True)
    print(f"   [Augmenter] Dataset size increased: {len(train_df)} -> {len(train_df_aug)}")

    # B. Model Setup
    model_ckpt = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=128)

    # Convert to Hugging Face Dataset format
    train_ds = Dataset.from_pandas(train_df_aug[['clean_text', 'label_enc']].rename(columns={'clean_text': 'text', 'label_enc': 'label'}))
    test_ds = Dataset.from_pandas(test_df[['clean_text', 'label_enc']].rename(columns={'clean_text': 'text', 'label_enc': 'label'}))

    print(">>> [Model] Tokenizing...")
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_classes)
    
    if torch.cuda.is_available():
        model.to('cuda')
        print(">>> [System] GPU Detected: Training on CUDA.")
    else:
        print(">>> [System] WARNING: CPU only. This will be slow.")

    # Training Config
    args = TrainingArguments(
        output_dir="clarity_model_output",
        num_train_epochs=2,              # Keep it fast (2 epochs is enough for demo)
        per_device_train_batch_size=8,   # Standard for T4 GPU
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",           # The FIX for the error you saw earlier
        save_strategy="no",
        fp16=True if torch.cuda.is_available() else False, # Speed boost
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, average='macro')}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    print(">>> [Model] Starting Training Loop...")
    trainer.train()
    
    print(">>> [Model] Evaluating on Test Set...")
    results = trainer.evaluate()
    return results['eval_f1']

# --- 5. EXECUTION BLOCK ---

if __name__ == "__main__":
    # 1. Load
    train_df, test_df = load_data()
    
    if train_df is not None:
        # 2. Process
        processor = TextPreprocessor()
        train_df, test_df, num_classes = processor.process_pipeline(train_df, test_df)
        
        # 3. Run Proposed
        f1_proposed = run_experiment_full(train_df, test_df, num_classes)
        
        # 4. Compare with Baseline (Value from Paper/Assignment 2)
        f1_baseline = 0.551 
        
        print(f"\n" + "="*30)
        print(f" FINAL RESULTS")
        print(f" Baseline F1: {f1_baseline}")
        print(f" Proposed F1: {f1_proposed:.4f}")
        print(f" Improvement: +{((f1_proposed - f1_baseline)/f1_baseline)*100:.2f}%")
        print("="*30)
        
        # 5. Plot
        res_data = pd.DataFrame([
            {'Model': 'Baseline', 'F1': f1_baseline}, 
            {'Model': 'Proposed (Augmented)', 'F1': f1_proposed}
        ])
        plt.figure(figsize=(8,6))
        sns.barplot(x='Model', y='F1', data=res_data, palette=['grey', 'green'])
        plt.title("Assignment 3: Incremental Improvement")
        plt.ylim(0.4, 0.7)
        plt.savefig("assignment3_result.pdf")
        print(">>> Success! Plot saved as 'assignment3_result.pdf'")
    else:
        print("!!! Critical Error: Could not download dataset.")