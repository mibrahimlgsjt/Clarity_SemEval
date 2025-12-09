from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def run_baseline_A(train_df, test_df):
    """
    BASELINE A: TF-IDF + Logistic Regression
    
    I'm using this as the 'sanity check' baseline. 
    If deep learning can't beat this simple math, we are doing something wrong.
    """
    print("\n" + "="*30)
    print(" RUNNING BASELINE A (TF-IDF)")
    print("="*30)
    
    # 1. Pipeline
    # I decided to use n-grams (1,2) to capture short phrases like "I cannot".
    # max_features=5000 keeps the memory usage low.
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced')) # Handle class imbalance!
    ])
    
    # 2. Train
    print(">>> [Baseline A] Training...")
    pipe.fit(train_df['clean_text'], train_df['label_enc'])
    
    # 3. Predict
    preds = pipe.predict(test_df['clean_text'])
    score = f1_score(test_df['label_enc'], preds, average='macro')
    print(f">>> [Baseline A] Macro F1: {score:.4f}")
    
    # 4. Save Confusion Matrix (Required for report)
    cm = confusion_matrix(test_df['label_enc'], preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Baseline A Confusion Matrix (F1: {score:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/cm_baseline_a.pdf")
    
    return preds, score