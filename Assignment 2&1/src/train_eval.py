print(">>> [System] Initializing CLARITY Benchmarking Suite v2.0...", flush=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys

# --- IMPORT ALL MODULES ---
try:
    from data_loader import load_data
    from preprocess import TextPreprocessor
    from model_baseline_A import run_baseline_A
    from model_baseline_B import run_baseline_B
    from model_baseline_C import run_baseline_C
    # NEW: Importing the Advanced Model file
    from model_advanced import run_deberta_model 
except ImportError as e:
    print(f"!!! [System] Critical Import Error: {e}", flush=True)
    sys.exit(1)

def main():
    print("\n-------------------------------------------------", flush=True)
    print("   CLARITY PROJECT: FINAL PIPELINE EXECUTION", flush=True)
    print("-------------------------------------------------", flush=True)
    
    # 1. LOAD
    train_df, test_df = load_data()
    if train_df is None: return

    # 2. PREPROCESS
    processor = TextPreprocessor()
    try:
        train_df, test_df, num_classes = processor.process_pipeline(train_df, test_df)
        if 'label_enc' not in train_df.columns: raise ValueError("Label encoding failed.")
    except Exception as e:
        print(f"!!! [Main] Preprocessing Error: {e}", flush=True)
        return
        
    results_log = []

    # --- MODEL ZOO ---

    # A. TF-IDF
    print("\n>>> [Main] Running Baseline A (Statistical)...", flush=True)
    start = time.time()
    _, f1_a = run_baseline_A(train_df, test_df)
    results_log.append({'Model': 'TF-IDF', 'Type': 'Baseline', 'F1': f1_a})
    print(f"    -> Score: {f1_a:.4f}")

    # B. LSTM
    print("\n>>> [Main] Running Baseline B (Recurrent)...", flush=True)
    start = time.time()
    _, f1_b = run_baseline_B(train_df, test_df, num_classes)
    results_log.append({'Model': 'Bi-LSTM', 'Type': 'Baseline', 'F1': f1_b})
    print(f"    -> Score: {f1_b:.4f}")

    # C. BERT
    print("\n>>> [Main] Running Baseline C (Transformer)...", flush=True)
    start = time.time()
    f1_c = run_baseline_C(train_df, test_df, num_classes)
    results_log.append({'Model': 'BERT-base', 'Type': 'Baseline', 'F1': f1_c})
    print(f"    -> Score: {f1_c:.4f}")

    # D. DeBERTa (NEW!)
    print("\n>>> [Main] Running Advanced Model (SOTA)...", flush=True)
    start = time.time()
    # Calling the function from model_advanced.py
    f1_d = run_deberta_model(train_df, test_df, num_classes)
    results_log.append({'Model': 'DeBERTa-v3', 'Type': 'Advanced', 'F1': f1_d})
    print(f"    -> Score: {f1_d:.4f}")

    # 3. VISUALIZATION
    print("\n>>> [Main] Generating Final Comparison Plot...", flush=True)
    res_df = pd.DataFrame(results_log)
    print(res_df) 

    plt.figure(figsize=(10, 6))
    # Color the Advanced model Red to make it pop, others Grey
    colors = ['#5D6D7E' if t == 'Baseline' else '#E74C3C' for t in res_df['Type']]
    
    sns.barplot(x='Model', y='F1', data=res_df, palette=colors, hue='Model', legend=False)
    plt.ylim(0, 0.7)
    plt.title('Model Performance Benchmark: Baselines vs. SOTA', fontsize=14)
    plt.ylabel('Macro F1 Score', fontsize=12)
    
    # Add numbers on top of bars
    for index, row in res_df.iterrows():
        plt.text(index, row.F1 + 0.01, f'{row.F1:.3f}', color='black', ha="center")

    output_path = 'plots/final_benchmark.pdf'
    plt.savefig(output_path)
    print(f">>> [Main] Benchmark plot saved to {output_path}", flush=True)
    print(">>> [Main] Done.")

if __name__ == "__main__":
    main()