import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import time

# Import your loader/processor to get data
from data_loader import load_data
from preprocess import TextPreprocessor

def generate_tsne_plot():
    print("\n" + "="*50)
    print(" GENERATING LATENT SPACE VISUALIZATION (t-SNE)")
    print("="*50)
    
    # 1. Load & Process Data
    print(">>> [Viz] Loading Dataset...")
    train_df, _ = load_data()
    processor = TextPreprocessor()
    train_df, _, _ = processor.process_pipeline(train_df, train_df.copy())
    
    # 2. Vectorize (Using TF-IDF for speed, represents semantic space)
    # We take a sample of 1000 points to keep the plot readable and fast
    sample_size = 1000
    if len(train_df) > sample_size:
        df_sample = train_df.sample(sample_size, random_state=42)
    else:
        df_sample = train_df
        
    print(f">>> [Viz] Vectorizing {len(df_sample)} samples...")
    tfidf = TfidfVectorizer(max_features=1000) # Limit features for cleaner projection
    vectors = tfidf.fit_transform(df_sample['clean_text']).toarray()
    
    # 3. Run t-SNE (Dimensionality Reduction: 1000d -> 2d)
    print(">>> [Viz] Running t-SNE algorithm (this projects 1000D text to 2D)...")
    time_start = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(vectors)
    print(f">>> [Viz] t-SNE converged in {time.time() - time_start:.2f}s")
    
    # 4. Plotting
    print(">>> [Viz] Rendering Manifold Projection...")
    df_sample['tsne-2d-one'] = tsne_results[:,0]
    df_sample['tsne-2d-two'] = tsne_results[:,1]
    
    # Map labels back to names for the legend
    # 0: Ambivalent, 1: Clear Non-Reply, 2: Clear Reply
    label_map = {0: 'Ambivalent', 1: 'Clear Non-Reply', 2: 'Clear Reply'}
    df_sample['Label'] = df_sample['label_enc'].map(label_map)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Label",
        palette=sns.color_palette("hsv", 3),
        data=df_sample,
        legend="full",
        alpha=0.7
    )
    
    plt.title('t-SNE Projection of Interview Responses (Semantic Separability)', fontsize=14)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    
    output_path = 'plots/tsne_manifold_projection.pdf'
    plt.savefig(output_path)
    print(f">>> [Viz] Saved high-res plot to {output_path}")

if __name__ == "__main__":
    generate_tsne_plot()