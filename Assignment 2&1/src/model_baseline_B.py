import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Helper: Vocab Builder ---
class Vocab:
    def __init__(self, texts, max_words=5000):
        # Flatten list of sentences into list of words
        words = [w for t in texts for w in t.split()]
        counts = Counter(words)
        # 0 is reserved for padding, so start words at 1
        self.word2idx = {w: i+1 for i, (w, _) in enumerate(counts.most_common(max_words))}
        self.pad_idx = 0
        
    def encode(self, text):
        return [self.word2idx.get(w, 0) for w in text.split()] # 0 if unknown

# --- Helper: LSTM Model ---
class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Bidirectional because context from the end of the sentence matters too
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Concat the final forward and backward hidden states
        hidden_final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden_final)

def run_baseline_B(train_df, test_df, num_classes):
    print("\n" + "="*30)
    print(" RUNNING BASELINE B (Bi-LSTM)")
    print("="*30)
    
    # 1. Prepare Data
    vocab = Vocab(train_df['clean_text'])
    
    def collate_fn(batch):
        # This handles the padding so all sentences in a batch are same length
        texts, labels = zip(*batch)
        encoded = [torch.tensor(vocab.encode(t)) for t in texts]
        padded = pad_sequence(encoded, batch_first=True, padding_value=0)
        return padded, torch.tensor(labels)

    train_loader = DataLoader(list(zip(train_df['clean_text'], train_df['label_enc'])), 
                              batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(list(zip(test_df['clean_text'], test_df['label_enc'])), 
                             batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 2. Init Model
    model = LSTMNet(len(vocab.word2idx)+1, 100, 64, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop (Short and sweet)
    print(">>> [Baseline B] Training for 3 epochs...")
    model.train()
    for epoch in range(3):
        for text, label in train_loader:
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
    # 4. Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for text, label in test_loader:
            out = model(text)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(label.numpy())
            
    score = f1_score(all_labels, all_preds, average='macro')
    print(f">>> [Baseline B] Macro F1: {score:.4f}")
    
    # 5. Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Baseline B Confusion Matrix (F1: {score:.2f})")
    plt.savefig("plots/cm_baseline_b.pdf")
    
    return all_preds, score