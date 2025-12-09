import pandas as pd
from datasets import load_dataset as hf_load_dataset # Renaming to avoid confusion

def load_data(): # <--- This is the function name you call in train_eval.py
    """
    Fetching the data from Hugging Face. 
    Using the 'ailsntua/QEvasion' dataset as specified in the assignment PDF.
    """
    print(">> [Data Loader] Connectng to HuggingFace to grab the dataset...")
    
    try:
        # Loading the dataset. It comes with 'train' and 'test' splits already.
        dataset = hf_load_dataset("ailsntua/QEvasion")
        
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        print(f">> [Data Loader] Success! Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        return train_df, test_df

    except Exception as e:
        print(f"!! [Data Loader] Something went wrong loading the data: {e}")
        return None, None