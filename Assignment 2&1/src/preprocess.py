import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TextPreprocessor:
    def __init__(self):
        # We need to turn text labels (like 'Ambivalent') into numbers (0, 1, 2)
        self.label_encoder = LabelEncoder()

    def clean_text_func(self, text):
        """
        Data Cleaning Module
        --------------------
        Removes noise identified in Assignment 1 (artifacts, URLs, speaker tags).
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Lowercase
        text = text.lower()
        # 2. Remove Transcription Artifacts [inaudible]
        text = re.sub(r'\[.*?\]', '', text)
        # 3. Remove URLs
        text = re.sub(r'http\S+', '', text)
        # 4. Remove Speaker Tags (e.g., "TRUMP:")
        text = re.sub(r'^[a-z\s]+:\s*', '', text) 
        # 5. Normalize Whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_pipeline(self, train_df, test_df):
        print(">> [Preprocessor] Starting Data Cleaning & Encoding Pipeline...")

        train_df = train_df.copy()
        test_df = test_df.copy()

        # 1. Clean
        print(f"   - Cleaning {len(train_df)} training samples...")
        train_df['clean_text'] = train_df['interview_answer'].apply(self.clean_text_func)
        test_df['clean_text'] = test_df['interview_answer'].apply(self.clean_text_func)

        # 2. Encode
        print("   - Encoding target labels...")
        target_col = 'clarity_label'
        
        # Fit on TRAIN, transform on TEST
        # Using 'label_enc' to match ALL models
        train_df['label_enc'] = self.label_encoder.fit_transform(train_df[target_col])
        
        # Safe transform for test
        test_df['label_enc'] = test_df[target_col].apply(
            lambda x: self.label_encoder.transform([x])[0] if x in self.label_encoder.classes_ else -1
        )
        
        mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        print(f"   - Class Mapping: {mapping}")
        
        return train_df, test_df, len(self.label_encoder.classes_)