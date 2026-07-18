import os
import re
import pandas as pd
import nltk

print("--- STEP 2: Downloading Necessary NLP Packages ---")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def run_step_2_preprocessing():
    input_file_path = "../data/processed/01_sliced_specialties.csv"
    output_file_path = "../data/processed/02_preprocessed_text.csv"
    
    if not os.path.exists(input_file_path):
        print(f"ERROR: Cannot find the file at {input_file_path}")
        return

    print("\n--- Starting Text Normalization Loop ---")
    df = pd.read_csv(input_file_path)
    print(f"Loaded dataset with {len(df)} rows.")

    # Initialize our NLP tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Define an intermediate, clean processing function
    def clean_single_note(text):
        if not isinstance(text, str):
            return ""
        
        # A. Convert all text to lowercase
        text = text.lower()
        
        # B. Strip out everything except normal alphabet letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # C. Split the text into a clean list of individual words
        words = text.split()
        
        # D. Loop through words, drop stop words, and apply lemmatization
        processed_words = []
        for w in words:
            if w not in stop_words:
                base_word = lemmatizer.lemmatize(w)
                processed_words.append(base_word)
                
        # E. Stitch the clean words back together with single spaces
        return " ".join(processed_words)

    print("Processing clinical notes. Please wait, this may take a few seconds...")
    # Apply  clean function to every row in the dataset
    df['clean_text'] = df['transcription'].apply(clean_single_note)
    
    # check to remove any row that became completely empty after cleaning
    df = df[df['clean_text'].str.strip() != ""].reset_index(drop=True)
    
    df.to_csv(output_file_path, index=False)
    print(f"SUCCESS: Preprocessed dataset shape: {df.shape}")
    print(f"Saved cleanly to: {output_file_path}")

if __name__ == "__main__":
    run_step_2_preprocessing()
