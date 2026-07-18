import os
import pandas as pd

def run_step_1_cleaning():
    
    input_file_path = "../data/raw/mtsamples.csv"
    output_file_path = "../data/processed/01_sliced_specialties.csv"
    
    
    if not os.path.exists(input_file_path):
        print(f"ERROR: Cannot find the file at {input_file_path}")
        print("Please check your folder structure and try again.")
        return

    print("--- STEP 1: Starting Data Cleaning Pipeline ---")
    
    # 2. Load the raw dataset
    df = pd.read_csv(input_file_path)
    print(f"Initial raw data shape: {df.shape}")
    
    # 3. Clean up the text inside the 'medical_specialty' column
    df['medical_specialty'] = df['medical_specialty'].astype(str).str.strip()
    
    # 4. Drop any rows where 'transcription' is completely missing (NaN)
    df = df.dropna(subset=['transcription']).reset_index(drop=True)
    print(f"Shape after dropping missing transcriptions: {df.shape}")
    
    # 5. Keep only the 5 distinct medical specialties of interest
    pure_specialties = [
        'Cardiovascular / Pulmonary',
        'Orthopedic',
        'Radiology',
        'Gastroenterology',
        'Neurology'
    ]
    
    # Filter the dataframe to keep only these 5 classes
    df_clean = df[df['medical_specialty'].isin(pure_specialties)].copy()
    print(f"Shape after keeping 5 distinct specialties: {df_clean.shape}")
    
    # 6. Remove low-signal data 
    # filters out transcriptions that are blank or have just 1 or 2 words.
    df_clean['word_count'] = df_clean['transcription'].apply(lambda text: len(str(text).split()))
    df_clean = df_clean[df_clean['word_count'] >= 20].reset_index(drop=True)
    print(f"Shape after removing short texts (< 20 words): {df_clean.shape}")
    
    # 7. Print the final clean class distribution to verify balance
    print("\n--- Final Cleaned Class Distribution ---")
    distribution = df_clean['medical_specialty'].value_counts()
    for specialty, count in distribution.items():
        percentage = (count / len(df_clean)) * 100
        print(f"* {specialty}: {count} rows ({percentage:.2f}%)")
        
    os.makedirs("../data/processed", exist_ok=True)
    df_clean.to_csv(output_file_path, index=False)
    print(f"\nSUCCESS: Cleaned data successfully saved to: {output_file_path}")

# Run the function
if __name__ == "__main__":
    run_step_1_cleaning()
