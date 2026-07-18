import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def run_step_4_6_transformer():
    input_file_path = "../data/processed/02_preprocessed_text.csv"
    if not os.path.exists(input_file_path):
        print(f"ERROR: Cannot find the file at {input_file_path}")
        return

    print("--- STEP 4.6: Training Pre-trained Clinical Transformer (BioClinicalBERT) ---")
    
    # 1. Load data
    df = pd.read_csv(input_file_path)
    df['clean_text'] = df['clean_text'].fillna("")
    
    # 2. Encode Labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['medical_specialty'])
    num_labels = len(label_encoder.classes_)
    
    # 3. Stratified Split
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['label']
    )
    
    # Convert Pandas dataframes directly into Hugging Face Dataset formats
    train_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['clean_text', 'label']])
    
    # 4. Load BioClinicalBERT Tokenizer from Hugging Face Hub
    model_name = "EmilyALSENTZER/Bio_ClinicalBERT"
    print("Loading specialized clinical tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Helper function to tokenize inputs sequentially
    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], padding="max_length", truncation=True, max_length=256)
    
    print("Tokenizing datasets into dense contextual tensors...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)
    
    # 5. Load Pre-trained Sequence Classification Architecture
    print("Downloading pre-trained BioClinicalBERT weights...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Running transformer optimization on: {device}")
    
    
    
    training_args = TrainingArguments(
        output_dir="./transformer_results",
        eval_strategy="epoch",        # Evaluate performance at the end of every epoch
        save_strategy="epoch",       
        learning_rate=2e-5,           # Slow learning rate keeps pre-trained features stable
        per_device_train_batch_size=4 , # Low batch size 
        per_device_eval_batch_size=4 ,
        num_train_epochs=1 ,             
        weight_decay=0.01,            # Regularization penalty to prevent overfitting
        logging_steps=10,
        load_best_model_at_end=True,  
        metric_for_best_model="loss", # Base best model selection on lowest validation loss
        report_to="none"              # Disables external tracker alerts
    )

    
    # 7. Initialize the Trainer Engine Loop
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        processing_class=tokenizer        
    )
    
    print("\nCommencing Transformer fine-tuning loops...")
    trainer.train()
    
    # 8. Evaluate Transformer Performance
    print("\nEvaluating model performance on unseen test data...")
    predictions_output = trainer.predict(test_tokenized)
    raw_logits = predictions_output.predictions
    predicted_labels = np.argmax(raw_logits, axis=1)
    
    # Decode integers back to medical string names
    y_pred_names = label_encoder.inverse_transform(predicted_labels)
    y_test_names = label_encoder.inverse_transform(test_df['label'])
    
    final_accuracy = accuracy_score(y_test_names, y_pred_names)
    print("\n" + "="*20 + " BIOCLINICALBERT EVALUATION " + "="*20)
    print(f"Overall Sliced Target Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)\n")
    print(classification_report(y_test_names, y_pred_names))
    
    # 9. Export Production Weights
    output_dir = "../models/bioclinicalbert_pipeline"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open("../models/transformer_label_encoder.pkl", "wb") as f:
        import pickle
        pickle.dump(label_encoder, f)
        
    print(f"SUCCESS: Fine-tuned transformer model saved to: {output_dir}")

if __name__ == "__main__":
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "False"
    run_step_4_6_transformer()
