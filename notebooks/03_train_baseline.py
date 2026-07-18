import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def run_step_3_baseline():
    input_file_path = "../data/processed/02_preprocessed_text.csv"
    
    if not os.path.exists(input_file_path):
        print(f"ERROR: Cannot find the file at {input_file_path}")
        return

    print("--- STEP 3: Training Baseline Machine Learning Model ---")
    df = pd.read_csv(input_file_path)
    
    # Fill any accidental missing values with empty strings
    df['clean_text'] = df['clean_text'].fillna("")
    
    X = df['clean_text']
    y = df['medical_specialty']

    # 1. Stratified Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 2. Extract Numerical Features using TF-IDF
    # include bi-grams (ngram_range=(1,2)) to capture word combos like 'chest pain'
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"Feature matrix footprint shape: {X_train_tfidf.shape}")

    # 3. Initialize and Train Logistic Regression
    print("Fitting Logistic Regression boundary lines...")
    model_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model_lr.fit(X_train_tfidf, y_train)

    # 4. Make Predictions and Evaluate
    y_pred = model_lr.predict(X_test_tfidf)
    final_accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*20 + " BASELINE EVALUATION REPORT " + "="*20)
    print(f"Overall Sliced Target Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)\n")
    print(classification_report(y_test, y_pred))

    # 5. Export artifacts to the models folder for deployment step
    os.makedirs("../models", exist_ok=True)
    
    with open("../models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
        
    with open("../models/logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model_lr, f)
        
    print("SUCCESS: Model and Vectorizer saved to '../models/' directory!")

if __name__ == "__main__":
    run_step_3_baseline()
