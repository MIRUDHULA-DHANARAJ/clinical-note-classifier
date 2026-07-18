import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

def run_step_4_xgboost():
    input_file_path = "../data/processed/02_preprocessed_text.csv"
    
    if not os.path.exists(input_file_path):
        print(f"ERROR: Cannot find the file at {input_file_path}")
        return

    print("--- STEP 4: Training Optimized XGBoost Classifier ---")
    df = pd.read_csv(input_file_path)
    df['clean_text'] = df['clean_text'].fillna("")
    
    X = df['clean_text']
    y = df['medical_specialty']

    # 1. Stratified Split (Identical split ratios to maintain consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )

    # 2. Encode categorical labels into integer numbers for XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 3. TF-IDF Feature Extraction
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 4. Initialize and Train XGBoost with Shallow regularized tree parameters
    print("Fitting Gradient Boosted decision trees...")
    model_xgb = XGBClassifier(
        objective='multi:softprob',
        learning_rate=0.08,
        max_depth=4,         # Shallow trees prevent overfitting on noisy medical terms
        n_estimators=250,
        subsample=0.8,       # Train on 80% of rows per tree to handle variance
        colsample_bytree=0.8, # Train on 80% of vocabulary features per tree
        random_state=42
    )
    
    model_xgb.fit(X_train_tfidf, y_train_encoded)

    # 5. Make Predictions and Decode back to string labels
    y_pred_encoded = model_xgb.predict(X_test_tfidf)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    final_accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*20 + " XGBOOST PRODUCTION REPORT " + "="*20)
    print(f"Overall Sliced Target Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)\n")
    print(classification_report(y_test, y_pred))

    # 6. Export models
    os.makedirs("../models", exist_ok=True)
    with open("../models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model_xgb, f)
    with open("../models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
        
    print("SUCCESS: XGBoost model and Label Encoder saved to '../models/' folder!")

if __name__ == "__main__":
    run_step_4_xgboost()
