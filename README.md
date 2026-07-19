# 🏥 Clinical Intake & Automated Specialty Routing System

An end-to-end NLP pipeline that classifies unstructured physician transcriptions into medical specialty queues, enabling automated clinical document routing. Built on the MTSamples dataset, benchmarked across linear, gradient-boosted, and transformer-based architectures, and deployed as an interactive Streamlit dashboard.

---

## 📌 Overview

Hospitals and healthcare operations teams routinely process large volumes of unstructured clinical text — physician dictations, triage notes, and chart summaries — that must be manually routed to the correct specialty department. This project automates that routing decision using NLP and machine learning, reducing manual triage overhead and operational delay.

The system classifies clinical notes into one of five specialty categories:

- Cardiovascular / Pulmonary
- Orthopedic
- Radiology
- Neurology
- Gastroenterology

---

## 🗺️ Pipeline Architecture

```
Raw MTSamples Data
        │
        ▼
Step 1 — Data Cleaning & Class Isolation
        │
        ▼
Step 2 — Text Normalization & Lemmatization
        │
        ▼
Step 3 — Baseline Model (TF-IDF + Logistic Regression)
        │
        ▼
Step 4 — Gradient-Boosted Model (XGBoost)
        │
        ▼
Step 4.6 — Fine-Tuned Transformer (BioClinicalBERT)
        │
        ▼
Deployment — Streamlit Dashboard (Hugging Face Hub model)
```

---

## 🔍 Problem Statement

Raw clinical text contains heavily overlapping vocabulary across departments — administrative terms like *"patient"* or *"incision"* appear regardless of specialty, which weakens naive statistical classifiers. Early iterations of this project also revealed a deeper issue: two of the largest labels in the raw dataset, **Surgery** and **Consult**, describe *document types* rather than *physiological specialties*, causing severe class overlap and capping initial accuracy at ~41%.

---

## 🛠️ What This Project Does

| Stage | Script | Purpose |
|---|---|---|
| 1 | `step_1_cleaning.py` | Strips whitespace from labels, drops missing transcriptions, isolates 5 pure specialty classes, removes short/low-signal notes (< 20 words) |
| 2 | `step_2_preprocessing.py` | Lowercases text, strips non-alphabetic characters, removes stopwords, applies WordNet lemmatization |
| 3 | `step_3_baseline.py` | Trains a `TfidfVectorizer` (bi-gram, 4,000 features) + class-weighted Logistic Regression baseline |
| 4 | `step_4_xgboost.py` | Trains a regularized, shallow-tree XGBoost classifier on the same TF-IDF features |
| 4.6 | `step_4_6_transformer.py` | Fine-tunes `Bio_ClinicalBERT` (domain-pretrained transformer) for sequence classification |
| — | `app.py` | Streamlit dashboard serving the production model with confidence scoring and human-in-the-loop review routing |

---

## 🚨 Key Engineering Challenges & Fixes

### 1. Class Contamination (41% Accuracy Ceiling)
**Problem:** `Surgery` and `Consult` labels represented administrative document types, not physiological departments, causing massive vocabulary crossover between unrelated specialties.
**Fix:** Removed both classes at the data layer; isolated 5 non-overlapping physiological categories and dropped sub-20-word notes.

### 2. Transformer Overfitting / Prediction Collapse
**Problem:** Initial resource-constrained fine-tuning (batch size 4, single epoch, CPU-only) caused the model to collapse to a single constant prediction.
**Fix:** Adjusted training configuration (learning rate, weight decay, evaluation strategy) and validated per-epoch loss to ensure stable convergence before promoting the transformer to production.

### 3. Label Index Drift
**Problem:** Untrimmed whitespace in raw specialty labels (e.g. `" Cardiovascular / Pulmonary"`) shifted alphabetical label-encoder ordering, misaligning predicted indices with displayed specialty names.
**Fix:** Enforced `.str.strip()` on the label column at the earliest cleaning stage (Step 1), so the fix is applied once at the source rather than patched downstream in the application layer.

---

## 🧠 Model Comparison & Trade-offs

| Model | Accuracy | Strengths | Limitations |
|---|---|---|---|
| Logistic Regression (TF-IDF, bi-gram, balanced) | 65.26% | Sub-millisecond inference, minimal footprint, stable on sparse features | Cannot capture non-linear contextual relationships |
| XGBoost (regularized, shallow trees) | 61.05% | Models non-linear feature interactions | Degrades on very high-dimensional sparse TF-IDF matrices |
| Fine-tuned BioClinicalBERT | **76.84%** | Domain-pretrained contextual embeddings, captures long-range semantic relationships | Higher compute cost, longer inference latency, requires more careful training configuration |

**Production model:** `BioClinicalBERT`, fine-tuned and hosted on the Hugging Face Hub, selected for its superior accuracy on unstructured clinical language once a stable training configuration was established.

---

## 📈 How Accuracy Was Improved

1. **Semantic Isolation** — removed administrative document-type labels contaminating the target classes.
2. **Bi-gram Feature Engineering** — `ngram_range=(1,2)` allows models to capture clinically meaningful phrases (e.g. *"chest pain"*, *"joint line"*) rather than isolated tokens.
3. **Class Balancing** — `class_weight='balanced'` in Logistic Regression prevents minority specialty classes from being under-predicted.
4. **Domain-Pretrained Transfer Learning** — fine-tuning BioClinicalBERT (pretrained on clinical corpora) rather than a general-purpose transformer gave a substantial accuracy lift over TF-IDF-based models.

---

## 🖥️ Deployment

The production model is served through a Streamlit dashboard (`app.py`) that:

- Loads the fine-tuned BioClinicalBERT model and tokenizer directly from the Hugging Face Hub
- Runs local inference on submitted clinical text
- Displays the predicted specialty with a confidence score
- Applies a configurable **Human-in-the-Loop Safety Threshold** — any prediction below the threshold is automatically flagged for manual review rather than auto-routed
- Tracks session-level routing analytics (specialty distribution, average confidence)

---

## 📂 Project Structure

```
clinical-note-classifier/
├── data/
│   ├── raw/
│   │   └── mtsamples.csv
│   └── processed/
│       ├── 01_sliced_specialties.csv
│       └── 02_preprocessed_text.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression_model.pkl
│   ├── xgboost_model.pkl
│   ├── label_encoder.pkl
│   ├── transformer_label_encoder.pkl
│   └── bioclinicalbert_pipeline/
├── src/
│   ├── step_1_cleaning.py
│   ├── step_2_preprocessing.py
│   ├── step_3_baseline.py
│   ├── step_4_xgboost.py
│   └── step_4_6_transformer.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# 1. Clone the repository
git clone https://github.com/MIRUDHULA-DHANARAJ/clinical-note-classifier
cd clinical-note-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline (from src/)
python step_1_cleaning.py
python step_2_preprocessing.py
python step_3_baseline.py
python step_4_xgboost.py
python step_4_6_transformer.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## 🧰 Tech Stack

- **Language:** Python
- **NLP:** NLTK (stopwords, lemmatization), Hugging Face Transformers
- **Modeling:** scikit-learn (Logistic Regression, TF-IDF), XGBoost, PyTorch, Bio_ClinicalBERT
- **Deployment:** Streamlit, Hugging Face Hub (model hosting)
- **Visualization:** Plotly

---

## 🔮 Future Improvements

- Expand training data volume to further stabilize transformer fine-tuning
- Add batch/bulk note upload support to the dashboard
- Introduce model explainability (e.g. attention visualization or SHAP) for clinician trust
- Extend specialty coverage beyond the current 5 categories
- Add automated retraining pipeline as new labeled data becomes available

---

## 📄 License

This project is intended for educational and portfolio demonstration purposes.
