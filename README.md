# 🩺 Multi-Class Clinical Note Specialty Classifier

An end-to-end, production-grade Natural Language Processing (NLP) system built to automatically classify messy, raw medical transcriptions and route them to their proper clinical departments.

The core system architecture features a decoupled, multi-service pipeline running locally on a dedicated Python virtual environment (`venv`) to preserve system resources. It consists of an asynchronous **FastAPI backend engine** serving real-time inferences from a cached, validated machine learning model to a highly responsive **Streamlit client user dashboard**.

---

## 📊 Experimental Evaluation Scorecard

During the development phase, multiple machine learning models were built, manually adjusted for class imbalance, and cross-evaluated using the exact same stratified test matrix [Phase 7, 9, 10].

| Model Architecture | Features / Inputs | Class Balancing Strategy | Overall Accuracy | Macro F1-Score | Status & Verdict |
|---|---|---|:---:|:---:|---|
| **Logistic Regression** | 5,000-Dim TF-IDF Vectors | Built-in `class_weight='balanced'` | **41.54%** | **0.40** | 🎉 **Winner: Deployed to Production** |
| **XGBoost Classifier** | 5,000-Dim TF-IDF Vectors | Manual `compute_sample_weight()` | 28.75% | 0.22 | ❌ Rejected: Overfitted on feature sparsity |
| **PyTorch LSTM Network** | Padded Token Sequences | Unweighted Cross-Entropy Loss | 29.57% | 0.08 | ❌ Rejected: Majority class collapse |

---

## 🧠 Core Engineering & Architectural Takeaways

Hiring managers reviewing this repository should note the specific data-driven trade-offs made during model selection:

1. **The Sparsity Trap (Why Logistic Regression Won):** The text feature extraction process utilizes a `TfidfVectorizer` capped at 5,000 features, generating highly sparse arrays (where over 99% of matrix cells are zero). Tree-based models like XGBoost struggle intensely in sparse, high-dimensional spaces because they attempt to split data along single feature axes at a time, leading to severe overfitting. Conversely, **Logistic Regression** calculates global linear weights across all words simultaneously, making it mathematically superior for sparse text inputs.

2. **Mitigating Majority Class Collapse (LSTM Failure):** Deep learning sequential models (like LSTMs) are incredibly data-hungry. On a highly skewed, low-volume text corpus (fewer than 3,000 total training samples), the LSTM failed to learn language context. To lower its global cross-entropy loss quickly, the neural network took a shortcut and collapsed into a majority-class predictor — guessing the largest class (*Surgery*) for almost everything, resulting in a poor **0.08 Macro F1-score**.

3. **Data Leakage Elimination:** To prevent look-ahead bias, structural feature matrices were carefully managed. The text vectorizer and the label encoder were strictly fitted *only* on the training data (`fit_transform`) and subsequently used to project the test data (`transform`), ensuring pristine validation integrity [Phase 6, 7].

4. **Class Imbalance Optimization:** Relying strictly on global accuracy is a deceptive antipattern for skewed clinical datasets; a model can guess the majority class endlessly to maintain a high accuracy score while remaining completely blind to minority classes (like *Urology*). This pipeline explicitly optimizes for **Macro F1-score**, ensuring every clinical department is treated with equal weight during evaluation.

---

## 🛠️ Data Pipeline & Cleaning Engine

The raw clinical dictations (`data/raw/mtsamples.csv`) contain extreme textual noise, formatting variations, and structural inconsistencies [Phase 2, 3]. A robust preprocessing script (`notebooks/02_Preprocessing.ipynb`) passes strings through an immutable pipeline [Phase 4]:

```
[Raw Note] ──> Lowercase ──> Regex Alpha Filter ([^a-zA-Z\s]) ──> Stopword Strip ──> WordNet Lemmatization ──> [Clean Text]
```

- **Lemmatization:** Reduces varying words (like *headaches, headaching*) to their baseline root form (*headache*), grouping linguistic indicators into unified feature signals.
- **Label Engineering:** Rare specialties with fewer than 10 total notes were eliminated via a **Top 10 Thresholding Strategy**, ensuring the models had sufficient sample sizes to converge [Phase 5].

---

## ⚡ Production Serving Infrastructure

The validated model components are serialized directly to disk as portable weights inside the `models/` directory using Python `pickle` [Phase 11].

```
[Streamlit UI]  ───── HTTP POST [JSON text] ─────>   [FastAPI Backend Engine]
   (Port 8501)   <──── HTTP 200 [JSON Response] ────      (Port 8000, Memory Cache)
```

- **Backend Microservice (`app/main.py`):** Powered by an asynchronous FastAPI framework [Phase 12]. To eliminate severe disk I/O latency bottlenecks, a dedicated `@app.on_event("startup")` trigger unpacks and caches the heavy model arrays and text vectorizers **exactly once into RAM memory** when the application boots up. Real-time inference calls bypass disk lookups entirely, ensuring rapid routing responses.
- **Executive Client UI (`streamlit_app.py`):** Features a cohesive, single-page layout designed for non-technical users [Phase 13]. It contains a transcript intake area at the top, a prominent headline prediction block presenting the routing target alongside top-5 probability distribution graphs, and comparative engineering charts at the base to substantiate deployment decisions.

---

## 💻 Local Setup & Operational Instructions

To spin up this multi-service pipeline on your local architecture without risking storage exhaustion, execute these steps inside a fresh command prompt terminal:

### 1. Environment Initialization & Dependency Load

```bash
# Clone the repository
git clone https://github.com/<your-username>/clinical-note-classifier.git
cd clinical-note-classifier

# Set up and activate a standard local virtual environment
python -m venv env
.\env\Scripts\activate   # On Mac/Linux use: source env/bin/activate

# Install lightweight operational dependencies
pip install -r requirements.txt
```

### 2. Launch the FastAPI Backend Microservice

```bash
uvicorn app.main:app --reload
```

The backend server initializes on port `8000`. You can interact with the live automated Swagger documentation interface by visiting `http://localhost:8000/docs`.

### 3. Launch the Streamlit User Interface Dashboard

Open a secondary terminal tab, activate the virtual environment, and start the frontend script:

```bash
streamlit run streamlit_app.py
```

The application will automatically open in your browser.

---

## 📁 Project Structure

```
clinical-note-classifier/
├── app/
│   └── main.py                  # FastAPI backend engine
├── data/
│   └── raw/
│       └── mtsamples.csv        # Raw clinical dictations
├── models/                      # Serialized model weights (pickle)
├── notebooks/
│   └── 02_Preprocessing.ipynb   # Data cleaning pipeline
├── streamlit_app.py             # Streamlit dashboard
├── requirements.txt
└── README.md
```