# Recruiter Verification and Fake Internship Detection Engine

AI-powered scam detection system for internship and job postings in the Indian job market.

## Virtual Environment Setup

Follow these steps to set up a clean Python virtual environment and install the required dependencies:

### Windows

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Linux / macOS

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ML Model Training Pipeline

To train the scam detection models directly from the local CSV dataset (`backend/data/processed_cleaned_data.csv`), execute the following command from the root folder:

```bash
python -m backend.ml.train_models
```

### Script Execution Sequence

When you run this command, the training script will:
1. **Validate Dependencies**: Automatically verify that all necessary third-party libraries (e.g. `pandas`, `xgboost`, `spacy`, `nltk`, etc.) are installed. If any is missing, it will display a clear error message and instructions to install.
2. **Auto-Download NLTK Resources**: Dynamically check and download required NLTK corpuses/tokenizers (`punkt`, `punkt_tab`, `stopwords`) only when necessary.
3. **Load CSV Dataset**: Read the local offline dataset directly using pandas and map column headers to standard formats.
4. **Generate Features**: Process text descriptions using spaCy NLP pipelines, and calculate TF-IDF and SentenceTransformer embeddings.
5. **Train Machine Learning Models**: Train XGBoost, Random Forest, and Isolation Forest models.
6. **Save Artifacts**: Output the following files to `backend/ml/models/`:
   - `tfidf_vectorizer.pkl` (TF-IDF vectorization vocabulary)
   - `xgboost.pkl` (XGBoost classifier)
   - `random_forest.pkl` (Random Forest classifier)
   - `isolation_forest.pkl` (Isolation Forest anomaly detector)
   - `scaler.pkl` (StandardScaler instance)
   - `features.csv` & `labels.csv` (Extracted training features and target labels)
   - `metadata.json` (Model parameters, training details, and accuracies)

---

## Prediction API

The training pipeline maintains backward compatibility. Once models are saved, the prediction backend can be tested or run locally using:

```bash
# Run prediction tests
python -m backend.ml.predict

# Run explanation tests (SHAP)
python -m backend.ml.explain
```
