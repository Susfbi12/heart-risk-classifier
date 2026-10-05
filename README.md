# Heart Disease Risk Classifier (PBL Prototype)

Classifies a patient as Low / Moderate / High cardiovascular risk using Logistic Regression.

## Files
| File | What it does |
|---|---|
| `heart.csv` | Dataset (Cleveland heart-disease data, 303 rows) |
| `train_model.py` | Preprocesses, trains, evaluates, saves `model.joblib` + the two plots |
| `app.py` | Streamlit web app: enter patient values -> get risk tier |
| `requirements.txt` | Python dependencies |

## Run it
```bash
pip install -r requirements.txt
python train_model.py        # trains model, prints metrics, saves model.joblib
streamlit run app.py         # opens the app in your browser
```

## How it works
1. Split data 75/25 (stratified) **before** any preprocessing.
2. Pipeline: median imputation -> StandardScaler (age, trestbps, chol, thalach, oldpeak)
   -> one-hot encoding (cp, restecg, slope, thal) -> Logistic Regression.
3. `predict_proba` gives disease probability: < 0.35 Low, > 0.65 High, otherwise Moderate.

For learning purposes only - not a medical device.
