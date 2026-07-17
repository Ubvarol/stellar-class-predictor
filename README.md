# 🌌 Stellar Class Predictor

Interactive Streamlit app that classifies an astronomical object as **GALAXY**,
**QSO** (quasar) or **STAR** from SDSS photometry and redshift — built for the
**Kaggle Playground Series 2026 (S6E6)** competition (metric: balanced accuracy).

- **Model:** multiclass LightGBM with balanced class weights
- **Competition result:** a 4-model stack reached **0.9661** OOF balanced accuracy
- **Live app:** deployed on Streamlit Community Cloud

## Features

- **Predict** — set redshift, the five photometric magnitudes and object type;
  get the predicted class with per-class probabilities.
- **How the model was built** — the approach, why redshift is the key separator,
  and feature importance.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

```
streamlit_app/
├── app.py                     # the Streamlit app
├── requirements.txt
└── model/
    ├── stellar_lgbm.joblib    # trained multiclass LightGBM
    └── stellar_meta.joblib    # feature metadata, ranges, importances
```

Trained in the accompanying notebook (`stellar_class_prediction.ipynb`), which
follows a 16-section template: EDA, feature engineering, cross-validation,
Optuna tuning, a 4-model stack with threshold calibration, and an experiment
tracker.
