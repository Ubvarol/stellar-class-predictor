# 🩺 Student Health Risk Predictor

Interactive Streamlit app that classifies a student's health condition
(**at-risk / unhealthy / fit**) from lifestyle and vitals — built for the
**Kaggle Playground Series 2026** competition (metric: balanced accuracy).

- **Model:** multiclass LightGBM with **balanced class weights**
- **Validation:** 5-fold CV, balanced accuracy ≈ 0.95
- **Live app:** deployed on Streamlit Community Cloud

## Features

- **Predict** — set a student's profile (sleep, BMI, activity, habits…) and get
  the predicted health class with per-class probabilities.
- **How the model was built** — the imbalance challenge, why balanced class
  weights matter for balanced accuracy, and feature importance.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

```
streamlit_app/
├── app.py                    # the Streamlit app
├── requirements.txt
└── model/
    ├── health_lgbm.joblib    # trained multiclass LightGBM
    └── health_meta.joblib    # feature metadata, ranges, importances
```

Trained in the accompanying notebook (`student_health_prediction.ipynb`), which
covers EDA, the class-imbalance strategy, 5-fold CV and an honest test of the
prior-adjustment decision rule.
