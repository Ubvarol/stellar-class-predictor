"""
Student Health Risk Predictor — Kaggle Playground Series 2026
Enter a student's profile; the model predicts their health condition
(at-risk / unhealthy / fit). Multiclass LightGBM with balanced class weights.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Health Predictor", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifacts():
    return (joblib.load("model/health_lgbm.joblib"),
            joblib.load("model/health_meta.joblib"))


model, meta = load_artifacts()
FEATURES = meta["features"]
CATEGORICAL = meta["categorical"]
CAT_OPTIONS = meta["cat_options"]
RANGES = meta["num_ranges"]      # {col: (low, median, high)}
CLASSES = meta["classes"]
INV = meta["inv"]

EMOJI = {"fit": "💪 Fit", "unhealthy": "😐 Unhealthy", "at-risk": "⚠️ At-risk"}
NICE = {c: c.replace("_", " ").title() for c in FEATURES}


def build_row(inputs):
    row = pd.DataFrame([inputs])[FEATURES]
    for c in CATEGORICAL:
        row[c] = pd.Categorical(row[c], categories=CAT_OPTIONS[c])
    return row


def predict(inputs):
    proba = model.predict_proba(build_row(inputs))[0]
    return proba


# ---------------------------------------------------------------------------
st.title("🩺 Student Health Risk Predictor")
st.caption("Kaggle Playground Series 2026 · multiclass LightGBM · "
           "balanced accuracy ≈ **0.95**")

st.markdown("Enter a student's lifestyle profile and the model predicts their "
            "health condition — trained on 690,088 students.")

tab_predict, tab_model = st.tabs(["🔮 Predict", "📊 How the model was built"])

with tab_predict:
    st.markdown("#### Lifestyle & vitals")
    c1, c2 = st.columns(2)
    num_vals = {}
    for i, col in enumerate(meta["numeric"]):
        lo, med, hi = RANGES[col]
        with (c1 if i % 2 == 0 else c2):
            num_vals[col] = st.slider(NICE[col], float(round(lo, 1)),
                                      float(round(hi, 1)), float(round(med, 1)))

    st.markdown("#### Habits")
    cats = meta["categorical"]
    cc = st.columns(3)
    cat_vals = {}
    for i, col in enumerate(cats):
        with cc[i % 3]:
            cat_vals[col] = st.selectbox(NICE[col], CAT_OPTIONS[col])

    inputs = {**num_vals, **cat_vals}
    proba = predict(inputs)
    pred = CLASSES[int(np.argmax(proba))]

    st.metric("Predicted health condition", EMOJI.get(pred, pred))
    probs = pd.Series({EMOJI.get(c, c): p for c, p in zip(CLASSES, proba)}
                      ).sort_values(ascending=False)
    st.bar_chart(probs)
    st.caption("The bars show the model's probability for each class. The "
               "prediction is the highest one.")

with tab_model:
    st.markdown("""
### The task

Classify each student as **at-risk**, **unhealthy**, or **fit** — a *three-class*
problem scored by **balanced accuracy** (the average recall across the three
classes).

### The crucial twist: imbalance + balanced accuracy

86% of students are `at-risk`, yet balanced accuracy weighs each class equally.
A lazy "everyone is at-risk" model scores just **0.333**. So the whole game is
recovering the rare `fit` and `unhealthy` students — which we do by training the
LightGBM with **balanced class weights** (each class up-weighted by the inverse
of its frequency).

### Why LightGBM

Tabular data with missing values and categorical columns → gradient boosting,
which handles both natively. No imputation or one-hot encoding needed.

### What the model pays attention to
""")
    imp = pd.Series(meta["importance"])
    imp.index = [NICE.get(i, i) for i in imp.index]
    st.bar_chart(imp.sort_values().tail(10))
    st.caption("LightGBM feature importance — sleep, BMI and activity lead.")

    st.markdown(
        "**Honest note:** we tested the usual imbalance trick of dividing "
        "probabilities by class priors before choosing a label. It *hurt* here "
        "(0.934 vs 0.949) — because the training weights already balanced the "
        "classes, adjusting again over-corrects. Measure, don't assume.")
