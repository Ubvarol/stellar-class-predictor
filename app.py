"""
Stellar Class Predictor — Kaggle Playground Series 2026 (S6E6)
Enter an object's photometry and redshift; the model classifies it as
GALAXY, QSO (quasar) or STAR. Multiclass LightGBM with balanced class weights.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stellar Class Predictor", page_icon="🌌", layout="centered")


@st.cache_resource
def load_artifacts():
    return (joblib.load("model/stellar_lgbm.joblib"),
            joblib.load("model/stellar_meta.joblib"))


model, meta = load_artifacts()
FEATURES = meta["features"]
CATEGORICAL = meta["categorical"]
CAT_OPTIONS = meta["cat_options"]
RANGES = meta["num_ranges"]      # {col: (low, median, high)}
CLASSES = meta["classes"]

EMOJI = {"GALAXY": "🌌 Galaxy", "QSO": "💫 Quasar (QSO)", "STAR": "⭐ Star"}
NICE = {"alpha": "Right ascension (α)", "delta": "Declination (δ)",
        "u": "u magnitude", "g": "g magnitude", "r": "r magnitude",
        "i": "i magnitude", "z": "z magnitude", "redshift": "Redshift"}


def build_row(inputs):
    row = pd.DataFrame([inputs])[FEATURES]
    for c in CATEGORICAL:
        row[c] = pd.Categorical(row[c], categories=CAT_OPTIONS[c])
    return row


def predict(inputs):
    return model.predict_proba(build_row(inputs))[0]


# ---------------------------------------------------------------------------
st.title("🌌 Stellar Class Predictor")
st.caption("Kaggle Playground Series 2026 (S6E6) · multiclass LightGBM · "
           "balanced accuracy ≈ **0.966**")

st.markdown("Classify an astronomical object as **galaxy**, **quasar** or **star** "
            "from its SDSS photometry and redshift. Trained on 577,347 objects.")

tab_predict, tab_model = st.tabs(["🔮 Predict", "📊 How the model was built"])

with tab_predict:
    st.markdown("#### The decisive feature: redshift")
    redshift = st.slider("Redshift", -0.01, 7.0,
                         float(round(RANGES["redshift"][1], 3)), 0.01,
                         help="Stars ~0, galaxies ~0.5, quasars far higher")

    st.markdown("#### Photometric magnitudes (SDSS u, g, r, i, z filters)")
    c1, c2, c3 = st.columns(3)
    mags = {}
    bands = ["u", "g", "r", "i", "z"]
    for k, band in enumerate(bands):
        with [c1, c2, c3][k % 3]:
            lo, med, hi = RANGES[band]
            mags[band] = st.slider(NICE[band], float(round(lo, 1)),
                                   float(round(hi, 1)), float(round(med, 1)))

    with st.expander("Sky coordinates & object type"):
        cc1, cc2 = st.columns(2)
        with cc1:
            alpha = st.slider(NICE["alpha"], 0.0, 360.0, float(round(RANGES["alpha"][1], 1)))
        with cc2:
            delta = st.slider(NICE["delta"], -20.0, 90.0, float(round(RANGES["delta"][1], 1)))
        d1, d2 = st.columns(2)
        with d1:
            spectral = st.selectbox("Spectral type", CAT_OPTIONS["spectral_type"])
        with d2:
            galpop = st.selectbox("Galaxy population", CAT_OPTIONS["galaxy_population"])

    inputs = {"alpha": alpha, "delta": delta, **mags, "redshift": redshift,
              "spectral_type": spectral, "galaxy_population": galpop}
    proba = predict(inputs)
    pred = CLASSES[int(np.argmax(proba))]

    st.metric("Predicted class", EMOJI.get(pred, pred))
    probs = pd.Series({EMOJI.get(c, c): p for c, p in zip(CLASSES, proba)}
                      ).sort_values(ascending=False)
    st.bar_chart(probs)
    st.caption("Try sweeping redshift from 0 upward — the prediction moves "
               "star → galaxy → quasar, exactly as the physics says.")

with tab_model:
    st.markdown("""
### The task

Classify each SDSS object as **GALAXY**, **QSO** or **STAR** — a three-class
problem scored by **balanced accuracy** (average recall across classes), so the
rarer star/quasar classes count as much as the common galaxies.

### The approach

- **Balanced class weights** handle the 65/20/14% imbalance — the key lever for
  balanced accuracy.
- **Redshift is the physical backbone**: stars sit near 0, galaxies around 0.5,
  quasars far higher. Photometric colours (differences between u,g,r,i,z filters)
  refine the rest.
- **LightGBM** handles the mixed numeric/categorical data natively.

The competition notebook goes further — a 4-model stack (LightGBM + XGBoost +
CatBoost + HistGradientBoosting) with a balanced-logistic-regression meta-learner
and honest nested cross-validation, reaching **0.9661** OOF balanced accuracy.
This app serves a single LightGBM for speed.

### What the model pays attention to
""")
    imp = pd.Series(meta["importance"])
    imp.index = [NICE.get(i, i) for i in imp.index]
    st.bar_chart(imp.sort_values().tail(10))

    st.markdown("**Median redshift by class:** " +
                ", ".join(f"{EMOJI.get(k, k)} {v}" for k, v in meta["redshift_by_class"].items()) +
                " — the single cleanest separator in the whole dataset.")
