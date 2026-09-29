"""
app.py
------
Streamlit dashboard for the HealthWise insurance premium model.

Two-stage model:
  1) Classifier predicts a customer's risk_tier (Low / Medium / High)
  2) A per-tier LinearRegression predicts their annual_charge (premium)

Run locally:
    streamlit run app.py

Place `healthwise.csv` next to this file (or upload it in the browser) the
first time you run the app -- the model will train automatically and save
itself to healthwise_model.joblib for instant reloads afterwards.
"""

import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Model logic lives directly in this file (not imported from train_model.py)
# so the dashboard can never break due to a missing sibling module on a
# hosting platform. train_model.py duplicates this logic as an optional
# standalone CLI script for offline training — the two are independent.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
NUM = ["age", "bmi", "children", "exercise_freq"]
CAT = ["sex", "smoker", "region"]
FEATURES = NUM + CAT
TIERS = ["Low", "Medium", "High"]

MODEL_PATH = "healthwise_model.joblib"
DATA_PATH = "healthwise.csv"
TIER_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}
TIER_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

st.set_page_config(page_title="Insurance Premium Predictor", page_icon="🏥", layout="wide")


def _clean_uploaded(csv_path_or_buffer):
    df = pd.read_csv(csv_path_or_buffer)
    df = df.drop_duplicates()
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["exercise_freq"] = df["exercise_freq"].fillna(df["exercise_freq"].median())
    return df


def _build_preprocessor():
    return ColumnTransformer(
        [
            ("num", "passthrough", NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT),
        ]
    )


def _train(df):
    med = {c: float(df[c].median()) for c in NUM}

    # Stage 1: risk-tier classifier, fit on all data for production use
    clf = Pipeline([("pre", _build_preprocessor()), ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))])
    clf.fit(df[FEATURES], df["risk_tier"])

    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        df[FEATURES], df["risk_tier"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["risk_tier"]
    )
    clf_eval = Pipeline([("pre", _build_preprocessor()), ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))])
    clf_eval.fit(Xtr_c, ytr_c)
    clf_acc = accuracy_score(yte_c, clf_eval.predict(Xte_c))

    # Stage 2: one LinearRegression per risk tier
    regs, reg_metrics = {}, {}
    for t in TIERS:
        sub = df[df["risk_tier"] == t]
        if len(sub) < 10:
            continue
        Xtr, Xte, ytr, yte = train_test_split(sub[FEATURES], sub["annual_charge"], test_size=0.2, random_state=RANDOM_STATE)
        eval_reg = Pipeline([("pre", _build_preprocessor()), ("model", LinearRegression())]).fit(Xtr, ytr)
        pred = eval_reg.predict(Xte)
        reg_metrics[t] = {"r2": float(r2_score(yte, pred)), "mae": float(mean_absolute_error(yte, pred)), "n": int(len(sub))}
        full_reg = Pipeline([("pre", _build_preprocessor()), ("model", LinearRegression())]).fit(sub[FEATURES], sub["annual_charge"])
        regs[t] = full_reg

    return {
        "clf": clf,
        "regs": regs,
        "features": FEATURES,
        "num": NUM,
        "cat": CAT,
        "med": med,
        "metrics": {"classifier_accuracy": float(clf_acc), "regression": reg_metrics},
    }


def predict(model, customer):
    row = pd.DataFrame([customer])
    for k in model["num"]:
        if k not in row or pd.isna(row.at[0, k]):
            row[k] = model["med"][k]
    tier = model["clf"].predict(row[model["features"]])[0]
    if tier not in model["regs"]:
        return tier, float("nan")
    premium = float(model["regs"][tier].predict(row[model["features"]])[0])
    return tier, round(premium, 2)


@st.cache_resource(show_spinner="Training model...")
def get_model(csv_path_or_buffer):
    if csv_path_or_buffer is None:
        return None
    df = _clean_uploaded(csv_path_or_buffer)
    return _train(df)


@st.cache_data
def get_raw_data(csv_path_or_buffer):
    if csv_path_or_buffer is None:
        return None
    if isinstance(csv_path_or_buffer, str):
        return pd.read_csv(csv_path_or_buffer)
    csv_path_or_buffer.seek(0)
    return pd.read_csv(csv_path_or_buffer)


# ---------------------------------------------------------------------------
# Load model: prefer an already-saved model artifact, otherwise train fresh
# from healthwise.csv (locally) or an uploaded file (e.g. on Streamlit Cloud).
# ---------------------------------------------------------------------------
st.title("🏥 Insurance Premium Predictor")
st.caption("Two-stage model: risk-tier classifier ➜ per-tier premium regression")

data_source = None
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    if os.path.exists(DATA_PATH):
        data_source = DATA_PATH
elif os.path.exists(DATA_PATH):
    data_source = DATA_PATH
    model = get_model(DATA_PATH)
    joblib.dump(model, MODEL_PATH)
else:
    st.info("No saved model or `healthwise.csv` found next to app.py — upload the dataset to train one.")
    uploaded_file = st.file_uploader("Upload healthwise.csv", type="csv")
    if uploaded_file is not None:
        data_source = uploaded_file
        model = get_model(uploaded_file)
        joblib.dump(model, MODEL_PATH)

if model is None:
    st.stop()

df_raw = get_raw_data(data_source)

tab_predict, tab_insights, tab_performance = st.tabs(["🔮 Predict", "📊 Data Insights", "📈 Model Performance"])

# ---------------------------------------------------------------------------
# Tab 1: Predict a premium for a single customer profile, plus a what-if tool
# ---------------------------------------------------------------------------
with tab_predict:
    col_form, col_result = st.columns([1, 1.3])

    with col_form:
        st.subheader("Customer profile")
        age = st.slider("Age", 18, 80, 35)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, step=0.1)
        children = st.slider("Children", 0, 6, 0)
        exercise_freq = st.slider("Exercise frequency (days / week)", 0, 7, 3)
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["north", "south", "east", "west"])

    profile = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "exercise_freq": exercise_freq,
        "sex": sex,
        "smoker": smoker,
        "region": region,
    }

    with col_result:
        st.subheader("Prediction")
        tier, premium = predict(model, profile)
        c1, c2 = st.columns(2)
        c1.metric("Risk tier", f"{TIER_EMOJI.get(tier, '')} {tier}")
        c2.metric("Estimated annual premium", f"${premium:,.2f}")

        st.markdown("---")
        st.subheader("What-if: change one factor")
        wf_key = st.selectbox(
            "Factor to change",
            ["smoker", "age", "bmi", "children", "exercise_freq", "region", "sex"],
        )

        wf_profile = dict(profile)
        if wf_key == "age":
            wf_profile["age"] = st.slider("New age", 18, 80, min(age + 10, 80), key="wf_age")
        elif wf_key == "bmi":
            wf_profile["bmi"] = st.slider("New BMI", 15.0, 50.0, min(bmi + 5.0, 50.0), step=0.1, key="wf_bmi")
        elif wf_key == "children":
            wf_profile["children"] = st.slider("New children", 0, 6, min(children + 1, 6), key="wf_children")
        elif wf_key == "exercise_freq":
            wf_profile["exercise_freq"] = st.slider("New exercise frequency", 0, 7, min(exercise_freq + 2, 7), key="wf_ex")
        elif wf_key == "smoker":
            wf_profile["smoker"] = "yes" if smoker == "no" else "no"
            st.caption(f"Toggled smoker: **{smoker} → {wf_profile['smoker']}**")
        elif wf_key == "region":
            options = ["north", "south", "east", "west"]
            wf_profile["region"] = st.selectbox("New region", options, index=(options.index(region) + 1) % 4, key="wf_region")
        elif wf_key == "sex":
            wf_profile["sex"] = "female" if sex == "male" else "male"
            st.caption(f"Toggled sex: **{sex} → {wf_profile['sex']}**")

        wf_tier, wf_premium = predict(model, wf_profile)
        d1, d2 = st.columns(2)
        d1.metric(
            "New risk tier",
            f"{TIER_EMOJI.get(wf_tier, '')} {wf_tier}",
            delta=None if wf_tier == tier else f"{tier} → {wf_tier}",
        )
        d2.metric("New premium", f"${wf_premium:,.2f}", delta=f"${wf_premium - premium:,.2f}")

# ---------------------------------------------------------------------------
# Tab 2: Data insights (only available when the raw dataset is on hand)
# ---------------------------------------------------------------------------
with tab_insights:
    if df_raw is None:
        st.info("Upload or place `healthwise.csv` alongside app.py to see dataset insights.")
    else:
        st.subheader("Annual charge distribution")
        fig = px.histogram(df_raw, x="annual_charge", nbins=40, marginal="box")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Charge by smoker status")
            fig2 = px.box(df_raw, x="smoker", y="annual_charge", color="smoker")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.subheader("Charge by risk tier")
            fig3 = px.box(df_raw, x="risk_tier", y="annual_charge", color="risk_tier", category_orders={"risk_tier": ["Low", "Medium", "High"]})
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Risk tier distribution")
        tier_counts = df_raw["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])
        fig4 = px.bar(tier_counts, color=tier_counts.index, color_discrete_map=TIER_COLORS)
        st.plotly_chart(fig4, use_container_width=True)

        with st.expander("Preview raw data"):
            st.dataframe(df_raw.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Model performance metrics captured at training time
# ---------------------------------------------------------------------------
with tab_performance:
    metrics = model.get("metrics")
    if not metrics:
        st.info("No stored metrics for this model artifact — retrain with train_model.py to populate them.")
    else:
        st.subheader("Stage 1 — Risk tier classifier")
        st.metric("Held-out accuracy", f"{metrics['classifier_accuracy']:.1%}")

        st.subheader("Stage 2 — Per-tier premium regression")
        reg_df = pd.DataFrame(metrics["regression"]).T
        reg_df.index.name = "risk_tier"
        reg_df = reg_df.rename(columns={"r2": "R²", "mae": "MAE ($)", "n": "n (tier size)"})
        st.dataframe(
            reg_df.style.format({"R²": "{:.3f}", "MAE ($)": "${:,.0f}", "n (tier size)": "{:.0f}"}),
            use_container_width=True,
        )
        fig5 = px.bar(reg_df.reset_index(), x="risk_tier", y="R²", color="risk_tier", color_discrete_map=TIER_COLORS)
        st.plotly_chart(fig5, use_container_width=True)
