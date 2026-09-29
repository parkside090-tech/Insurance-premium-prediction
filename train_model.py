"""
train_model.py
---------------
Trains the two-stage insurance premium model used by the Streamlit dashboard:

  Stage 1: DecisionTreeClassifier  -> predicts risk_tier (Low / Medium / High)
  Stage 2: LinearRegression (one per tier) -> predicts annual_charge (premium)

Run directly to (re)build healthwise_model.joblib from healthwise.csv:

    python train_model.py

The dashboard (app.py) will also call these functions automatically the
first time it runs if no saved model is found.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Features used by the model (matches the source notebook's final pipeline)
NUM = ["age", "bmi", "children", "exercise_freq"]
CAT = ["sex", "smoker", "region"]
FEATURES = NUM + CAT
TIERS = ["Low", "Medium", "High"]

MODEL_PATH = "healthwise_model.joblib"
DATA_PATH = "healthwise.csv"


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply the same cleaning steps as the notebook:
    drop duplicate rows and median-impute the two columns that have gaps."""
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["exercise_freq"] = df["exercise_freq"].fillna(df["exercise_freq"].median())
    return df


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", "passthrough", NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT),
        ]
    )


def train(df: pd.DataFrame) -> dict:
    """Fit the classifier + per-tier regressors and collect held-out metrics."""
    med = {c: float(df[c].median()) for c in NUM}

    # ---- Stage 1: risk-tier classifier (fit on ALL data for production use) ----
    pre = _build_preprocessor()
    clf = Pipeline([("pre", pre), ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))])
    clf.fit(df[FEATURES], df["risk_tier"])

    # Held-out accuracy estimate
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        df[FEATURES], df["risk_tier"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["risk_tier"]
    )
    clf_eval = Pipeline([("pre", _build_preprocessor()), ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))])
    clf_eval.fit(Xtr_c, ytr_c)
    clf_acc = accuracy_score(yte_c, clf_eval.predict(Xte_c))

    # ---- Stage 2: one LinearRegression per risk tier ----
    regs, reg_metrics = {}, {}
    for t in TIERS:
        sub = df[df["risk_tier"] == t]
        if len(sub) < 10:
            continue
        Xtr, Xte, ytr, yte = train_test_split(sub[FEATURES], sub["annual_charge"], test_size=0.2, random_state=RANDOM_STATE)

        eval_reg = Pipeline([("pre", _build_preprocessor()), ("model", LinearRegression())]).fit(Xtr, ytr)
        pred = eval_reg.predict(Xte)
        reg_metrics[t] = {
            "r2": float(r2_score(yte, pred)),
            "mae": float(mean_absolute_error(yte, pred)),
            "n": int(len(sub)),
        }

        # Refit on the full tier's data for the production model
        full_reg = Pipeline([("pre", _build_preprocessor()), ("model", LinearRegression())]).fit(sub[FEATURES], sub["annual_charge"])
        regs[t] = full_reg

    artifact = {
        "clf": clf,
        "regs": regs,
        "features": FEATURES,
        "num": NUM,
        "cat": CAT,
        "med": med,
        "metrics": {"classifier_accuracy": float(clf_acc), "regression": reg_metrics},
    }
    return artifact


def predict(model: dict, customer: dict):
    """Run a single customer profile through the two-stage model.
    Returns (risk_tier, predicted_premium)."""
    row = pd.DataFrame([customer])
    for k in model["num"]:
        if k not in row or pd.isna(row.at[0, k]):
            row[k] = model["med"][k]

    tier = model["clf"].predict(row[model["features"]])[0]
    if tier not in model["regs"]:
        # Fallback: not enough data to fit that tier's regressor
        return tier, float("nan")
    premium = float(model["regs"][tier].predict(row[model["features"]])[0])
    return tier, round(premium, 2)


def main():
    df = load_and_clean()
    artifact = train(df)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved {MODEL_PATH}")
    print(f"Classifier accuracy: {artifact['metrics']['classifier_accuracy']:.3f}")
    for t, m in artifact["metrics"]["regression"].items():
        print(f"  {t:6}: R2={m['r2']:.3f} | MAE=${m['mae']:,.0f} | n={m['n']}")


if __name__ == "__main__":
    main()
