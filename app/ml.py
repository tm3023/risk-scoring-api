"""
Core model-training and prediction logic for the Risk Scoring API.

Deliberately kept simple and dependency-light: a logistic regression
pipeline (StandardScaler + LogisticRegression) is the right default for
a general-purpose "upload your data, get a validated risk model" service
-- it trains in seconds even on six-figure row counts, is well-calibrated
out of the box, and is easy to explain to a non-technical client, which
matters more here than squeezing out marginal AUC with a heavier model.
"""
import json
import uuid
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

STORE_DIR = Path(__file__).resolve().parent.parent / "models_store"
STORE_DIR.mkdir(exist_ok=True)


class TrainingError(ValueError):
    """Raised for any user-input problem during training (bad column, etc.)."""


def _select_features(df: pd.DataFrame, target_column: str, feature_columns):
    if target_column not in df.columns:
        raise TrainingError(f"target_column '{target_column}' not found in uploaded data.")

    if feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise TrainingError(f"feature_columns not found in data: {missing}")
        feats = feature_columns
    else:
        # Auto-select: all numeric columns except the target, plus one-hot
        # encoded low-cardinality categoricals (<=10 unique values).
        feats = []
        for col in df.columns:
            if col == target_column:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feats.append(col)
            elif df[col].nunique(dropna=True) <= 10:
                feats.append(col)
        if not feats:
            raise TrainingError("No usable feature columns were found or auto-detected.")
    return feats


def _prepare_matrix(df: pd.DataFrame, feats):
    """One-hot encode categoricals, keep a record of the resulting column
    order so prediction-time inputs can be aligned to the same schema."""
    X = pd.get_dummies(df[feats], drop_first=True)
    return X


def train_model(df: pd.DataFrame, target_column: str, feature_columns=None, model_name: str = None):
    if len(df) < 50:
        raise TrainingError("Need at least 50 rows to train a meaningful model.")

    if target_column not in df.columns:
        raise TrainingError(f"target_column '{target_column}' not found in uploaded data. "
                             f"Available columns: {list(df.columns)}")

    y = df[target_column]
    if y.nunique() != 2:
        raise TrainingError(
            f"target_column '{target_column}' must be binary (found {y.nunique()} unique values). "
            "This service currently trains binary classifiers only."
        )
    # Coerce target to 0/1
    classes = sorted(y.dropna().unique().tolist())
    y_bin = (y == classes[1]).astype(int)

    feats = _select_features(df, target_column, feature_columns)
    X = _prepare_matrix(df, feats)

    # Drop rows with missing values in the working set (documented, not silently lossy)
    work = pd.concat([X, y_bin.rename("__target__")], axis=1).dropna()
    dropped = len(df) - len(work)
    X, y_bin = work.drop(columns="__target__"), work["__target__"]

    if len(work) < 50:
        raise TrainingError("Fewer than 50 complete rows remain after dropping missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.25, random_state=42,
        stratify=y_bin if y_bin.nunique() == 2 and y_bin.value_counts().min() >= 2 else None,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)

    test_proba = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, test_proba) if y_test.nunique() == 2 else None

    threshold = float(np.quantile(test_proba, 0.85))
    pred_labels = (test_proba >= threshold).astype(int)
    precision = precision_score(y_test, pred_labels, zero_division=0)
    recall = recall_score(y_test, pred_labels, zero_division=0)
    brier = brier_score_loss(y_test, test_proba)

    # Calibration table (10 bins)
    calib_df = pd.DataFrame({"pred": test_proba, "actual": y_test.values})
    try:
        calib_df["bin"] = pd.qcut(calib_df["pred"], min(10, calib_df["pred"].nunique()), duplicates="drop")
        calib = calib_df.groupby("bin", observed=True).agg(
            predicted=("pred", "mean"), observed=("actual", "mean"), n=("actual", "size")
        ).round(4)
        calibration_table = calib.reset_index(drop=True).to_dict(orient="records")
    except Exception:
        calibration_table = []

    feature_importance = sorted(
        [{"feature": f, "coefficient": round(float(c), 4)} for f, c in zip(X.columns, clf.coef_[0])],
        key=lambda r: abs(r["coefficient"]), reverse=True,
    )

    model_id = str(uuid.uuid4())[:8]
    bundle = {
        "scaler": scaler,
        "clf": clf,
        "feature_columns": list(X.columns),
        "target_column": target_column,
        "positive_class": classes[1],
        "base_rate": float(y_bin.mean()),
    }
    joblib.dump(bundle, STORE_DIR / f"{model_id}.joblib")

    metadata = {
        "model_id": model_id,
        "name": model_name or f"model-{model_id}",
        "created_at": time.time(),
        "target_column": target_column,
        "positive_class": str(classes[1]),
        "n_rows_used": len(work),
        "n_rows_dropped_missing": int(dropped),
        "n_features": len(X.columns),
        "feature_columns": list(X.columns),
        "base_rate": round(float(y_bin.mean()), 4),
        "validation": {
            "test_set_size": len(y_test),
            "auc_roc": round(float(auc), 4) if auc is not None else None,
            "precision_at_top15pct": round(float(precision), 4),
            "recall_at_top15pct": round(float(recall), 4),
            "brier_score": round(float(brier), 4),
            "calibration_table": calibration_table,
        },
        "feature_importance": feature_importance[:15],
    }
    with open(STORE_DIR / f"{model_id}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def list_models():
    out = []
    for meta_path in sorted(STORE_DIR.glob("*.json")):
        with open(meta_path) as f:
            m = json.load(f)
        out.append({
            "model_id": m["model_id"], "name": m["name"], "created_at": m["created_at"],
            "target_column": m["target_column"], "auc_roc": m["validation"]["auc_roc"],
            "n_rows_used": m["n_rows_used"],
        })
    return out


def get_model_metadata(model_id: str):
    path = STORE_DIR / f"{model_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def delete_model(model_id: str):
    joblib_path = STORE_DIR / f"{model_id}.joblib"
    json_path = STORE_DIR / f"{model_id}.json"
    existed = joblib_path.exists()
    joblib_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)
    return existed


def predict(model_id: str, records: list):
    bundle_path = STORE_DIR / f"{model_id}.joblib"
    if not bundle_path.exists():
        return None
    bundle = joblib.load(bundle_path)

    df = pd.DataFrame(records)
    X = pd.get_dummies(df)
    # Align to training-time schema: add any missing dummy columns as 0,
    # drop any the model has never seen, preserve training column order.
    for col in bundle["feature_columns"]:
        if col not in X.columns:
            X[col] = 0
    X = X[bundle["feature_columns"]]

    Xs = bundle["scaler"].transform(X)
    proba = bundle["clf"].predict_proba(Xs)[:, 1]

    base_rate = bundle["base_rate"]
    results = []
    for i, p in enumerate(proba):
        if p < base_rate * 0.5:
            band = "low"
        elif p < base_rate * 1.5:
            band = "moderate"
        else:
            band = "high"
        results.append({"row": i, "probability": round(float(p), 4), "risk_band": band})
    return results
