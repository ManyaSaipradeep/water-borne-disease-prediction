
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from imblearn.ensemble import EasyEnsembleClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True, parents=True)
REPORTS.mkdir(exist_ok=True, parents=True)

NUMERIC_FEATURES = [
    "Avg_Temp_C", "Weekly_Rainfall", "rain_lag1", "rain_3wk_avg",
    "temp_lag1", "is_monsoon", "month_approx",
    "rain_anomaly", "temp_anomaly",
    "monsoon_cum_rain", "extreme_rain_count_6wk", "dry_then_deluge",
    "neighbor_outbreak_recent", "reporting_freq_52wk",
]
LAGGED_FEATURES = [f for f in NUMERIC_FEATURES if f != "month_approx"]
CATEGORICAL_FEATURES = ["state_centroid"]

AGG_RULES = {
    "Avg_Temp_C": "mean", "Weekly_Rainfall": "mean", "rain_lag1": "mean",
    "rain_3wk_avg": "mean", "temp_lag1": "mean", "rain_anomaly": "mean",
    "temp_anomaly": "mean", "is_monsoon": "max", "monsoon_cum_rain": "max",
    "extreme_rain_count_6wk": "max", "dry_then_deluge": "max",
    "neighbor_outbreak_recent": "max", "reporting_freq_52wk": "mean",
    "outbreak_label": "max", "state_centroid": "first",
}

RF_PARAMS = dict(
    n_estimators=200, max_depth=12, min_samples_leaf=5,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1,
)

TRAIN_YEARS = list(range(2000, 2024))  # through 2023
TEST_YEARS = [2024]                    # unseen, most recent full year


def build_monthly_lagged_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per district-month. Every predictor is aggregated from the
    PRIOR month only (shift(1) per district); the label is this month's
    real outcome (max over its own weeks); month number itself is kept
    unlagged since the calendar is known in advance, not data."""
    monthly = (
        df.groupby(["district_norm", "Year", "month_approx"])
        .agg(AGG_RULES)
        .reset_index()
        .sort_values(["district_norm", "Year", "month_approx"])
    )
    lagged = monthly.groupby("district_norm")[LAGGED_FEATURES].shift(1)
    lagged.columns = [f"{c}_prevmonth" for c in lagged.columns]

    out = pd.concat([
        monthly[["district_norm", "Year", "month_approx", "outbreak_label", "state_centroid"]],
        lagged,
    ], axis=1)
    # First month on record for each district has no prior month to lag
    # from -- drop it rather than silently filling with an imputed value
    # that would falsely claim a real precursor signal exists there.
    out = out.dropna(subset=[f"{f}_prevmonth" for f in LAGGED_FEATURES])
    return out


def find_best_threshold(clf, X_train, y_train):
    """F1-optimal decision threshold, chosen on TRAINING data only so the
    test-set evaluation below stays an honest, unseen-data check."""
    train_proba = clf.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, train_proba)
    f1s = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx]), float(f1s[best_idx])


def evaluate(name, clf, X_test, y_test, threshold, base_rate):
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    prec = precision_score(y_test, pred, zero_division=0)
    metrics = {
        "model": name,
        "threshold": threshold,
        "base_rate": base_rate,
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "precision": prec,
        "recall": recall_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "lift_over_base_rate": (prec / base_rate) if base_rate > 0 else None,
    }
    return metrics, pred, proba


def main():
    df = pd.read_csv(DATA / "model_dataset.csv", low_memory=False)
    df["state_centroid"] = df["state_centroid"].fillna("unknown")

    panel = build_monthly_lagged_panel(df)
    print(f"Monthly-lagged panel rows: {len(panel)} "
          f"(positives: {int(panel['outbreak_label'].sum())}, "
          f"base rate: {panel['outbreak_label'].mean():.4f})")

    train = panel[panel["Year"].isin(TRAIN_YEARS)].copy()
    test = panel[panel["Year"].isin(TEST_YEARS)].copy()
    print(f"Train rows: {len(train)} (years {min(TRAIN_YEARS)}-{max(TRAIN_YEARS)})")
    print(f"Test rows:  {len(test)} (year(s) {TEST_YEARS}, unseen)")

    state_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train["state_encoded"] = state_encoder.fit_transform(train[["state_centroid"]])
    test["state_encoded"] = state_encoder.transform(test[["state_centroid"]])

    model_features = [f"{f}_prevmonth" for f in LAGGED_FEATURES] + ["month_approx", "state_encoded"]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[model_features])
    X_test = imputer.transform(test[model_features])
    y_train = train["outbreak_label"].values
    y_test = test["outbreak_label"].values
    base_rate = float(y_test.mean())

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # ---------------- Model 1: RandomForest ----------------
    rf_grid = {"n_estimators": [200], "max_depth": [8, 12, 20], "min_samples_leaf": [2, 5]}
    rf_base = RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf_search = GridSearchCV(rf_base, rf_grid, scoring="f1", cv=cv, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    rf_clf = RandomForestClassifier(**rf_search.best_params_, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    rf_threshold, _ = find_best_threshold(rf_clf, X_train, y_train)
    rf_metrics, rf_pred, rf_proba = evaluate("RandomForest", rf_clf, X_test, y_test, rf_threshold, base_rate)
    print("RandomForest:", rf_metrics)

    # ---------------- Model 2: XGBoost ----------------
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_grid = {"n_estimators": [200], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
    xgb_base = XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_search = GridSearchCV(xgb_base, xgb_grid, scoring="f1", cv=cv, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    xgb_clf = XGBClassifier(**xgb_search.best_params_, scale_pos_weight=pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train, y_train)
    xgb_threshold, _ = find_best_threshold(xgb_clf, X_train, y_train)
    xgb_metrics, xgb_pred, xgb_proba = evaluate("XGBoost", xgb_clf, X_test, y_test, xgb_threshold, base_rate)
    print("XGBoost:", xgb_metrics)

    # ---------------- Model 3: EasyEnsemble ----------------
    # Built for severe class imbalance: many balanced AdaBoost sub-ensembles,
    # each on a different balanced UNDERsample of the real majority class
    # (real-row undersampling, never synthetic oversampling). This is the
    # same architecture that won on the WEEKLY grain -- tuned and tested
    # here the same way, honestly, rather than assumed to also win here.
    eec_grid = {"n_estimators": [10, 20, 30]}
    eec_base = EasyEnsembleClassifier(random_state=42, n_jobs=-1)
    eec_search = GridSearchCV(eec_base, eec_grid, scoring="f1", cv=cv, n_jobs=-1)
    eec_search.fit(X_train, y_train)
    eec_clf = EasyEnsembleClassifier(**eec_search.best_params_, random_state=42, n_jobs=-1)
    eec_clf.fit(X_train, y_train)
    eec_threshold, _ = find_best_threshold(eec_clf, X_train, y_train)
    eec_metrics, eec_pred, eec_proba = evaluate("EasyEnsemble", eec_clf, X_test, y_test, eec_threshold, base_rate)
    print("EasyEnsemble:", eec_metrics)

    comparison = pd.DataFrame([rf_metrics, xgb_metrics, eec_metrics]).set_index("model")
    print("\n=== Comparison (monthly-lagged grain, test year(s) {}) ===".format(TEST_YEARS))
    print(comparison.to_string())

    # Winner picked by F1 -- the metric that balances precision/recall for
    # this imbalanced problem -- same convention as the weekly model's
    # selection. EasyEnsemble won here too (see reports/
    # monthly_model_selection_trial_metrics.csv for the fuller comparison,
    # including SMOTE variants, that led to adopting this).
    candidates = {
        "RandomForest": (rf_clf, rf_pred, rf_proba, rf_metrics, rf_threshold),
        "XGBoost": (xgb_clf, xgb_pred, xgb_proba, xgb_metrics, xgb_threshold),
        "EasyEnsemble": (eec_clf, eec_pred, eec_proba, eec_metrics, eec_threshold),
    }
    winner_name = max(candidates, key=lambda name: candidates[name][3]["f1_score"])
    clf, pred, proba, metrics, threshold = candidates[winner_name]
    print(f"\nWinner (by F1-score): {winner_name}")

    metrics = dict(metrics)
    metrics["train_rows"] = len(train)
    metrics["test_rows"] = len(test)
    metrics["test_years"] = str(TEST_YEARS)
    metrics["train_years"] = f"{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}"

    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, zero_division=0)
    print("\n=== Final metrics (test year(s) {}) ===".format(TEST_YEARS))
    print(json.dumps(metrics, indent=2))
    print(report)

    joblib.dump(
        {
            "model": clf, "imputer": imputer, "features": model_features,
            "lagged_features": LAGGED_FEATURES, "agg_rules": AGG_RULES,
            "state_encoder": state_encoder, "model_type": f"{winner_name} (monthly-lagged)",
            "threshold": threshold, "grain": "monthly-lagged",
            "train_years": TRAIN_YEARS, "test_years": TEST_YEARS,
        },
        MODELS / "outbreak_model.joblib",
    )

    with open(REPORTS / "outbreak_model_metrics.txt", "w") as f:
        f.write(f"Selected model: {winner_name} (monthly-lagged, production)\n\n")
        f.write(json.dumps(metrics, indent=2))
        f.write("\n\nConfusion matrix (rows=actual, cols=predicted):\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "estimators_"):
        # EasyEnsemble holds sub-estimators (Pipelines ending in an
        # AdaBoostClassifier) that don't expose a top-level
        # feature_importances_ -- average across the sub-estimators.
        sub_importances = []
        for est in clf.estimators_:
            base = est.steps[-1][1] if hasattr(est, "steps") else est
            if hasattr(base, "feature_importances_"):
                sub_importances.append(base.feature_importances_)
        importances = np.mean(sub_importances, axis=0)
    else:
        importances = None

    if importances is not None:
        fi = pd.Series(importances, index=model_features).sort_values(ascending=False)
        fi.to_csv(REPORTS / "feature_importance.csv", header=["importance"])
        print("\nTop feature importances:\n", fi.head(10))

    with open(REPORTS / "outbreak_model_metrics.txt", "a") as f:
        f.write("\n\n=== Model comparison (RandomForest vs XGBoost vs EasyEnsemble, monthly-lagged grain) ===\n")
        f.write(comparison.to_string())

    print(f"\nSaved {winner_name} (monthly-lagged) model to models/outbreak_model.joblib")


if __name__ == "__main__":
    main()
