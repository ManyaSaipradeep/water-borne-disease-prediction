
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_train_mod = import_module("03_train_outbreak_model")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True, parents=True)


def main():
    bundle = joblib.load(MODELS / "outbreak_model.joblib")
    clf, imputer, features = bundle["model"], bundle["imputer"], bundle["features"]
    state_encoder = bundle["state_encoder"]
    threshold = bundle.get("threshold", 0.5)
    test_years = bundle.get("test_years", [2024])

    df = pd.read_csv(DATA / "model_dataset.csv", low_memory=False)
    df["state_centroid"] = df["state_centroid"].fillna("unknown")

    panel = _train_mod.build_monthly_lagged_panel(df)
    test = panel[panel["Year"].isin(test_years)].copy()
    test["state_encoded"] = state_encoder.transform(test[["state_centroid"]])

    X_test = imputer.transform(test[features])
    y_test = test["outbreak_label"].values
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Outbreak", "Outbreak"])
    ax.set_yticklabels(["No Outbreak", "Outbreak"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Outbreak Risk Model")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    ax.set_title("ROC Curve — Outbreak Risk Model")
    fig.tight_layout()
    fig.savefig(REPORTS / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Feature importance
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "estimators_"):
        # Ensembles like EasyEnsembleClassifier hold sub-estimators (here,
        # Pipelines ending in an AdaBoostClassifier) that don't expose a
        # top-level feature_importances_ -- average across the sub-estimators.
        import numpy as np
        sub_importances = []
        for est in clf.estimators_:
            base = est.steps[-1][1] if hasattr(est, "steps") else est
            if hasattr(base, "feature_importances_"):
                sub_importances.append(base.feature_importances_)
        importances = np.mean(sub_importances, axis=0)
    else:
        raise AttributeError(f"{type(clf)} has no feature importance available")
    fi = pd.Series(importances, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    fi.tail(12).plot.barh(ax=ax, color="#2b6cb0")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    print("Saved plots to reports/: confusion_matrix.png, roc_curve.png, feature_importance.png")


if __name__ == "__main__":
    main()