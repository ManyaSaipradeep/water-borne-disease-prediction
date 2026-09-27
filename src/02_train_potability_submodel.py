
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True, parents=True)
REPORTS.mkdir(exist_ok=True, parents=True)

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]


def main():
    df = pd.read_csv(RAW / "water_potability.csv")
    X = df[FEATURES]
    y = df["Potability"]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred)

    print(f"Potability sub-model -- Accuracy: {acc:.3f}  F1: {f1:.3f}")
    print(report)

    joblib.dump({"model": clf, "imputer": imputer, "features": FEATURES},
                MODELS / "potability_submodel.joblib")

    with open(REPORTS / "potability_submodel_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1-score: {f1:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    print("Saved potability sub-model to models/potability_submodel.joblib")


if __name__ == "__main__":
    main()
