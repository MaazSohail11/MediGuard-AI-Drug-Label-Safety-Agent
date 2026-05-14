"""
train_ml_model.py
Trains a Random Forest classifier on mediguard_ml_dataset.csv.
Outputs: models/risk_model.pkl  and  data/processed/ml_metrics.txt

Usage:
    python src/train_ml_model.py
"""

import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT   = Path(__file__).parent.parent
PROC_DIR  = PROJECT / "data" / "processed"
MODEL_DIR = PROJECT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "has_contraindications", "has_warnings", "has_precautions",
    "has_drug_interactions", "has_adverse_reactions",
    "has_pediatric_warning", "has_geriatric_warning",
    "has_pregnancy_warning", "has_overdosage_info",
    "serious_keyword_count", "risk_score",
]
LABEL = "risk_level"


def main():
    csv_path = PROC_DIR / "mediguard_ml_dataset.csv"
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        print("Run  python src/preprocess_json.py  first.")
        sys.exit(1)

    print("Loading dataset …")
    df = pd.read_csv(csv_path)
    print(f"  Total records : {len(df):,}")
    print(f"  Label counts  :\n{df[LABEL].value_counts().to_string()}")

    X = df[FEATURES].fillna(0)
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"\n  Train : {len(X_train):,}   Test : {len(X_test):,}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("\nTraining Random Forest …")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    labels = ["Low", "Moderate", "High"]
    cm     = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\nAccuracy : {acc:.4f}")
    print(report)
    print("Confusion Matrix (Low | Moderate | High):")
    print(cm)

    model_path = MODEL_DIR / "risk_model.pkl"
    joblib.dump(clf, model_path)
    print(f"\n[OK] Model saved : {model_path}")

    metrics_path = PROC_DIR / "ml_metrics.txt"
    fi = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    with open(metrics_path, "w") as f:
        f.write("MediGuard — Random Forest Classifier Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset        : {csv_path}\n")
        f.write(f"Total records  : {len(df):,}\n")
        f.write(f"Train / Test   : {len(X_train):,} / {len(X_test):,}  (70 / 30 split)\n")
        f.write(f"Features       : {FEATURES}\n\n")
        f.write(f"Accuracy       : {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix  (rows = actual, cols = predicted)\n")
        f.write("Labels: Low, Moderate, High\n")
        f.write(str(cm) + "\n\n")
        f.write("Feature Importances:\n")
        f.write(fi.to_string() + "\n")

    print(f"[OK] Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
