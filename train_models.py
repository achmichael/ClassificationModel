from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output")
LOG_PATH = Path("training_log.txt")

DATASET_CONFIGS = (
    {
        "label": "baseline",
        "data_path": MODELS_DIR / "train_test_data_baseline.joblib",
        "vectorizer_path": MODELS_DIR / "tfidf_vectorizer_baseline.joblib",
    },
    {
        "label": "augmented",
        "data_path": MODELS_DIR / "train_test_data_augmented.joblib",
        "vectorizer_path": MODELS_DIR / "tfidf_vectorizer_augmented.joblib",
    },
)


def train_models(X_train, y_train):
    models = {}

    print("\n" + "=" * 80)
    print("MELATIH MODEL MACHINE LEARNING")
    print("=" * 80)

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    models["Multinomial Naive Bayes"] = mnb
    print("✓ Multinomial Naive Bayes selesai")

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    models["K-Nearest Neighbor"] = knn
    print("✓ K-Nearest Neighbor selesai")

    svm = LinearSVC(C=1.0, max_iter=3000, random_state=42)
    svm.fit(X_train, y_train)
    models["Support Vector Machine"] = svm
    print("✓ Support Vector Machine selesai")

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    print("✓ Random Forest selesai")

    print(f"\n✓ Total {len(models)} model berhasil dilatih")
    return models


def log_training_result(dataset_label: str, model_name: str, train_acc: float, test_acc: float, train_size: int) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    log_line = (
        f"{timestamp} | dataset={dataset_label} | model={model_name} | "
        f"train_samples={train_size} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(log_line)


def evaluate_models(models, X_train, y_train, X_test, y_test, dataset_label: str) -> Dict[str, dict]:
    results = {}

    print("\n" + "=" * 80)
    print(f"EVALUASI MODEL ({dataset_label.upper()})")
    print("=" * 80)

    for model_name, model in models.items():
        print(f"\n{'=' * 80}")
        print(model_name)
        print(f"{'=' * 80}")

        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        diff = train_accuracy - test_accuracy

        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
        print(f"Testing Accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
        if diff > 0.10:
            print("⚠️  potensi overfitting (selisih > 10%)")

        print("\nClassification Report (Testing Set):")
        print(classification_report(y_test, y_test_pred, target_names=["halal", "haram"]))

        print("Confusion Matrix (Testing Set):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"              Predicted")
        print(f"              halal  haram")
        print(f"Actual halal  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       haram  {cm[1][0]:5d}  {cm[1][1]:5d}")

        log_training_result(dataset_label, model_name, train_accuracy, test_accuracy, X_train.shape[0])

        results[model_name] = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True),
            "confusion_matrix": cm,
        }

    return results


def save_models(models: dict, dataset_label: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / f"trained_models_{dataset_label}.joblib"
    joblib.dump(models, filepath)
    print(f"\n✓ Model tersimpan di {filepath}")
    return filepath


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_by_dataset: Dict[str, dict] = {}
    models_by_dataset: Dict[str, dict] = {}

    for config in DATASET_CONFIGS:
        label = config["label"]
        data_path = config["data_path"]
        if not data_path.exists():
            print(f"\n⚠️  Data {label} belum tersedia. Jalankan feature_extraction.py terlebih dahulu.")
            continue

        data = joblib.load(data_path)
        X_train_tfidf = data["X_train_tfidf"]
        X_test_tfidf = data["X_test_tfidf"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        print(f"\nDataset {label} -> train: {X_train_tfidf.shape[0]}, test: {X_test_tfidf.shape[0]}")

        models = train_models(X_train_tfidf, y_train)
        results = evaluate_models(models, X_train_tfidf, y_train, X_test_tfidf, y_test, label)

        models_by_dataset[label] = models
        results_by_dataset[label] = results

        save_models(models, label)

        summary_rows = []
        for model_name, metrics in results.items():
            weighted_avg = metrics["classification_report"]["weighted avg"]
            summary_rows.append(
                {
                    "Model": model_name,
                    "Train Accuracy": metrics["train_accuracy"],
                    "Test Accuracy": metrics["test_accuracy"],
                    "Precision": weighted_avg["precision"],
                    "Recall": weighted_avg["recall"],
                    "F1": weighted_avg["f1-score"],
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
        summary_path = OUTPUT_DIR / f"model_evaluation_results_{label}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Ringkasan evaluasi disimpan ke {summary_path}")

    if len(results_by_dataset) > 1:
        baseline = results_by_dataset.get("baseline")
        augmented = results_by_dataset.get("augmented")
        if baseline and augmented:
            comparison_rows = []
            for model_name, metrics in baseline.items():
                if model_name not in augmented:
                    continue
                base_weighted = metrics["classification_report"]["weighted avg"]
                aug_weighted = augmented[model_name]["classification_report"]["weighted avg"]
                comparison_rows.append(
                    {
                        "Model": model_name,
                        "Baseline Accuracy": metrics["test_accuracy"],
                        "Augmented Accuracy": augmented[model_name]["test_accuracy"],
                        "Accuracy Δ": augmented[model_name]["test_accuracy"] - metrics["test_accuracy"],
                        "Baseline Precision": base_weighted["precision"],
                        "Augmented Precision": aug_weighted["precision"],
                        "Precision Δ": aug_weighted["precision"] - base_weighted["precision"],
                        "Baseline Recall": base_weighted["recall"],
                        "Augmented Recall": aug_weighted["recall"],
                        "Recall Δ": aug_weighted["recall"] - base_weighted["recall"],
                        "Baseline F1": base_weighted["f1-score"],
                        "Augmented F1": aug_weighted["f1-score"],
                        "F1 Δ": aug_weighted["f1-score"] - base_weighted["f1-score"],
                    }
                )

            if comparison_rows:
                comparison_df = pd.DataFrame(comparison_rows).sort_values("Augmented Accuracy", ascending=False)
                comparison_path = OUTPUT_DIR / "augmentation_comparison.csv"
                comparison_df.to_csv(comparison_path, index=False)
                print("\n" + "=" * 80)
                print("PERBANDINGAN METRIK SEBELUM & SESUDAH AUGMENTASI")
                print("=" * 80)
                print(
                    comparison_df.to_string(
                        index=False,
                        formatters={
                            "Baseline Accuracy": "{:.4f}".format,
                            "Augmented Accuracy": "{:.4f}".format,
                            "Accuracy Δ": "{:+.4f}".format,
                            "Baseline Precision": "{:.4f}".format,
                            "Augmented Precision": "{:.4f}".format,
                            "Precision Δ": "{:+.4f}".format,
                            "Baseline Recall": "{:.4f}".format,
                            "Augmented Recall": "{:.4f}".format,
                            "Recall Δ": "{:+.4f}".format,
                            "Baseline F1": "{:.4f}".format,
                            "Augmented F1": "{:.4f}".format,
                            "F1 Δ": "{:+.4f}".format,
                        },
                    )
                )
                print(f"\n✓ Hasil perbandingan disimpan ke {comparison_path}")

    best_model_info = None
    best_score = -1.0

    for label, results in results_by_dataset.items():
        for model_name, metrics in results.items():
            if metrics["test_accuracy"] > best_score:
                best_score = metrics["test_accuracy"]
                best_model_info = (label, model_name, metrics)

    if best_model_info:
        dataset_label, model_name, metrics = best_model_info
        best_model = models_by_dataset[dataset_label][model_name]
        vectorizer_path = MODELS_DIR / f"tfidf_vectorizer_{dataset_label}.joblib"
        vectorizer = joblib.load(vectorizer_path)

        best_model_payload = {
            "model": best_model,
            "model_name": model_name,
            "vectorizer": vectorizer,
            "accuracy": metrics["test_accuracy"],
            "dataset_label": dataset_label,
        }

        best_model_path = MODELS_DIR / "best_model.joblib"
        joblib.dump(best_model_payload, best_model_path)

        print("\n" + "=" * 80)
        print(f"🏆 MODEL TERBAIK: {model_name} ({dataset_label})")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy'] * 100:.2f}%)")
        print(f"   Train Accuracy: {metrics['train_accuracy']:.4f} ({metrics['train_accuracy'] * 100:.2f}%)")
        print("=" * 80)
        print(f"✓ Model terbaik disimpan di {best_model_path}")


if __name__ == "__main__":
    main()
