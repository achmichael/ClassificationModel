from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from text_preprocessing import preprocess_text

MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output")


def evaluate_models(models, X_test, y_test, X_train=None, y_train=None, dataset_label: str = "baseline"):
    """
    Fungsi untuk mengevaluasi semua model dan menampilkan confusion matrix
    
    Parameters:
    -----------
    models : dict
        Dictionary berisi {nama_model: model_terlatih}
    X_test : sparse matrix
        Fitur testing dalam bentuk TF-IDF matrix
    y_test : pandas Series atau array
        Label testing
    
    Returns:
    --------
    DataFrame : DataFrame berisi hasil evaluasi semua model
    """
    
    results = []
    print("\n" + "=" * 80)
    print(f"EVALUASI DETAIL SEMUA MODEL ({dataset_label.upper()})")
    print("=" * 80)

    n_models = len(models)
    cols = 2 if n_models > 1 else 1
    rows = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"Confusion Matrix untuk Semua Model ({dataset_label})", fontsize=16, fontweight="bold")

    for idx, (model_name, model) in enumerate(models.items()):
        print(f"\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"{'=' * 80}")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="haram", average="binary")
        recall = recall_score(y_test, y_pred, pos_label="haram", average="binary")
        f1 = f1_score(y_test, y_pred, pos_label="haram", average="binary")

        cm = confusion_matrix(y_test, y_pred, labels=["halal", "haram"])

        if X_train is not None and y_train is not None:
            train_accuracy = accuracy_score(y_train, model.predict(X_train))
            print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
            print(f"Testing Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
            if train_accuracy - accuracy > 0.10:
                print("⚠️  potensi overfitting (selisih > 10%)")
        else:
            print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")

        print("Precision : {:.4f} ({:.2f}%)".format(precision, precision * 100))
        print("Recall    : {:.4f} ({:.2f}%)".format(recall, recall * 100))
        print("F1-Score : {:.4f} ({:.2f}%)".format(f1, f1 * 100))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["halal", "haram"]))

        print("Confusion Matrix:")
        print(f"              Predicted")
        print(f"              halal  haram")
        print(f"Actual halal  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       haram  {cm[1][0]:5d}  {cm[1][1]:5d}")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["halal", "haram"],
            yticklabels=["halal", "haram"],
            ax=axes[idx],
            cbar=True,
        )
        axes[idx].set_title(f"{model_name}\nAccuracy: {accuracy:.4f}", fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("Actual Label")
        axes[idx].set_xlabel("Predicted Label")

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "True Halal": cm[0][0],
                "False Haram": cm[0][1],
                "False Halal": cm[1][0],
                "True Haram": cm[1][1],
            }
        )

    for extra_axis in axes[len(models) :]:
        extra_axis.axis("off")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / f"confusion_matrices_{dataset_label}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Confusion matrices disimpan ke: {plot_path}")
    plt.close()

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("TABEL PERBANDINGAN PERFORMA MODEL")
    print("=" * 80)
    print(results_df.to_string(index=False))

    summary_path = OUTPUT_DIR / f"model_evaluation_results_{dataset_label}.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\n✓ Hasil evaluasi disimpan ke: {summary_path}")

    return results_df


def predict_halal_status(text_input, best_model, vectorizer):
    """
    Fungsi untuk memprediksi status halal/haram dari teks input
    
    Parameters:
    -----------
    text_input : str
        Teks komposisi bahan makanan
    best_model : sklearn model
        Model terbaik yang sudah dilatih
    vectorizer : TfidfVectorizer
        Vectorizer yang sudah di-fit pada data training
    
    Returns:
    --------
    str : 'Halal' atau 'Haram'
    """
    
    # 1. Preprocessing teks
    clean_text = preprocess_text(text_input)
    
    if clean_text == "":
        return "Unknown (teks kosong setelah preprocessing)"
    
    # 2. Transformasi ke TF-IDF vector
    text_vector = vectorizer.transform([clean_text])
    
    # 3. Prediksi
    prediction = best_model.predict(text_vector)[0]
    
    # 4. Return hasil dengan kapitalisasi
    return prediction.capitalize()


def test_prediction_examples(best_model, vectorizer):
    """
    Fungsi untuk testing prediksi dengan contoh-contoh bahan makanan
    
    Parameters:
    -----------
    best_model : sklearn model
        Model terbaik yang sudah dilatih
    vectorizer : TfidfVectorizer
        Vectorizer yang sudah di-fit pada data training
    """
    
    print("\n" + "="*80)
    print("TESTING PREDIKSI DENGAN CONTOH BAHAN MAKANAN")
    print("="*80)
    
    # Contoh bahan halal
    halal_examples = [
        "sugar vegetable oil salt natural flavoring",
        "chicken stock water salt natural flavor",
        "organic wheat flour water sea salt",
        "filtered water apple juice concentrate vitamin c",
        "rice flour palm oil sugar sea salt"
    ]
    
    # Contoh bahan haram
    haram_examples = [
        "pork water salt sugar",
        "beef stock gelatin natural flavor",
        "bacon cured with water salt sodium nitrite",
        "wine vinegar alcohol natural flavor",
        "chicken mechanically separated turkey pork"
    ]
    
    print("\n--- CONTOH BAHAN HALAL ---")
    for i, example in enumerate(halal_examples, 1):
        prediction = predict_halal_status(example, best_model, vectorizer)
        status = "✓" if prediction.lower() == "halal" else "✗"
        print(f"{i}. {status} Input: {example[:60]}...")
        print(f"   Prediksi: {prediction}\n")
    
    print("\n--- CONTOH BAHAN HARAM ---")
    for i, example in enumerate(haram_examples, 1):
        prediction = predict_halal_status(example, best_model, vectorizer)
        status = "✓" if prediction.lower() == "haram" else "✗"
        print(f"{i}. {status} Input: {example[:60]}...")
        print(f"   Prediksi: {prediction}\n")


def main():
    """
    Fungsi utama untuk menjalankan evaluasi dan testing prediksi
    """
    parser = argparse.ArgumentParser(description="Evaluasi model halal/haram")
    parser.add_argument(
        "--dataset",
        choices=["baseline", "augmented"],
        default="baseline",
        help="Pilih artefak dataset yang ingin dievaluasi",
    )
    args = parser.parse_args()

    dataset_label = args.dataset
    models_path = MODELS_DIR / f"trained_models_{dataset_label}.joblib"
    if not models_path.exists():
        raise FileNotFoundError(f"File model {models_path} tidak ditemukan. Jalankan train_models.py terlebih dahulu.")

    print("Loading trained models...")
    models = joblib.load(models_path)
    print(f"✓ {len(models)} models loaded")

    data_path = MODELS_DIR / f"train_test_data_{dataset_label}.joblib"
    print("\nLoading train-test data...")
    data = joblib.load(data_path)
    X_test_tfidf = data["X_test_tfidf"]
    y_test = data["y_test"]
    X_train_tfidf = data.get("X_train_tfidf")
    y_train = data.get("y_train")
    print(f"✓ Test data loaded: {X_test_tfidf.shape[0]} samples")

    vectorizer_path = MODELS_DIR / f"tfidf_vectorizer_{dataset_label}.joblib"
    print("\nLoading TF-IDF vectorizer...")
    vectorizer = joblib.load(vectorizer_path)
    print("✓ Vectorizer loaded")

    results_df = evaluate_models(models, X_test_tfidf, y_test, X_train_tfidf, y_train, dataset_label)

    best_model_name = results_df.iloc[0]["Model"]
    best_accuracy = results_df.iloc[0]["Accuracy"]
    best_model = models[best_model_name]

    print("\n" + "=" * 80)
    print(f"🏆 MODEL TERBAIK: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print("=" * 80)

    test_prediction_examples(best_model, vectorizer)

    best_model_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "vectorizer": vectorizer,
        "accuracy": best_accuracy,
        "dataset_label": dataset_label,
    }

    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model_payload, best_model_path)
    print(f"\n✓ Model terbaik disimpan ke: {best_model_path}")

    print("\n" + "=" * 80)
    print("SELESAI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
