from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


DATASETS = (
    (Path("data/preprocessed_dataset.csv"), "baseline"),
    (Path("data/augmented_dataset.csv"), "augmented"),
)


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def _summarise_features(vectorizer: TfidfVectorizer, X_train_matrix, top_n: int = 20) -> None:
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X_train_matrix.mean(axis=0).A1
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    for rank, feature_idx in enumerate(top_indices, 1):
        print(f"{rank:2d}. {feature_names[feature_idx]:30s} - Score: {tfidf_scores[feature_idx]:.4f}")


def process_dataset(dataset_path: Path, label: str) -> None:
    if not dataset_path.exists():
        print(f"\n⚠️  Dataset {dataset_path} tidak ditemukan — lewati tahap {label}.")
        return

    print("\n" + "=" * 80)
    print(f"Menyiapkan dataset: {label.upper()}")
    print("=" * 80)

    df = pd.read_csv(dataset_path)
    if "clean_text" not in df:
        raise ValueError(f"Kolom 'clean_text' tidak ditemukan pada {dataset_path}")

    X = df["clean_text"].astype(str)
    y = df["label"]

    print(f"Total samples: {len(X)}")
    print("Distribusi label:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nJumlah data training: {len(X_train)}")
    print(f"Jumlah data testing : {len(X_test)}")

    vectorizer = _vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nTF-IDF shape (train): {X_train_tfidf.shape}")
    print(f"TF-IDF shape (test) : {X_test_tfidf.shape}")

    if label == "baseline":
        print("\nTop 20 fitur (berdasarkan skor rata-rata TF-IDF):")
        _summarise_features(vectorizer, X_train_tfidf)

    models_path = Path("models")
    models_path.mkdir(parents=True, exist_ok=True)

    vectorizer_path = models_path / f"tfidf_vectorizer_{label}.joblib"
    train_test_path = models_path / f"train_test_data_{label}.joblib"

    dump(vectorizer, vectorizer_path)
    dump(
        {
            "X_train_tfidf": X_train_tfidf,
            "X_test_tfidf": X_test_tfidf,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_text": X_train,
            "X_test_text": X_test,
        },
        train_test_path,
    )

    print(f"\n✓ Vectorizer disimpan ke {vectorizer_path}")
    print(f"✓ Data split disimpan ke {train_test_path}")


def main() -> None:
    for dataset_path, label in DATASETS:
        process_dataset(dataset_path, label)


if __name__ == "__main__":
    main()
