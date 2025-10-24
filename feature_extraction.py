import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# 1. Load preprocessed dataset
print("Loading preprocessed dataset...")
df = pd.read_csv('data/preprocessed_dataset.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Pisahkan fitur (X) dan label (y)
print("\nMemisahkan fitur dan label...")
X = df['clean_text']
y = df['label']

print(f"Total samples: {len(X)}")
print(f"Label distribution:")
print(y.value_counts())

# 3. Train-test split (80:20)
print("\nMelakukan train-test split (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Memastikan distribusi label seimbang di train dan test
)

print(f"\nJumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

print(f"\nDistribusi label di data training:")
print(y_train.value_counts())
print(f"\nDistribusi label di data testing:")
print(y_test.value_counts())

# 4. TF-IDF Vectorization
print("\n" + "="*80)
print("Melakukan TF-IDF Vectorization...")
print("="*80)

# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Batasi fitur maksimal
    min_df=2,           # Kata harus muncul minimal di 2 dokumen
    max_df=0.8,         # Kata tidak boleh muncul di lebih dari 80% dokumen
    ngram_range=(1, 2), # Unigram dan bigram
    sublinear_tf=True   # Gunakan skala logaritmik untuk term frequency
)

# Fit dan transform data training
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
print(f"\nTF-IDF fit pada data training")
print(f"Shape X_train_tfidf: {X_train_tfidf.shape}")
print(f"Jumlah fitur yang dihasilkan: {X_train_tfidf.shape[1]}")

# Transform data testing (menggunakan vocabulary dari training)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"Shape X_test_tfidf: {X_test_tfidf.shape}")

# 5. Tampilkan top 20 fitur dengan TF-IDF score tertinggi
print("\n" + "="*80)
print("Top 20 Fitur dengan TF-IDF Score Tertinggi:")
print("="*80)

# Hitung rata-rata TF-IDF score untuk setiap fitur
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = X_train_tfidf.mean(axis=0).A1
top_indices = tfidf_scores.argsort()[-20:][::-1]

for idx, feature_idx in enumerate(top_indices, 1):
    print(f"{idx:2d}. {feature_names[feature_idx]:30s} - Score: {tfidf_scores[feature_idx]:.4f}")

# 6. Simpan TF-IDF vectorizer dan hasil ekstraksi fitur
print("\n" + "="*80)
print("Menyimpan hasil...")
print("="*80)

# Simpan TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("✓ TF-IDF Vectorizer disimpan ke: models/tfidf_vectorizer.pkl")

# Simpan data yang sudah di-split
with open('models/train_test_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train_tfidf': X_train_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_text': X_train,
        'X_test_text': X_test
    }, f)
print("✓ Data train-test disimpan ke: models/train_test_data.pkl")

# 7. Tampilkan contoh transformasi
print("\n" + "="*80)
print("Contoh Transformasi TF-IDF:")
print("="*80)

sample_idx = 0
sample_text = X_train.iloc[sample_idx]
sample_label = y_train.iloc[sample_idx]

print(f"Text: {sample_text[:200]}...")
print(f"Label: {sample_label}")
print(f"\nTop 10 fitur TF-IDF untuk contoh ini:")

# Ambil vector TF-IDF untuk sample ini
sample_vector = X_train_tfidf[sample_idx].toarray()[0]
top_feature_indices = sample_vector.argsort()[-10:][::-1]

for idx, feature_idx in enumerate(top_feature_indices, 1):
    if sample_vector[feature_idx] > 0:
        print(f"{idx:2d}. {feature_names[feature_idx]:30s} - Score: {sample_vector[feature_idx]:.4f}")

print("\n" + "="*80)
print("RINGKASAN:")
print("="*80)
print(f"✓ Data berhasil di-split menjadi train dan test")
print(f"✓ Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"✓ TF-IDF features: {X_train_tfidf.shape[1]} fitur")
print(f"✓ Semua file berhasil disimpan")
print("="*80)
