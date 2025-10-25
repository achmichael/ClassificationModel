from pathlib import Path

import pandas as pd

from text_preprocessing import preprocess_text


# 1. Load dataset
DATA_PATH = Path('data/cleaned_dataset.csv')
OUTPUT_PATH = Path('data/preprocessed_dataset.csv')

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nKolom dataset: {df.columns.tolist()}")
print(f"\nDistribusi label:")
print(df['label'].value_counts())

# 2. Preprocessing teks
print("\nMelakukan preprocessing teks...")
df['text'] = df['text'].fillna('')
df['clean_text'] = df['text'].apply(preprocess_text)

# 3. Hapus baris dengan clean_text kosong (jika ada)
df = df[df['clean_text'].str.len() > 0]
print(f"\nDataset setelah preprocessing: {df.shape[0]} rows")

# 4. Tampilkan 5 data pertama
print("\n" + "="*80)
print("5 Data Pertama Setelah Preprocessing:")
print("="*80)
for idx, row in df.head().iterrows():
    print(f"\nData #{idx+1}")
    print(f"Label: {row['label']}")
    print(f"Text (original):\n{row['text'][:150]}...")
    print(f"Clean Text:\n{row['clean_text'][:150]}...")
    print("-"*80)

# 5. Simpan hasil preprocessing
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Hasil preprocessing disimpan ke: {OUTPUT_PATH}")

# Statistik tambahan
print(f"\nStatistik Dataset:")
print(f"- Total data: {len(df)}")
print(f"- Halal: {len(df[df['label'] == 'halal'])}")
print(f"- Haram: {len(df[df['label'] == 'haram'])}")
print(f"- Rata-rata panjang teks (kata): {df['clean_text'].apply(lambda x: len(x.split())).mean():.2f}")
print(f"- Min panjang teks (kata): {df['clean_text'].apply(lambda x: len(x.split())).min()}")
print(f"- Max panjang teks (kata): {df['clean_text'].apply(lambda x: len(x.split())).max()}")
