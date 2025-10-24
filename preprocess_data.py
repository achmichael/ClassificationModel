import pandas as pd
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download stopwords jika belum ada
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load stopwords bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Dictionary untuk normalisasi kata-kata terkait babi dan produk haram
normalization_dict = {
    'pork': 'babi',
    'pig': 'babi',
    'gelatin': 'babi',
    'gelatine': 'babi',
    'lard': 'babi',
    'bacon': 'babi',
    'ham': 'babi',
    'swine': 'babi',
    'mechanically separated chicken': 'ayam mekanis',
    'mechanically separated turkey': 'kalkun mekanis',
    'alcohol': 'alkohol',
    'wine': 'anggur alkohol',
    'beer': 'bir',
    'rum': 'rum',
    'vodka': 'vodka',
    'sake': 'sake',
    'bourbon': 'bourbon',
    'rennet': 'rennet',
    'lipase': 'lipase',
    'pepsin': 'pepsin',
    'tallow': 'lemak hewan',
    'animal fat': 'lemak hewan',
    'beef fat': 'lemak sapi',
    'chicken fat': 'lemak ayam',
    'duck fat': 'lemak bebek'
}

def preprocess_text(text):
    """
    Fungsi untuk preprocessing teks
    """
    if pd.isna(text):
        return ""
    
    # Ubah ke huruf kecil
    text = text.lower()
    
    # Normalisasi kata-kata asing yang terkait dengan bahan haram
    for foreign_word, indonesian_word in normalization_dict.items():
        text = text.replace(foreign_word, indonesian_word)
    
    # Hapus tanda baca dan angka (kecuali spasi)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenisasi
    words = text.split()
    
    # Hapus stopwords
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    words = [stemmer.stem(word) for word in words]
    
    # Gabungkan kembali
    clean_text = ' '.join(words)
    
    return clean_text


# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv('data/cleaned_dataset.csv')

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nKolom dataset: {df.columns.tolist()}")
print(f"\nDistribusi label:")
print(df['label'].value_counts())

# 2. Preprocessing teks
print("\nMelakukan preprocessing teks...")
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
output_file = 'data/preprocessed_dataset.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Hasil preprocessing disimpan ke: {output_file}")

# Statistik tambahan
print(f"\nStatistik Dataset:")
print(f"- Total data: {len(df)}")
print(f"- Halal: {len(df[df['label'] == 'halal'])}")
print(f"- Haram: {len(df[df['label'] == 'haram'])}")
print(f"- Rata-rata panjang teks (kata): {df['clean_text'].apply(lambda x: len(x.split())).mean():.2f}")
print(f"- Min panjang teks (kata): {df['clean_text'].apply(lambda x: len(x.split())).min()}")
print(f"- Max panjang teks (kata): {df['clean_text'].apply(lambda x: len(x.split())).max()}")
