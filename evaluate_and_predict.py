import pandas as pd
import numpy as np
import pickle
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer dan stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Dictionary normalisasi (sama dengan preprocess_data.py)
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
    Fungsi untuk preprocessing teks (sama dengan di preprocess_data.py)
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Ubah ke huruf kecil
    text = text.lower()
    
    # Normalisasi kata-kata asing
    for foreign_word, indonesian_word in normalization_dict.items():
        text = text.replace(foreign_word, indonesian_word)
    
    # Hapus tanda baca dan angka
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


def evaluate_models(models, X_test, y_test):
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
    
    print("\n" + "="*80)
    print("EVALUASI DETAIL SEMUA MODEL")
    print("="*80)
    
    # Buat figure untuk confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrix untuk Semua Model', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='haram', average='binary')
        recall = recall_score(y_test, y_pred, pos_label='haram', average='binary')
        f1 = f1_score(y_test, y_pred, pos_label='haram', average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=['halal', 'haram'])
        
        # Tampilkan metrik
        print(f"\nMetrik Evaluasi:")
        print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision : {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall    : {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score  : {f1:.4f} ({f1*100:.2f}%)")
        
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              halal  haram")
        print(f"Actual halal  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       haram  {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['halal', 'haram'],
                    yticklabels=['halal', 'haram'],
                    ax=axes[idx], cbar=True)
        axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.4f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual Label')
        axes[idx].set_xlabel('Predicted Label')
        
        # Simpan hasil ke list
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'True Halal': cm[0][0],
            'False Haram': cm[0][1],
            'False Halal': cm[1][0],
            'True Haram': cm[1][1]
        })
    
    # Simpan plot confusion matrices
    plt.tight_layout()
    plt.savefig('output/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrices disimpan ke: output/confusion_matrices.png")
    plt.close()
    
    # Buat DataFrame hasil evaluasi
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    # Tampilkan tabel hasil
    print("\n" + "="*80)
    print("TABEL PERBANDINGAN PERFORMA MODEL")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Simpan hasil ke CSV
    results_df.to_csv('output/model_evaluation_results.csv', index=False)
    print(f"\n✓ Hasil evaluasi disimpan ke: output/model_evaluation_results.csv")
    
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
    
    # Load trained models
    print("Loading trained models...")
    with open('models/trained_models.pkl', 'rb') as f:
        models = pickle.load(f)
    print(f"✓ {len(models)} models loaded")
    
    # Load train-test data
    print("\nLoading train-test data...")
    with open('models/train_test_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test_tfidf = data['X_test_tfidf']
    y_test = data['y_test']
    print(f"✓ Test data loaded: {X_test_tfidf.shape[0]} samples")
    
    # Load TF-IDF vectorizer
    print("\nLoading TF-IDF vectorizer...")
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✓ Vectorizer loaded")
    
    # Evaluate all models
    results_df = evaluate_models(models, X_test_tfidf, y_test)
    
    # Tentukan model terbaik
    best_model_name = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Accuracy']
    best_model = models[best_model_name]
    
    print("\n" + "="*80)
    print(f"🏆 MODEL TERBAIK: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("="*80)
    
    # Test prediction dengan contoh
    test_prediction_examples(best_model, vectorizer)
    
    # Simpan model terbaik secara terpisah
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_model_name,
            'vectorizer': vectorizer,
            'accuracy': best_accuracy
        }, f)
    print(f"\n✓ Model terbaik disimpan ke: models/best_model.pkl")
    
    print("\n" + "="*80)
    print("SELESAI!")
    print("="*80)


if __name__ == "__main__":
    main()
