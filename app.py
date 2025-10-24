import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import cv2
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

# Konfigurasi Tesseract (sesuaikan path jika diperlukan)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Inisialisasi stemmer dan stopwords
@st.cache_resource
def initialize_nlp_tools():
    """Inisialisasi tools NLP (stemmer dan stopwords)"""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    return stemmer, stop_words

# Dictionary normalisasi
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


def extract_text_from_image(image_file):
    """
    Ekstraksi teks dari gambar menggunakan Tesseract OCR
    
    Parameters:
    -----------
    image_file : UploadedFile atau str
        File gambar atau path ke file gambar
    
    Returns:
    --------
    str : Teks hasil OCR
    """
    try:
        # Baca gambar menggunakan PIL
        if isinstance(image_file, str):
            image = Image.open(image_file)
        else:
            image = Image.open(image_file)
        
        # Konversi ke numpy array untuk OpenCV
        img_array = np.array(image)
        
        # Konversi RGB ke BGR jika perlu (OpenCV menggunakan BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        # Preprocessing gambar untuk meningkatkan akurasi OCR
        # Konversi ke grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Aplikasikan thresholding untuk meningkatkan kontras
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Ekstraksi teks menggunakan Tesseract
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(denoised, config=custom_config)
        
        return text.strip()
    
    except Exception as e:
        st.error(f"Error saat ekstraksi teks dari gambar: {str(e)}")
        return ""


def preprocess_text(text, stemmer, stop_words):
    """
    Fungsi untuk preprocessing teks
    
    Parameters:
    -----------
    text : str
        Teks yang akan diproses
    stemmer : Stemmer
        Sastrawi stemmer
    stop_words : set
        Set stopwords bahasa Indonesia
    
    Returns:
    --------
    str : Teks yang sudah diproses
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


@st.cache_resource
def load_model_and_vectorizer():
    """
    Load model terbaik dan TF-IDF vectorizer
    
    Returns:
    --------
    tuple : (model, vectorizer, model_name, accuracy)
    """
    try:
        with open('models/best_model.pkl', 'rb') as f:
            best_model_data = pickle.load(f)
        
        model = best_model_data['model']
        vectorizer = best_model_data['vectorizer']
        model_name = best_model_data['model_name']
        accuracy = best_model_data['accuracy']
        
        return model, vectorizer, model_name, accuracy
    
    except FileNotFoundError:
        st.error("⚠️ Model belum dilatih! Silakan jalankan `train_models.py` dan `evaluate_and_predict.py` terlebih dahulu.")
        return None, None, None, None


def predict_halal_status(text, model, vectorizer, stemmer, stop_words):
    """
    Prediksi status halal/haram dari teks
    
    Parameters:
    -----------
    text : str
        Teks komposisi produk
    model : sklearn model
        Model yang sudah dilatih
    vectorizer : TfidfVectorizer
        Vectorizer yang sudah di-fit
    stemmer : Stemmer
        Sastrawi stemmer
    stop_words : set
        Set stopwords
    
    Returns:
    --------
    tuple : (prediction, confidence)
    """
    # Preprocessing teks
    clean_text = preprocess_text(text, stemmer, stop_words)
    
    if clean_text == "":
        return "Unknown", 0.0
    
    # Transformasi ke TF-IDF vector
    text_vector = vectorizer.transform([clean_text])
    
    # Prediksi
    prediction = model.predict(text_vector)[0]
    
    # Confidence score (jika model mendukung predict_proba)
    try:
        proba = model.predict_proba(text_vector)[0]
        if prediction.lower() == 'halal':
            confidence = proba[0] if model.classes_[0] == 'halal' else proba[1]
        else:
            confidence = proba[1] if model.classes_[1] == 'haram' else proba[0]
    except AttributeError:
        # Model tidak mendukung predict_proba (misalnya SVM)
        # Gunakan decision_function untuk estimasi confidence
        try:
            decision = model.decision_function(text_vector)[0]
            confidence = 1 / (1 + np.exp(-decision))  # Sigmoid
            if prediction.lower() == 'halal':
                confidence = 1 - confidence
        except:
            confidence = None
    
    return prediction.capitalize(), confidence


def process_input(input_data, input_type, model, vectorizer, stemmer, stop_words):
    """
    Memproses input (teks atau gambar) dan mengembalikan hasil prediksi
    
    Parameters:
    -----------
    input_data : str atau UploadedFile
        Data input (teks string atau file gambar)
    input_type : str
        Jenis input: 'text' atau 'image'
    model : sklearn model
        Model yang sudah dilatih
    vectorizer : TfidfVectorizer
        Vectorizer yang sudah di-fit
    stemmer : Stemmer
        Sastrawi stemmer
    stop_words : set
        Set stopwords
    
    Returns:
    --------
    dict : Dictionary berisi hasil pemrosesan dan prediksi
    """
    result = {
        'input_type': input_type,
        'original_text': '',
        'ocr_text': '',
        'preprocessed_text': '',
        'prediction': '',
        'confidence': 0.0
    }
    
    # [1] & [2] Extract teks dari input
    if input_type == 'image':
        # Extract teks dari gambar menggunakan OCR
        ocr_text = extract_text_from_image(input_data)
        result['ocr_text'] = ocr_text
        result['original_text'] = ocr_text
        text_to_process = ocr_text
    else:
        # Gunakan teks langsung
        result['original_text'] = input_data
        text_to_process = input_data
    
    # [3] Preprocessing teks
    preprocessed = preprocess_text(text_to_process, stemmer, stop_words)
    result['preprocessed_text'] = preprocessed
    
    # [4] & [5] Prediksi menggunakan model
    if preprocessed:
        prediction, confidence = predict_halal_status(
            text_to_process, model, vectorizer, stemmer, stop_words
        )
        result['prediction'] = prediction
        result['confidence'] = confidence
    else:
        result['prediction'] = 'Unknown'
        result['confidence'] = 0.0
    
    return result


def main():
    """
    Fungsi utama untuk Streamlit app
    """
    # Konfigurasi halaman
    st.set_page_config(
        page_title="Klasifikasi Halal-Haram",
        page_icon="🍽️",
        layout="wide"
    )
    
    # Header
    st.title("🍽️ Sistem Klasifikasi Halal-Haram")
    st.markdown("""
    Sistem ini dapat mengklasifikasikan produk makanan sebagai **HALAL** atau **HARAM** 
    berdasarkan komposisi bahannya menggunakan Machine Learning.
    """)
    
    st.markdown("---")
    
    # Load model dan tools
    with st.spinner("Loading model dan tools..."):
        model, vectorizer, model_name, accuracy = load_model_and_vectorizer()
        stemmer, stop_words = initialize_nlp_tools()
    
    if model is None:
        st.stop()
    
    # Tampilkan info model
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Model:** {model_name}")
    with col2:
        st.info(f"**Akurasi:** {accuracy*100:.2f}%")
    
    st.markdown("---")
    
    # Pilihan input method
    st.subheader("📝 Pilih Metode Input")
    input_method = st.radio(
        "Pilih jenis input:",
        ["Teks Manual", "Upload Gambar Label Produk"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Input area
    input_data = None
    input_type = None
    
    if input_method == "Teks Manual":
        st.subheader("✍️ Input Komposisi Produk")
        input_data = st.text_area(
            "Masukkan komposisi/ingredients produk:",
            height=150,
            placeholder="Contoh: sugar, wheat flour, vegetable oil, salt, natural flavoring..."
        )
        input_type = "text"
        
    else:  # Upload Gambar
        st.subheader("📷 Upload Gambar Label Produk")
        uploaded_file = st.file_uploader(
            "Pilih gambar label produk (JPG, PNG, JPEG):",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)
            
            input_data = uploaded_file
            input_type = "image"
    
    st.markdown("---")
    
    # Tombol Prediksi
    if st.button("🔍 Prediksi Status Halal/Haram", type="primary", use_container_width=True):
        if input_data is None or (input_type == "text" and input_data.strip() == ""):
            st.warning("⚠️ Silakan masukkan teks atau upload gambar terlebih dahulu!")
        else:
            with st.spinner("Memproses..."):
                # Proses input dan prediksi
                result = process_input(
                    input_data, input_type, model, vectorizer, stemmer, stop_words
                )
                
                # Tampilkan hasil
                st.markdown("---")
                st.subheader("📊 Hasil Analisis")
                
                # Tampilkan teks OCR jika input adalah gambar
                if result['input_type'] == 'image':
                    st.markdown("#### 📝 Teks Hasil OCR:")
                    if result['ocr_text']:
                        st.text_area("Teks yang terdeteksi:", result['ocr_text'], height=100, disabled=True)
                    else:
                        st.warning("⚠️ Tidak ada teks yang terdeteksi dari gambar")
                
                # Tampilkan hasil preprocessing
                with st.expander("🔧 Lihat Teks Setelah Preprocessing"):
                    st.text(result['preprocessed_text'] if result['preprocessed_text'] else "Teks kosong setelah preprocessing")
                
                # Tampilkan hasil prediksi
                st.markdown("#### 🎯 Hasil Klasifikasi:")
                
                if result['prediction'] != 'Unknown':
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Tampilkan status dengan warna
                        if result['prediction'].lower() == 'halal':
                            st.success(f"### ✅ {result['prediction'].upper()}")
                        else:
                            st.error(f"### ❌ {result['prediction'].upper()}")
                    
                    with col2:
                        # Tampilkan confidence score
                        if result['confidence'] is not None:
                            st.metric(
                                "Confidence Score",
                                f"{result['confidence']*100:.2f}%"
                            )
                            
                            # Progress bar untuk confidence
                            st.progress(result['confidence'])
                        else:
                            st.info("Model tidak mendukung confidence score")
                    
                    # Penjelasan tambahan
                    st.markdown("---")
                    st.markdown("#### ℹ️ Informasi Tambahan:")
                    
                    if result['prediction'].lower() == 'halal':
                        st.success("""
                        **Produk ini diprediksi HALAL** berdasarkan komposisi yang dianalisis. 
                        Namun, tetap disarankan untuk memverifikasi dengan sertifikat halal resmi dari MUI atau lembaga serupa.
                        """)
                    else:
                        st.error("""
                        **Produk ini diprediksi HARAM** berdasarkan komposisi yang dianalisis. 
                        Kemungkinan mengandung bahan-bahan seperti: babi (pork), gelatin, alkohol, atau turunannya.
                        """)
                else:
                    st.warning("⚠️ Tidak dapat melakukan prediksi. Pastikan input berisi informasi komposisi produk.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>💡 <b>Catatan:</b> Hasil prediksi ini adalah estimasi berdasarkan Machine Learning. 
    Untuk kepastian status halal, silakan cek sertifikat halal resmi dari MUI.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
