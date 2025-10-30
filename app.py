import streamlit as st
import numpy as np
import joblib
import re
import cv2
import pytesseract
from PIL import Image
from pathlib import Path

from text_preprocessing import preprocess_text
from ocr_simple import run_ocr_from_ui, preprocess_for_cli_match

OCR_REGEX_CORRECTIONS = [
    (re.compile(r'\b([A-Z]{2,})1([A-Z]{2,})\b'), r'\1I\2'),
]

def _apply_regex_corrections(text: str) -> str:
    for pattern, replacement in OCR_REGEX_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


def _normalize_section_headers(text: str) -> str:
    text = re.sub(r'INGREDIENTS\s*(?![:])', 'INGREDIENTS: ', text)
    text = re.sub(r'MAY CONTAIN\s*(?![:])', 'MAY CONTAIN: ', text)
    text = re.sub(r'CONTAINS\s+(?!LESS|UP TO|\d|[:])', 'CONTAINS: ', text)
    return text


def _remove_consecutive_duplicates(text: str) -> str:
    parts = text.split()
    cleaned_parts = []
    previous_key = None

    for part in parts:
        key = re.sub(r'[^A-Z0-9%]', '', part)
        if key and key == previous_key:
            continue
        cleaned_parts.append(part)
        previous_key = key

    return ' '.join(cleaned_parts)


def _tidy_punctuation_and_spacing(text: str) -> str:
    text = re.sub(r'(?<![A-Z])\s+([,.;)])', r'\1', text)
    return text.strip(' ,;')


def clean_ocr_text(raw_text: str) -> str:
    """Bersihkan dan format ulang teks OCR agar menyerupai label asli."""
    if not raw_text:
        return ""

    text = str(raw_text)
    text = text.replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.upper()

    text = _apply_regex_corrections(text)

    text = re.sub(r'\bAI\b', '(', text)
    text = re.sub(r'\bA1\b', '(', text)
    text = re.sub(r'\bI\)', ')', text)
    text = text.replace(' )', ')').replace('( ', '(')

    text = text.replace('..', '.')
    text = text.replace(',,', ',')

    text = _remove_consecutive_duplicates(text)
    text = _normalize_section_headers(text)
    text = _tidy_punctuation_and_spacing(text)

    if text and not text.endswith('.'):
        text += '.'

    return text

def extract_text_from_image(image_file, return_debug=False):
    """
    Ekstraksi teks dari gambar dengan preprocessing minimal untuk match CLI behavior.
    Menggunakan simplified pipeline yang produces hasil identical dengan:
        tesseract input.png stdout -l eng --psm 6
    """
    try:
        if hasattr(image_file, "seek"):
            image_file.seek(0)
        
        # Read image bytes
        image_bytes = image_file.read()
        
        # Use the simplified OCR function
        text = run_ocr_from_ui(image_bytes)
        
        # Debug data (minimal)
        if return_debug:
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            preprocessed_image = preprocess_for_cli_match(img_bgr)
            
            debug_data = {
                'original': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                'preprocessed': cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB),
                'text_length': len(text),
                'word_count': len(text.split())
            }
            return text, debug_data
        
        return text

    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        if return_debug:
            return "", {'error': str(e)}
        return ""


def clean_ocr_text(text):
    """Clean OCR text output"""
    if not text:
        return ""
    # Just basic cleanup
    text = text.strip()
    return text


@st.cache_resource
def load_model_and_vectorizer():
    """
    Load model terbaik dan TF-IDF vectorizer
    
    Returns:
    --------
    tuple : (model, vectorizer, model_name, accuracy)
    """
    try:
        best_model_path = Path('models/best_model.joblib')
        best_model_data = joblib.load(best_model_path)

        model = best_model_data['model']
        vectorizer = best_model_data['vectorizer']
        model_name = best_model_data.get('model_name', 'Unknown Model')
        accuracy = best_model_data.get('accuracy', 0.0)

        return model, vectorizer, model_name, accuracy
    
    except FileNotFoundError:
        st.error("⚠️ Model belum dilatih! Silakan jalankan `train_models.py` dan `evaluate_and_predict.py` terlebih dahulu.")
        return None, None, None, None


def predict_halal_status(text: str, model, vectorizer):
    """Prediksi status halal/haram dari teks tunggal."""
    clean_text = preprocess_text(text)
    if not clean_text:
        return "Unknown", 0.0

    text_vector = vectorizer.transform([clean_text])
    prediction = model.predict(text_vector)[0]

    confidence = None
    try:
        proba = model.predict_proba(text_vector)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = proba[class_index]
    except AttributeError:
        try:
            decision = model.decision_function(text_vector)[0]
            confidence = 1 / (1 + np.exp(-decision))
            if prediction.lower() == "halal":
                confidence = 1 - confidence
        except Exception:
            confidence = None

    return prediction.capitalize(), confidence


def process_input(input_data, input_type, model, vectorizer, return_debug=False):
    """Proses input teks atau gambar sebelum diprediksi."""
    result = {
        "input_type": input_type,
        "original_text": "",
        "ocr_text": "",
        "preprocessed_text": "",
        "prediction": "",
        "confidence": 0.0,
        "ocr_debug": None,
    }

    if input_type == "image":
        ocr_output = extract_text_from_image(input_data, return_debug=return_debug)
        if return_debug:
            ocr_text, debug_data = ocr_output
            result["ocr_debug"] = debug_data
        else:
            ocr_text = ocr_output
        result["ocr_text"] = ocr_text
        result["original_text"] = ocr_text
        text_to_process = ocr_text
    else:
        result["original_text"] = input_data
        text_to_process = input_data

    preprocessed = preprocess_text(text_to_process)
    result["preprocessed_text"] = preprocessed

    if preprocessed:
        prediction, confidence = predict_halal_status(text_to_process, model, vectorizer)
        result["prediction"] = prediction
        result["confidence"] = confidence
    else:
        result["prediction"] = "Unknown"
        result["confidence"] = 0.0

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
    show_debug = False
    
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

            with col2:
                show_debug = st.checkbox(
                    "Tampilkan debug OCR",
                    value=False,
                    help="Perlihatkan tahapan preprocessing dan area teks yang terdeteksi."
                )
    
    st.markdown("---")
    
    # Tombol Prediksi
    if st.button("🔍 Prediksi Status Halal/Haram", type="primary", use_container_width=True):
        if input_data is None or (input_type == "text" and input_data.strip() == ""):
            st.warning("⚠️ Silakan masukkan teks atau upload gambar terlebih dahulu!")
        else:
            with st.spinner("Memproses..."):
                # Proses input dan prediksi
                result = process_input(
                    input_data,
                    input_type,
                    model,
                    vectorizer,
                    return_debug=show_debug
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

                    if result.get('ocr_debug'):
                        debug_data = result['ocr_debug']
                        with st.expander("🔍 Debug OCR"):
                            st.caption("Tahapan preprocessing dan area teks terpilih.")
                            if debug_data.get('raw_text'):
                                st.write("**Raw OCR Text:**")
                                st.text(debug_data['raw_text'])
                            if debug_data.get('clean_text'):
                                st.write("**Cleaned OCR Text:**")
                                st.text(debug_data['clean_text'])
                            if debug_data.get('original') is not None:
                                st.image(debug_data['original'], caption="Gambar setelah penyesuaian dimensi", use_column_width=True)
                            if debug_data.get('preprocessed') is not None:
                                st.image(debug_data['preprocessed'], caption="Masker biner untuk OCR", use_column_width=True)
                            if debug_data.get('overlay') is not None:
                                st.image(debug_data['overlay'], caption="Deteksi area teks", use_column_width=True)
                            if debug_data.get('regions'):
                                st.write("Region teks yang dianalisis:")
                                for region in debug_data['regions']:
                                    region_caption = f"Region {region['index']}"
                                    if region.get('text'):
                                        snippet = region['text'].replace('\n', ' ')
                                        if len(snippet) > 80:
                                            snippet = f"{snippet[:77]}..."
                                        region_caption = f"{region_caption} — {snippet}"
                                    st.image(region['image'], caption=region_caption, use_column_width=True)
                
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
