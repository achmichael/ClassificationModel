import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import cv2
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words
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

# Kumpulan kata penting untuk memfilter hasil OCR
COMMON_INGREDIENTS = {
    "sugar", "salt", "oil", "flour", "fat", "milk", "water", "protein",
    "carbohydrate", "carbohydrates", "fiber", "fibre", "cholesterol", "vitamin",
    "vitamins", "iron", "ingredient", "ingredients", "contains", "may", "contain",
    "wheat", "soy", "egg", "eggs", "milkfat", "butter", "cream", "powder", "yeast",
    "cocoa", "flavor", "flavour", "extract", "vanilla", "natural", "artificial",
    "citric", "acid", "color", "colour", "spice", "seasoning", "corn", "starch",
    "whey", "gluten", "malt", "barley", "rice", "oats", "palm", "canola",
    "sunflower", "vegetable", "gelatin", "gelatine", "pork", "beef", "chicken",
    "turkey", "fish", "shrimp", "shellfish", "garlic", "onion", "pepper",
    "vinegar", "lemon", "juice", "strawberry", "chocolate", "honey", "coconut",
    "almond", "peanut", "hazelnut", "cashew", "lactose", "sucrose", "fructose",
    "glucose", "sorbitol", "xylitol", "syrup", "palmolein", "shortening",
    "emulsifier", "stabilizer", "preservative", "additive", "sweetener",
    "cookie", "cookies"
}

NUTRITION_TERMS = {
    "nutrition", "facts", "serving", "size", "per", "calorie", "calories",
    "total", "daily", "value", "amount", "energy", "saturated", "trans",
    "polyunsaturated", "monounsaturated", "unsaturated", "fat", "fats", "sodium",
    "potassium", "calcium", "dietary", "fiber", "fibers", "sugars", "sugar",
    "includes", "added", "protein", "proteins", "cholesterol", "carbohydrate",
    "carbohydrates", "percent", "dv", "servings", "per", "container", "portion",
    "nutritionals", "facts", "mg", "daily", "value"
}

FUNCTION_WORDS = {
    "and", "or", "with", "from", "for", "of", "in", "on", "by", "to",
    "as", "at", "the", "this", "that", "made", "using", "based", "are",
    "about", "less", "than", "more", "your", "needs", "higher", "lower",
    "depending", "diet", "per", "each", "their", "may", "be", "is"
}

SIGNAL_TERMS = COMMON_INGREDIENTS.union(
    NUTRITION_TERMS,
    {
        "diet", "needs", "calorie", "calories", "daily", "values", "serving",
        "servings", "size", "container", "amount", "percent", "value",
        "nutrition", "facts", "fat", "saturated", "trans", "polyunsaturated",
        "monounsaturated", "cholesterol", "sodium", "potassium", "carbohydrate",
        "fiber", "sugars", "protein", "vitamin", "calcium", "iron"
    }
)

MEASUREMENT_PATTERN = re.compile(
    r"^\d+(?:,\d{3})*(?:\.\d+)?(?:mg|g|kg|mcg|kj|kcal|cal|%)?$",
    re.IGNORECASE
)


@st.cache_resource
def load_english_vocabulary():
    """Load daftar kata bahasa Inggris dari NLTK (fallback ke set kosong jika tidak tersedia)."""
    try:
        english_vocab = {word.lower() for word in nltk_words.words()}
    except LookupError:
        english_vocab = set()
    return english_vocab


@st.cache_resource
def load_valid_ocr_vocabulary():
    """Siapkan vocabulary gabungan untuk pembersihan teks OCR."""
    english_vocab = load_english_vocabulary()
    combined_vocab = set().union(english_vocab, COMMON_INGREDIENTS, NUTRITION_TERMS, FUNCTION_WORDS)
    return combined_vocab


def clean_ocr_text(text: str) -> str:
    """Bersihkan teks OCR dari karakter acak dan kata yang tidak relevan."""
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Hapus rangkaian huruf tunggal berulang (contoh: "g g g")
    text = re.sub(r'(?:\b[a-z]\b[\s,.;:!?-]*){3,}', ' ', text)

    allowed_vocab = load_valid_ocr_vocabulary()
    cleaned_lines = []

    for raw_line in text.splitlines():
        line = re.sub(r'[^a-z0-9\s/%().,-]', ' ', raw_line)
        line = re.sub(r'[\[\]{}]', ' ', line)
        line = re.sub(r'\s+', ' ', line).strip()
        if not line:
            continue

        valid_tokens = []
        line_has_signal = False

        for token in line.split(' '):
            if not token:
                continue

            cleaned_token = token.strip('()')
            cleaned_token = cleaned_token.rstrip('.,;:')
            if not cleaned_token:
                continue

            if len(cleaned_token) == 1 and not cleaned_token.isdigit():
                continue

            if cleaned_token.isalpha() and len(set(cleaned_token)) == 1 and len(cleaned_token) > 2:
                continue

            normalized = cleaned_token.lower()

            measurement = bool(MEASUREMENT_PATTERN.match(cleaned_token))

            base_candidates = {normalized}
            if len(normalized) > 3:
                base_candidates.update({normalized.rstrip('s'), normalized.rstrip('es')})

            is_known_word = any(base in allowed_vocab for base in base_candidates if base)

            if measurement or is_known_word:
                valid_tokens.append(cleaned_token)
                if measurement or normalized in SIGNAL_TERMS:
                    line_has_signal = True

        if valid_tokens and line_has_signal:
            cleaned_lines.append(' '.join(valid_tokens))

    if not cleaned_lines:
        return ""

    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def extract_text_from_image(image_file, return_debug=False):
    """Ekstraksi teks dari gambar dengan deteksi area teks dan opsi debug."""
    try:
        if hasattr(image_file, "seek"):
            image_file.seek(0)
        pil_image = Image.open(image_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        height, width = img_bgr.shape[:2]
        debug_data = {
            'original': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            'preprocessed': None,
            'overlay': None,
            'regions': []
        }

        if width < 1200:
            scale = 1200.0 / width
            new_size = (int(width * scale), int(height * scale))
            img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_CUBIC)
            debug_data['original'] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        binary_for_ocr = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            3
        )

        binary_inv = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5
        )

        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        connected = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, connect_kernel, iterations=2)

        debug_data['preprocessed'] = cv2.cvtColor(binary_for_ocr, cv2.COLOR_GRAY2RGB)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = img_bgr.shape[0] * img_bgr.shape[1]
        regions = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / float(h) if h else 0.0
            if area < 0.015 * image_area:
                continue
            if w < 60 or h < 60:
                continue
            if aspect_ratio < 0.2:
                continue

            roi_mask = binary_inv[y:y + h, x:x + w]
            white_ratio = np.mean(roi_mask / 255.0)
            if white_ratio < 0.08:
                continue

            regions.append((x, y, w, h))

        if not regions:
            regions = [(0, 0, img_bgr.shape[1], img_bgr.shape[0])]

        regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        merged_regions = []
        for rect in regions:
            x, y, w, h = rect
            merged = False
            for idx, (mx, my, mw, mh) in enumerate(merged_regions):
                if not (x > mx + mw or x + w < mx or y > my + mh or y + h < my):
                    nx = min(x, mx)
                    ny = min(y, my)
                    nw = max(x + w, mx + mw) - nx
                    nh = max(y + h, my + mh) - ny
                    merged_regions[idx] = (nx, ny, nw, nh)
                    merged = True
                    break
            if not merged:
                merged_regions.append(rect)

        regions = merged_regions[:5]

        overlay = img_bgr.copy()
        texts = []

        def run_ocr(image_gray):
            configs = [
                r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
                r'--oem 3 --psm 4 -c preserve_interword_spaces=1',
                r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
            ]
            candidates = [
                pytesseract.image_to_string(image_gray, config=cfg, lang='eng')
                for cfg in configs
            ]
            return max(candidates, key=lambda txt: sum(ch.isalnum() for ch in txt)).strip()

        for idx, (x, y, w, h) in enumerate(regions, 1):
            margin_w = int(w * 0.05)
            margin_h = int(h * 0.05)
            x0 = max(x - margin_w, 0)
            y0 = max(y - margin_h, 0)
            x1 = min(x + w + margin_w, binary_for_ocr.shape[1])
            y1 = min(y + h + margin_h, binary_for_ocr.shape[0])

            roi_binary = binary_for_ocr[y0:y1, x0:x1]
            roi_inverted = cv2.bitwise_not(roi_binary)

            text_candidate = run_ocr(roi_inverted)
            cleaned = re.sub(r'[^\nA-Za-z0-9%.,:;()\- ]+', ' ', text_candidate)
            cleaned_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            cleaned_text = '\n'.join(cleaned_lines)

            if cleaned_text:
                texts.append(cleaned_text)

            region_display = cv2.cvtColor(roi_inverted, cv2.COLOR_GRAY2RGB)
            debug_data['regions'].append({
                'index': idx,
                'bbox': (x0, y0, x1 - x0, y1 - y0),
                'image': region_display,
                'text': cleaned_text
            })

            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 180, 0), 2)

        debug_data['overlay'] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        combined_text = '\n'.join(dict.fromkeys(texts)).strip()

        if not combined_text:
            fallback = run_ocr(cv2.bitwise_not(binary_for_ocr))
            fallback_clean = re.sub(r'[^\nA-Za-z0-9%.,:;()\- ]+', ' ', fallback)
            fallback_lines = [line.strip() for line in fallback_clean.splitlines() if line.strip()]
            combined_text = '\n'.join(fallback_lines)

        cleaned_text = clean_ocr_text(combined_text)
        if not cleaned_text:
            cleaned_text = combined_text

        debug_data['raw_text'] = combined_text
        debug_data['clean_text'] = cleaned_text

        if return_debug:
            return cleaned_text, debug_data

        return cleaned_text

    except Exception as e:
        if return_debug:
            return "", {'error': str(e)}
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


def process_input(input_data, input_type, model, vectorizer, stemmer, stop_words, return_debug=False):
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
        'confidence': 0.0,
        'ocr_debug': None
    }
    
    # [1] & [2] Extract teks dari input
    if input_type == 'image':
        # Extract teks dari gambar menggunakan OCR
        ocr_output = extract_text_from_image(input_data, return_debug=return_debug)
        if return_debug:
            ocr_text, debug_data = ocr_output
            result['ocr_debug'] = debug_data
        else:
            ocr_text = ocr_output
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
                    stemmer,
                    stop_words,
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
