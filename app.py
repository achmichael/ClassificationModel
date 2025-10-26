import streamlit as st
import numpy as np
import joblib
import re
import cv2
import pytesseract
from PIL import Image
from pathlib import Path

from text_preprocessing import preprocess_text

OCR_SIMPLE_CORRECTIONS = {
    "URBLEACHED": "UNBLEACHED",
    "HURBLEACHED": "UNBLEACHED",
    "BLEACHEDD": "BLEACHED",
    "FLOLJR": "FLOUR",
    "FLOIIR": "FLOUR",
    "FI.OUR": "FLOUR",
    "FLOURR": "FLOUR",
    "FLOUR-": "FLOUR",
    "HONONITRATE": "MONONITRATE",
    "MONONlTRATE": "MONONITRATE",
    "MONONLTRATE": "MONONITRATE",
    "THLAMINE": "THIAMINE",
    "THlAMINE": "THIAMINE",
    "THlAMlNE": "THIAMINE",
    "THAMINE": "THIAMINE",
    "VITAMLN": "VITAMIN",
    "VITAMlN": "VITAMIN",
    "VITAM1N": "VITAMIN",
    "RIBOFLAVLN": "RIBOFLAVIN",
    "RIBOFLAVlN": "RIBOFLAVIN",
    "RIBOFLAVINN": "RIBOFLAVIN",
    "FOLLC": "FOLIC",
    "FOLIC": "FOLIC",
    "FOLLC ACID": "FOLIC ACID",
    "FOLIC ACID": "FOLIC ACID",
    "L RON": "IRON",
    "LRON": "IRON",
    "I RON": "IRON",
    "lRON": "IRON",
    "NlACIN": "NIACIN",
    "NIAClN": "NIACIN",
    "REDUCED lRON": "REDUCED IRON",
    "REDUCED LRON": "REDUCED IRON",
    "REDUCED L RON": "REDUCED IRON",
    "MlLK": "MILK",
    "SODlUM": "SODIUM"
}

OCR_REGEX_CORRECTIONS = [
    (re.compile(r'(?<=\b[A-Z]{2,})1(?=[A-Z]{2,}\b)'), 'I'),
]

SUBINGREDIENT_ANCHORS = (
    "ENRICHED FLOUR",
    "ENRICHED WHEAT FLOUR",
    "UNBLEACHED ENRICHED FLOUR",
    "BLEACHED ENRICHED FLOUR"
)

SUBINGREDIENT_TERMINATORS = (
    "FOLIC ACID",
    "FOLATE",
    "VITAMIN B9",
    "RIBOFLAVIN",
    "THIAMINE MONONITRATE",
    "REDUCED IRON",
    "NIACIN",
    "WHEAT FLOUR"
)

SECTION_BOUNDARIES = (
    " INGREDIENTS",
    " CONTAINS",
    " MAY CONTAIN",
    " ALLERGEN",
    " ALLERGENS",
    " DISTRIBUTED",
    " MANUFACTURED",
    " PRODUCED",
    " PACKAGED",
    " NET ",
    " BEST",
    " KEEP",
    " STORE",
    " DIRECTIONS",
    " PREPARED",
    " WARNING",
    " NUTRITION"
)


def _apply_regex_corrections(text: str) -> str:
    for pattern, replacement in OCR_REGEX_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


def _apply_simple_corrections(text: str) -> str:
    for wrong, right in OCR_SIMPLE_CORRECTIONS.items():
        text = re.sub(rf'\b{re.escape(wrong)}\b', right, text)
    return text


def _normalize_section_headers(text: str) -> str:
    text = re.sub(r'INGREDIENTS\s*(?![:])', 'INGREDIENTS: ', text)
    text = re.sub(r'MAY CONTAIN\s*(?![:])', 'MAY CONTAIN: ', text)
    text = re.sub(r'CONTAINS\s+(?!LESS|UP TO|\d|[:])', 'CONTAINS: ', text)
    return text


def _wrap_enriched_flour_sections(text: str) -> str:
    for anchor in SUBINGREDIENT_ANCHORS:
        search_pos = 0
        while True:
            idx = text.find(anchor, search_pos)
            if idx == -1:
                break
            after_anchor = idx + len(anchor)
            if '(' in text[after_anchor:stop_idx]:
                continue
            if text[after_anchor:after_anchor + 2] == ' (':
                search_pos = after_anchor
                continue

            stop_idx = -1
            for terminator in SUBINGREDIENT_TERMINATORS:
                term_idx = text.find(terminator, after_anchor)
                if term_idx != -1:
                    stop_idx = max(stop_idx, term_idx + len(terminator))

            if stop_idx == -1:
                boundary_idx = len(text)
                for boundary in SECTION_BOUNDARIES:
                    boundary_pos = text.find(boundary, after_anchor)
                    if boundary_pos != -1 and boundary_pos < boundary_idx:
                        boundary_idx = boundary_pos
                stop_idx = boundary_idx

            tail_match = re.match(r'\s*\([^)]*\)', text[stop_idx:])
            if tail_match:
                stop_idx += tail_match.end()

            include_comma = False
            if stop_idx < len(text) and text[stop_idx] == ',':
                include_comma = True
                stop_idx += 1

            segment = text[after_anchor:stop_idx].strip()
            if segment.startswith('('):
                search_pos = after_anchor
                continue

            if segment:
                segment = segment.strip(', ')
                insertion = f" ({segment})"
                next_slice = text[stop_idx:]
                if include_comma:
                    insertion += ','
                else:
                    next_char = next_slice.lstrip()[:1]
                    if next_char and next_char not in {'.', ';', ')', ','}:
                        insertion += ','
                text = text[:after_anchor] + insertion + text[stop_idx:]
                search_pos = after_anchor + len(insertion)
            else:
                search_pos = after_anchor

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
    text = _apply_simple_corrections(text)

    text = re.sub(r'\bAI\b', '(', text)
    text = re.sub(r'\bA1\b', '(', text)
    text = re.sub(r'\bI\)', ')', text)
    text = text.replace(' )', ')').replace('( ', '(')

    text = text.replace('..', '.')
    text = text.replace(',,', ',')

    text = _remove_consecutive_duplicates(text)
    text = _normalize_section_headers(text)
    text = _wrap_enriched_flour_sections(text)
    text = _tidy_punctuation_and_spacing(text)

    if text and not text.endswith('.'):
        text += '.'

    return text


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
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
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
            if area < 0.005 * image_area:
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
                r'--oem 3 --psm 3 -c preserve_interword_spaces=1',
                r'--oem 3 --psm 12 -c preserve_interword_spaces=1'
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
        
        regions.sort(key=lambda r: (r[1], r[0]))
        
        combined_text = '\n'.join([t for t in texts if t.strip()])

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
