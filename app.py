import streamlit as st
import numpy as np
import joblib
import re
import cv2
import pytesseract
from PIL import Image
from pathlib import Path

from text_preprocessing import preprocess_text
from ocr_preprocessing import preprocess_for_ocr

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
    Ekstraksi teks dari gambar dengan preprocessing robust untuk berbagai kondisi cahaya.
    Menggunakan pipeline preprocessing baru yang dapat menangani background terang maupun gelap.
    """
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
            'regions': [],
            'is_dark_background': False
        }

        print("=" * 60)
        print("MEMULAI PREPROCESSING OCR")
        print("=" * 60)
        
        # Gunakan pipeline preprocessing baru yang lebih robust
        preprocess_result = preprocess_for_ocr(
            image=img_bgr,
            denoise_method='bilateral',      # Bilateral filter untuk preserve edges
            threshold_method='both',          # Coba adaptive dan otsu, pilih yang terbaik
            morphology='open_close',          # Cleanup noise dan isi gaps
            auto_resize=True,
            min_width=1200
        )
        
        # Ambil hasil preprocessing
        binary_for_ocr = preprocess_result['binary']
        enhanced = preprocess_result['grayscale']
        is_light_background = preprocess_result['is_light_background']
        mean_intensity = preprocess_result['mean_intensity']
        
        # Update debug data
        is_dark_background = not is_light_background
        debug_data['is_dark_background'] = is_dark_background
        
        # Jika gambar di-resize, update img_bgr
        if preprocess_result['processed_size'] != preprocess_result['original_size']:
            new_width, new_height = preprocess_result['processed_size']
            img_bgr = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            debug_data['original'] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Binary untuk deteksi region - gunakan hasil preprocessing yang sama
        binary_inv = binary_for_ocr.copy()

        # Koneksi morfologi untuk menggabungkan karakter
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        connected = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, connect_kernel, iterations=2)
        
        # Dilasi untuk memperbesar region
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        connected = cv2.dilate(connected, dilate_kernel, iterations=1)

        debug_data['preprocessed'] = cv2.cvtColor(binary_for_ocr, cv2.COLOR_GRAY2RGB)
        debug_data['connected'] = cv2.cvtColor(connected, cv2.COLOR_GRAY2RGB)

        # Deteksi contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = img_bgr.shape[0] * img_bgr.shape[1]
        regions = []

        print(f"Found {len(contours)} contours")

        # Filter regions dengan threshold yang lebih rendah
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / float(h) if h else 0.0
            
            # Filter: area minimum sangat kecil
            if area < 0.00005 * image_area:
                continue
            
            # Filter: ukuran minimum sangat kecil
            if w < 20 or h < 15:
                continue
            
            # Filter: aspect ratio sangat lebar
            if aspect_ratio > 50 or aspect_ratio < 0.05:
                continue

            # Check white pixel ratio di region
            roi_mask = binary_inv[y:y + h, x:x + w]
            if roi_mask.size == 0:
                continue
                
            white_ratio = np.mean(roi_mask / 255.0)
            
            # Filter: white ratio minimum
            if white_ratio < 0.02:
                continue

            regions.append((x, y, w, h))
            print(f"  ✓ Region: ({x},{y}) {w}x{h}, area={area}, aspect={aspect_ratio:.2f}, white={white_ratio:.3f}")

        # Jika tidak ada region terdeteksi, gunakan seluruh gambar
        if not regions:
            print("⚠️ Tidak ada region terdeteksi, menggunakan seluruh gambar")
            regions = [(0, 0, img_bgr.shape[1], img_bgr.shape[0])]

        # Sort regions by area (largest first)
        regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        # Merge overlapping regions
        merged_regions = []
        for rect in regions:
            x, y, w, h = rect
            merged = False
            for idx, (mx, my, mw, mh) in enumerate(merged_regions):
                # Check if rectangles overlap
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

        # Ambil maksimal 8 region terbesar
        regions = merged_regions[:8]
        print(f"After merging: {len(regions)} regions")

        overlay = img_bgr.copy()
        texts = []

        def run_ocr(image_gray, invert=False):
            """
            Run OCR dengan multiple konfigurasi untuk mendapatkan hasil terbaik.
            Mencoba berbagai page segmentation mode (PSM) dan OEM.
            """
            if invert:
                image_gray = cv2.bitwise_not(image_gray)
            
            # Konfigurasi OCR dengan berbagai PSM mode:
            # PSM 6 = Uniform block of text (paling umum untuk label produk)
            # PSM 3 = Fully automatic page segmentation
            # PSM 11 = Sparse text (untuk teks yang jarang/terpisah)
            # PSM 4 = Single column of text
            configs = [
                r'--oem 3 --psm 6',   # LSTM + uniform block (terbaik untuk paragraf)
                r'--oem 3 --psm 3',   # LSTM + auto (fleksibel)
                r'--oem 3 --psm 11',  # LSTM + sparse text
                r'--oem 3 --psm 4',   # LSTM + single column
                r'--oem 1 --psm 6',   # Tesseract legacy + uniform block (backup)
            ]
            
            best_text = ""
            max_alnum = 0
            
            for cfg in configs:
                try:
                    text = pytesseract.image_to_string(image_gray, config=cfg, lang='eng')
                    text = text.strip()
                    alnum_count = sum(ch.isalnum() for ch in text)
                    if alnum_count > max_alnum:
                        max_alnum = alnum_count
                        best_text = text
                except Exception as e:
                    continue
            
            return best_text

        # Process setiap region
        for idx, (x, y, w, h) in enumerate(regions, 1):
            # Tambah margin untuk memastikan tidak ada teks yang terpotong
            margin_w = int(w * 0.03)
            margin_h = int(h * 0.03)
            x0 = max(x - margin_w, 0)
            y0 = max(y - margin_h, 0)
            x1 = min(x + w + margin_w, binary_for_ocr.shape[1])
            y1 = min(y + h + margin_h, binary_for_ocr.shape[0])

            # Extract ROI dari binary image yang sudah di-preprocess
            roi_binary = binary_for_ocr[y0:y1, x0:x1]
            
            if roi_binary.size == 0:
                continue
            
            # Coba OCR langsung dan inverted (untuk handling edge cases)
            text_normal = run_ocr(roi_binary, invert=False)
            text_inverted = run_ocr(roi_binary, invert=True)
            
            # Pilih yang lebih banyak alphanumeric characters
            alnum_normal = sum(ch.isalnum() for ch in text_normal)
            alnum_inverted = sum(ch.isalnum() for ch in text_inverted)
            
            text_candidate = text_normal if alnum_normal >= alnum_inverted else text_inverted
            used_invert = alnum_inverted > alnum_normal

            # Clean text: hapus karakter yang tidak valid
            cleaned = re.sub(r'[^\nA-Za-z0-9%.,:;()\- ]+', ' ', text_candidate)
            cleaned_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            cleaned_text = '\n'.join(cleaned_lines)

            if cleaned_text and len(cleaned_text.strip()) >= 3:
                texts.append(cleaned_text)
                print(f"✓ Region {idx}: {len(cleaned_text)} chars (inverted={used_invert})")
            else:
                print(f"✗ Region {idx}: Empty or too short")

            # Debug info
            region_display = cv2.cvtColor(roi_binary, cv2.COLOR_GRAY2RGB)
            debug_data['regions'].append({
                'index': idx,
                'bbox': (x0, y0, x1 - x0, y1 - y0),
                'image': region_display,
                'text': cleaned_text,
                'inverted': used_invert
            })

            # Draw rectangle
            color = (0, 255, 0) if cleaned_text else (255, 0, 0)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
            cv2.putText(overlay, f"R{idx}", (x0, y0-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        debug_data['overlay'] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Combine texts
        combined_text = '\n'.join([t for t in texts if t.strip()])

        # Fallback: jika hasil terlalu sedikit, coba full image OCR
        if not combined_text or len(combined_text.strip()) < 20:
            print("⚠️ Hasil region terlalu pendek, mencoba fallback full OCR")
            
            # Coba dengan berbagai preprocessing
            fallback_candidates = []
            
            # 1. Binary original
            text1 = run_ocr(binary_for_ocr, invert=False)
            fallback_candidates.append(text1)
            
            # 2. Binary inverted
            text2 = run_ocr(binary_for_ocr, invert=True)
            fallback_candidates.append(text2)
            
            # 3. Gray original
            text3 = run_ocr(enhanced, invert=False)
            fallback_candidates.append(text3)
            
            # 4. Gray inverted
            text4 = run_ocr(enhanced, invert=True)
            fallback_candidates.append(text4)
            
            # Pilih fallback terbaik
            best_fallback = max(fallback_candidates, key=lambda t: sum(ch.isalnum() for ch in t))
            
            if best_fallback:
                fallback_clean = re.sub(r'[^\nA-Za-z0-9%.,:;()\- ]+', ' ', best_fallback)
                fallback_lines = [line.strip() for line in fallback_clean.splitlines() if line.strip()]
                combined_text = '\n'.join(fallback_lines)
                print(f"✓ Fallback OCR: {len(combined_text)} chars")

        # Clean OCR text dengan fungsi pembersih
        cleaned_text = clean_ocr_text(combined_text)
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            cleaned_text = combined_text

        debug_data['raw_text'] = combined_text
        debug_data['clean_text'] = cleaned_text
        debug_data['background_type'] = 'Dark' if is_dark_background else 'Light'

        print(f"✓ Final text length: {len(cleaned_text)} chars")
        if cleaned_text:
            print(f"Preview: {cleaned_text[:100]}...")

        if return_debug:
            return cleaned_text, debug_data

        return cleaned_text

    except Exception as e:
        import traceback
        error_msg = f"Error saat ekstraksi teks dari gambar: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        if return_debug:
            return "", {'error': error_msg}
        st.error(error_msg)
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
