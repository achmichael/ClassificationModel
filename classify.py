"""
Script standalone untuk klasifikasi halal-haram dengan input teks atau gambar
Dapat digunakan tanpa Streamlit untuk testing atau integrasi ke sistem lain
"""

import numpy as np
import re
import cv2
import pytesseract
from PIL import Image
import argparse
import os
from pathlib import Path

import joblib

from text_preprocessing import preprocess_text
from ocr_preprocessing import preprocess_for_ocr

# Konfigurasi Tesseract (uncomment dan sesuaikan jika diperlukan)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class HalalClassifier:
    """
    Class untuk klasifikasi halal-haram dengan support input teks dan gambar
    """
    
    def __init__(self, model_path='models/best_model.joblib'):
        """
        Inisialisasi classifier
        
        Parameters:
        -----------
        model_path : str
            Path ke file model yang sudah dilatih
        """
        # Load model dan vectorizer
        self.load_model(model_path)
        
        print("✓ HalalClassifier siap digunakan")
        print(f"  Model: {self.model_name}")
        print(f"  Akurasi: {self.accuracy*100:.2f}%\n")
    
    
    def load_model(self, model_path):
        """Load model dan vectorizer dari file"""
        model_file = Path(model_path)
        try:
            data = joblib.load(model_file)

            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.model_name = data.get('model_name', 'Unknown Model')
            self.accuracy = data.get('accuracy', 0.0)

        except FileNotFoundError:
            raise FileNotFoundError(
                "Model tidak ditemukan! Jalankan train_models.py dan evaluate_and_predict.py terlebih dahulu."
            )
    
    
    def extract_text_from_image(self, image_path):
        """
        Ekstraksi teks dari gambar menggunakan Tesseract OCR dengan preprocessing robust.
        Menggunakan pipeline preprocessing baru yang dapat menangani berbagai kondisi cahaya.
        
        Parameters:
        -----------
        image_path : str
            Path ke file gambar
        
        Returns:
        --------
        str : Teks hasil OCR
        """
        try:
            # Baca gambar dengan OpenCV
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
            
            print(f"\n{'='*60}")
            print(f"MEMPROSES GAMBAR: {os.path.basename(image_path)}")
            print(f"{'='*60}")
            
            # PREPROCESSING MENGGUNAKAN PIPELINE BARU
            # Pipeline ini secara otomatis:
            # 1. Resize gambar jika terlalu kecil
            # 2. Konversi ke grayscale
            # 3. Deteksi tipe background (terang/gelap)
            # 4. Penghilangan noise dengan bilateral filter
            # 5. Normalisasi kontras dengan CLAHE
            # 6. Gaussian blur untuk smoothing
            # 7. Binarisasi dengan adaptive/otsu threshold (otomatis pilih terbaik)
            # 8. Morphological operations untuk cleanup
            
            preprocess_result = preprocess_for_ocr(
                image=image,
                denoise_method='bilateral',      # Bilateral filter mempertahankan edges
                threshold_method='both',          # Coba adaptive dan otsu, pilih optimal
                morphology='open_close',          # Hapus noise kecil dan isi gaps
                auto_resize=True,
                min_width=1200                    # Resize ke minimum 1200px untuk OCR lebih akurat
            )
            
            # Ambil binary image yang sudah di-preprocess
            # Binary image ini sudah optimal: teks = putih (255), background = hitam (0)
            preprocessed_image = preprocess_result['binary']
            
            print(f"\n{'='*60}")
            print("MENJALANKAN OCR DENGAN MULTIPLE KONFIGURASI")
            print(f"{'='*60}")
            
            # EKSTRAKSI TEKS DENGAN MULTIPLE PSM MODES
            # PSM (Page Segmentation Mode) menentukan bagaimana Tesseract membagi halaman
            # Kita coba beberapa mode dan pilih yang menghasilkan teks terbanyak
            
            # Mode 1: PSM 6 - Uniform block of text (paling umum untuk label produk)
            print("\n1. Mencoba PSM 6 (Uniform block of text)...")
            custom_config_1 = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            text_1 = pytesseract.image_to_string(preprocessed_image, config=custom_config_1, lang='eng')
            print(f"   ✓ Hasil: {len(text_1)} karakter, {len(text_1.split())} kata")
            
            # Mode 2: PSM 11 - Sparse text (untuk teks yang jarang/terpisah)
            print("\n2. Mencoba PSM 11 (Sparse text)...")
            custom_config_2 = r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
            text_2 = pytesseract.image_to_string(preprocessed_image, config=custom_config_2, lang='eng')
            print(f"   ✓ Hasil: {len(text_2)} karakter, {len(text_2.split())} kata")
            
            # Mode 3: PSM 3 - Fully automatic page segmentation
            print("\n3. Mencoba PSM 3 (Fully automatic)...")
            custom_config_3 = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
            text_3 = pytesseract.image_to_string(preprocessed_image, config=custom_config_3, lang='eng')
            print(f"   ✓ Hasil: {len(text_3)} karakter, {len(text_3.split())} kata")
            
            # Mode 4: PSM 4 - Single column of text
            print("\n4. Mencoba PSM 4 (Single column)...")
            custom_config_4 = r'--oem 3 --psm 4 -c preserve_interword_spaces=1'
            text_4 = pytesseract.image_to_string(preprocessed_image, config=custom_config_4, lang='eng')
            print(f"   ✓ Hasil: {len(text_4)} karakter, {len(text_4.split())} kata")
            
            # Pilih hasil terbaik berdasarkan jumlah kata
            # Lebih banyak kata = kemungkinan lebih banyak informasi terdeteksi
            results = [text_1, text_2, text_3, text_4]
            best_text = max(results, key=lambda x: len(x.split()))
            best_index = results.index(best_text) + 1
            
            print(f"\n{'='*60}")
            print(f"HASIL TERBAIK: Mode {best_index} dengan {len(best_text.split())} kata")
            print(f"{'='*60}")
            
            # POST-PROCESSING: Clean up hasil OCR
            # 1. Trim whitespace di awal dan akhir
            cleaned_text = best_text.strip()
            
            # 2. Hapus karakter non-ASCII (karakter aneh hasil OCR error)
            cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)
            
            # 3. Normalize whitespace (hapus spasi ganda, tab, newline berlebih)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            
            # 4. Hapus karakter kontrol yang tidak diinginkan
            cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_text)
            
            print(f"\nTeks final (setelah cleaning): {len(cleaned_text)} karakter")
            print(f"Preview: {cleaned_text[:200]}..." if len(cleaned_text) > 200 else f"Full text: {cleaned_text}")
            
            return cleaned_text
        
        except Exception as e:
            print(f"\n❌ Error saat OCR: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    
    def preprocess_text(self, text):
        """
        Preprocessing teks
        
        Parameters:
        -----------
        text : str
            Teks yang akan diproses
        
        Returns:
        --------
        str : Teks yang sudah diproses
        """
        return preprocess_text(text)
    
    
    def predict(self, text):
        """
        Prediksi status halal/haram dari teks
        
        Parameters:
        -----------
        text : str
            Teks komposisi produk
        
        Returns:
        --------
        dict : Dictionary berisi hasil prediksi
        """
        # Preprocessing
        clean_text = self.preprocess_text(text)
        
        if clean_text == "":
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'preprocessed_text': ''
            }
        
        # Transform ke TF-IDF
        text_vector = self.vectorizer.transform([clean_text])
        
        # Prediksi
        prediction = self.model.predict(text_vector)[0]
        
        # Confidence score
        try:
            proba = self.model.predict_proba(text_vector)[0]
            class_index = list(self.model.classes_).index(prediction)
            confidence = proba[class_index]
        except AttributeError:
            try:
                decision = self.model.decision_function(text_vector)[0]
                confidence = 1 / (1 + np.exp(-decision))
                if prediction.lower() == 'halal':
                    confidence = 1 - confidence
            except:
                confidence = None
        
        return {
            'prediction': prediction.capitalize(),
            'confidence': confidence,
            'preprocessed_text': clean_text
        }
    
    
    def classify_from_text(self, text):
        """
        Klasifikasi dari input teks langsung
        
        Parameters:
        -----------
        text : str
            Teks komposisi produk
        
        Returns:
        --------
        dict : Hasil klasifikasi
        """
        print("="*80)
        print("KLASIFIKASI DARI TEKS")
        print("="*80)
        print(f"\nTeks Input:\n{text}\n")
        
        result = self.predict(text)
        
        print(f"Teks Setelah Preprocessing:\n{result['preprocessed_text']}\n")
        print("="*80)
        print("HASIL PREDIKSI")
        print("="*80)
        print(f"Status: {result['prediction']}")
        
        if result['confidence'] is not None:
            print(f"Confidence: {result['confidence']*100:.2f}%")
        
        print("="*80)
        
        return result
    
    
    def classify_from_image(self, image_path):
        """
        Klasifikasi dari gambar label produk
        
        Parameters:
        -----------
        image_path : str
            Path ke file gambar
        
        Returns:
        --------
        dict : Hasil klasifikasi termasuk teks OCR
        """
        print("="*80)
        print("KLASIFIKASI DARI GAMBAR")
        print("="*80)
        print(f"\nFile Gambar: {image_path}\n")
        
        # Extract teks dari gambar
        print("Melakukan OCR...")
        ocr_text = self.extract_text_from_image(image_path)
        
        if not ocr_text:
            print("⚠️ Tidak ada teks yang terdeteksi dari gambar")
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'ocr_text': '',
                'preprocessed_text': ''
            }
        
        print(f"\nTeks Hasil OCR:\n{ocr_text}\n")
        
        # Prediksi
        result = self.predict(ocr_text)
        result['ocr_text'] = ocr_text
        
        print(f"Teks Setelah Preprocessing:\n{result['preprocessed_text']}\n")
        print("="*80)
        print("HASIL PREDIKSI")
        print("="*80)
        print(f"Status: {result['prediction']}")
        
        if result['confidence'] is not None:
            print(f"Confidence: {result['confidence']*100:.2f}%")
        
        print("="*80)
        
        return result


def main():
    """
    Fungsi utama untuk command-line interface
    """
    parser = argparse.ArgumentParser(description='Klasifikasi Halal-Haram dari Teks atau Gambar')
    parser.add_argument('--text', type=str, help='Teks komposisi produk')
    parser.add_argument('--image', type=str, help='Path ke gambar label produk')
    parser.add_argument('--model', type=str, default='models/best_model.joblib', 
                       help='Path ke file model (default: models/best_model.joblib)')
    
    args = parser.parse_args()
    
    # Inisialisasi classifier
    try:
        classifier = HalalClassifier(model_path=args.model)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Proses input
    if args.text:
        classifier.classify_from_text(args.text)
    
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: File gambar tidak ditemukan: {args.image}")
            return
        classifier.classify_from_image(args.image)
    
    else:
        # Mode interaktif
        print("\n" + "="*80)
        print("SISTEM KLASIFIKASI HALAL-HARAM")
        print("="*80)
        print("\nPilih mode input:")
        print("1. Input Teks")
        print("2. Input Gambar")
        
        choice = input("\nPilihan (1/2): ").strip()
        
        if choice == '1':
            text = input("\nMasukkan komposisi produk:\n")
            classifier.classify_from_text(text)
        
        elif choice == '2':
            image_path = input("\nMasukkan path gambar: ").strip()
            if os.path.exists(image_path):
                classifier.classify_from_image(image_path)
            else:
                print(f"Error: File tidak ditemukan: {image_path}")
        
        else:
            print("Pilihan tidak valid!")


if __name__ == "__main__":
    main()
