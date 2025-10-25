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
        Ekstraksi teks dari gambar menggunakan Tesseract OCR dengan preprocessing advanced
        
        Parameters:
        -----------
        image_path : str
            Path ke file gambar
        
        Returns:
        --------
        str : Teks hasil OCR
        """
        try:
            # Baca gambar
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
            
            # PREPROCESSING ADVANCED UNTUK MENINGKATKAN AKURASI OCR
            
            # 1. Resize gambar jika terlalu kecil (minimum 1000px width)
            height, width = image.shape[:2]
            if width < 1000:
                scale = 1000 / width
                new_width = 1000
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 2. Konversi ke grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 3. Denoise SEBELUM thresholding
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # 4. Increase contrast dengan CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(denoised)
            
            # 5. Sharpen image
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1,  9, -1],
                                          [-1, -1, -1]])
            sharpened = cv2.filter2D(contrast_enhanced, -1, kernel_sharpening)
            
            # 6. Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                sharpened, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                blockSize=11, 
                C=2
            )
            
            # 7. Morphological operations
            kernel = np.ones((1, 1), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # 8. Invert jika perlu
            if np.mean(morph) > 127:
                morph = cv2.bitwise_not(morph)
            
            # EKSTRAKSI TEKS DENGAN MULTIPLE MODES
            
            # Mode 1: PSM 6 (Uniform block of text)
            custom_config_1 = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            text_1 = pytesseract.image_to_string(morph, config=custom_config_1, lang='eng')
            
            # Mode 2: PSM 11 (Sparse text)
            custom_config_2 = r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
            text_2 = pytesseract.image_to_string(morph, config=custom_config_2, lang='eng')
            
            # Mode 3: PSM 3 (Fully automatic)
            custom_config_3 = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
            text_3 = pytesseract.image_to_string(morph, config=custom_config_3, lang='eng')
            
            # Pilih hasil terbaik
            results = [text_1, text_2, text_3]
            best_text = max(results, key=lambda x: len(x.split()))
            
            # Clean up hasil OCR
            cleaned_text = best_text.strip()
            cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)  # Hapus non-ASCII
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize spaces
            
            return cleaned_text
        
        except Exception as e:
            print(f"Error saat OCR: {str(e)}")
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
