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
from ocr_simple import preprocess_for_cli_match

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
            
            # SIMPLIFIED PREPROCESSING TO MATCH CLI BEHAVIOR
            # Minimal processing: grayscale → bilateral filter → upscale if needed → optional invert
            # NO thresholding, NO morphology, NO CLAHE
            # This produces results identical to: tesseract input.png stdout -l eng --psm 6
            
            preprocessed_image = preprocess_for_cli_match(image)
            
            print(f"\n{'='*60}")
            print("MENJALANKAN OCR DENGAN TESSERACT")
            print(f"{'='*60}")
            
            # EKSTRAKSI TEKS DENGAN TESSERACT OCR
            # Exact CLI configuration for consistency
            print("\nEkstraksi teks menggunakan Tesseract...")
            text = pytesseract.image_to_string(
                preprocessed_image,
                lang='eng',
                config='--psm 6'
            )
            
            print(f"✓ Hasil: {len(text)} karakter, {len(text.split())} kata")
            print(f"{'='*60}")
            
            # POST-PROCESSING: Light cleanup only
            cleaned_text = text.strip()
            
            print(f"\nTeks final: {len(cleaned_text)} karakter")
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
