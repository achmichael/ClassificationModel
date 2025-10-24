"""
Script standalone untuk klasifikasi halal-haram dengan input teks atau gambar
Dapat digunakan tanpa Streamlit untuk testing atau integrasi ke sistem lain
"""

import pandas as pd
import numpy as np
import pickle
import re
import cv2
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import argparse
import os

# Konfigurasi Tesseract (uncomment dan sesuaikan jika diperlukan)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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


class HalalClassifier:
    """
    Class untuk klasifikasi halal-haram dengan support input teks dan gambar
    """
    
    def __init__(self, model_path='models/best_model.pkl'):
        """
        Inisialisasi classifier
        
        Parameters:
        -----------
        model_path : str
            Path ke file model yang sudah dilatih
        """
        # Load model dan vectorizer
        self.load_model(model_path)
        
        # Inisialisasi NLP tools
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.stop_words = set(stopwords.words('indonesian'))
        
        print("✓ HalalClassifier siap digunakan")
        print(f"  Model: {self.model_name}")
        print(f"  Akurasi: {self.accuracy*100:.2f}%\n")
    
    
    def load_model(self, model_path):
        """Load model dan vectorizer dari file"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.model_name = data['model_name']
            self.accuracy = data['accuracy']
            
        except FileNotFoundError:
            raise FileNotFoundError(
                "Model tidak ditemukan! Jalankan train_models.py dan evaluate_and_predict.py terlebih dahulu."
            )
    
    
    def extract_text_from_image(self, image_path):
        """
        Ekstraksi teks dari gambar menggunakan Tesseract OCR
        
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
            
            # Konversi ke grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # OCR
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(denoised, config=custom_config)
            
            return text.strip()
        
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
        if pd.isna(text) or text == "":
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Normalisasi
        for foreign_word, indonesian_word in normalization_dict.items():
            text = text.replace(foreign_word, indonesian_word)
        
        # Hapus tanda baca dan angka
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenisasi dan hapus stopwords
        words = [word for word in text.split() if word not in self.stop_words]
        
        # Stemming
        words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    
    
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
            if prediction.lower() == 'halal':
                confidence = proba[0] if self.model.classes_[0] == 'halal' else proba[1]
            else:
                confidence = proba[1] if self.model.classes_[1] == 'haram' else proba[0]
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
    parser.add_argument('--model', type=str, default='models/best_model.pkl', 
                       help='Path ke file model (default: models/best_model.pkl)')
    
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
