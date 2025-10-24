"""
Script untuk setup awal dan download dependencies NLTK
"""

import nltk
import sys

def download_nltk_data():
    """Download NLTK stopwords"""
    print("="*80)
    print("DOWNLOADING NLTK DATA")
    print("="*80)
    
    try:
        print("\n[1/1] Downloading stopwords...")
        nltk.download('stopwords', quiet=False)
        print("✓ Stopwords berhasil didownload")
        
        # Verify download
        from nltk.corpus import stopwords
        indo_stopwords = stopwords.words('indonesian')
        print(f"✓ Stopwords bahasa Indonesia: {len(indo_stopwords)} kata")
        
        print("\n" + "="*80)
        print("NLTK DATA SETUP COMPLETE!")
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nSilakan download manual dengan:")
        print("  python -c \"import nltk; nltk.download('stopwords')\"")
        return False


def check_tesseract():
    """Check apakah Tesseract OCR sudah terinstall"""
    print("\n" + "="*80)
    print("CHECKING TESSERACT OCR")
    print("="*80)
    
    try:
        import pytesseract
        
        # Try to get version
        version = pytesseract.get_tesseract_version()
        print(f"\n✓ Tesseract OCR terinstall: v{version}")
        return True
    
    except Exception as e:
        print("\n⚠️ Tesseract OCR tidak ditemukan!")
        print("\nSilakan install Tesseract OCR:")
        print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Linux: sudo apt-get install tesseract-ocr")
        print("  macOS: brew install tesseract")
        return False


def check_model_files():
    """Check apakah model sudah dilatih"""
    import os
    
    print("\n" + "="*80)
    print("CHECKING MODEL FILES")
    print("="*80)
    
    model_file = 'models/best_model.pkl'
    
    if os.path.exists(model_file):
        print(f"\n✓ Model ditemukan: {model_file}")
        
        # Load dan tampilkan info
        try:
            import pickle
            with open(model_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"  Model: {data['model_name']}")
            print(f"  Accuracy: {data['accuracy']*100:.2f}%")
            return True
        except:
            print("  ⚠️ Model corrupt atau format tidak valid")
            return False
    else:
        print(f"\n⚠️ Model belum dilatih!")
        print("\nJalankan pipeline training:")
        print("  1. python preprocess_data.py")
        print("  2. python feature_extraction.py")
        print("  3. python train_models.py")
        print("  4. python evaluate_and_predict.py")
        return False


def main():
    """Main setup function"""
    print("\n" + "="*80)
    print("SETUP SISTEM KLASIFIKASI HALAL-HARAM")
    print("="*80)
    
    results = []
    
    # 1. Download NLTK data
    results.append(("NLTK Data", download_nltk_data()))
    
    # 2. Check Tesseract
    results.append(("Tesseract OCR", check_tesseract()))
    
    # 3. Check model
    results.append(("Model Files", check_model_files()))
    
    # Summary
    print("\n" + "="*80)
    print("SETUP SUMMARY")
    print("="*80)
    
    for name, status in results:
        status_str = "✓ OK" if status else "✗ MISSING"
        print(f"  {name:<20}: {status_str}")
    
    print("\n" + "="*80)
    
    all_ok = all(status for _, status in results)
    
    if all_ok:
        print("✓ SETUP LENGKAP! Sistem siap digunakan.")
        print("\nJalankan aplikasi:")
        print("  streamlit run app.py")
    else:
        print("⚠️ Setup belum lengkap. Lihat instruksi di atas.")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
