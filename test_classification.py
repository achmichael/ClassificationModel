"""
Script untuk testing klasifikasi dengan berbagai contoh input
"""

from classify import HalalClassifier

def test_text_examples():
    """Test dengan contoh-contoh teks"""
    
    classifier = HalalClassifier()
    
    print("\n" + "="*80)
    print("TESTING DENGAN CONTOH TEKS")
    print("="*80)
    
    # Contoh produk HALAL
    halal_examples = [
        "sugar vegetable oil salt natural flavoring",
        "chicken stock water salt natural flavor",
        "organic wheat flour water sea salt yeast",
        "rice flour palm oil sugar coconut milk",
        "filtered water apple juice concentrate vitamin c"
    ]
    
    # Contoh produk HARAM
    haram_examples = [
        "pork water salt sugar sodium nitrite",
        "beef stock gelatin natural flavor",
        "bacon cured with water salt smoke flavoring",
        "wine vinegar alcohol natural flavor",
        "chicken mechanically separated turkey pork water"
    ]
    
    print("\n" + "-"*80)
    print("CONTOH PRODUK HALAL")
    print("-"*80)
    
    for i, text in enumerate(halal_examples, 1):
        print(f"\n[{i}] Input: {text}")
        result = classifier.predict(text)
        status = "✓" if result['prediction'].lower() == 'halal' else "✗"
        print(f"    {status} Prediksi: {result['prediction']}", end="")
        if result['confidence'] is not None:
            print(f" ({result['confidence']*100:.2f}%)")
        else:
            print()
    
    print("\n" + "-"*80)
    print("CONTOH PRODUK HARAM")
    print("-"*80)
    
    for i, text in enumerate(haram_examples, 1):
        print(f"\n[{i}] Input: {text}")
        result = classifier.predict(text)
        status = "✓" if result['prediction'].lower() == 'haram' else "✗"
        print(f"    {status} Prediksi: {result['prediction']}", end="")
        if result['confidence'] is not None:
            print(f" ({result['confidence']*100:.2f}%)")
        else:
            print()
    
    print("\n" + "="*80)


def test_image_example():
    """Test dengan contoh gambar (jika ada)"""
    import os
    
    classifier = HalalClassifier()
    
    print("\n" + "="*80)
    print("TESTING DENGAN GAMBAR")
    print("="*80)
    
    # Cari file gambar di folder saat ini
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        print(f"\nDitemukan {len(image_files)} file gambar:")
        for img in image_files:
            print(f"  - {img}")
        
        print("\nPilih file gambar untuk testing (kosongkan untuk skip):")
        choice = input("Nama file: ").strip()
        
        if choice and os.path.exists(choice):
            classifier.classify_from_image(choice)
        else:
            print("Skip testing gambar.")
    else:
        print("\n⚠️ Tidak ada file gambar untuk testing")
        print("Untuk testing gambar, letakkan file .jpg/.png di folder ini")
    
    print("\n" + "="*80)


def main():
    """Main testing function"""
    
    print("\n" + "="*80)
    print("AUTOMATED TESTING - SISTEM KLASIFIKASI HALAL-HARAM")
    print("="*80)
    
    try:
        # Test dengan teks
        test_text_examples()
        
        # Test dengan gambar (opsional)
        print("\n\nIngin testing dengan gambar? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            test_image_example()
        
        print("\n" + "="*80)
        print("TESTING SELESAI")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        print("\nPastikan model sudah dilatih dengan menjalankan:")
        print("  python train_models.py")
        print("  python evaluate_and_predict.py")


if __name__ == "__main__":
    main()
