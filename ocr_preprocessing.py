"""
OCR Preprocessing Module
=========================
Module ini menyediakan fungsi-fungsi preprocessing untuk meningkatkan akurasi OCR,
terutama untuk gambar dengan latar belakang terang (putih) dan teks gelap (hitam).

Author: Classification Model Team
Date: October 2025
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def detect_background_type(gray_image: np.ndarray) -> Tuple[bool, float]:
    """
    Mendeteksi apakah gambar memiliki background terang atau gelap.
    
    Parameters:
    -----------
    gray_image : np.ndarray
        Gambar grayscale (single channel)
    
    Returns:
    --------
    Tuple[bool, float]
        - is_light_background: True jika background terang, False jika gelap
        - mean_intensity: Nilai rata-rata intensitas (0-255)
    """
    # Hitung rata-rata intensitas
    mean_intensity = np.mean(gray_image)
    
    # Hitung median intensitas untuk lebih robust terhadap outliers
    median_intensity = np.median(gray_image)
    
    # Kombinasi mean dan median untuk deteksi lebih akurat
    average_intensity = (mean_intensity + median_intensity) / 2.0
    
    # Threshold: > 127 = background terang (putih), <= 127 = background gelap
    is_light_background = average_intensity > 127
    
    return is_light_background, average_intensity


def normalize_contrast(gray_image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Normalisasi kontras menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization).
    CLAHE sangat efektif untuk meningkatkan kontras lokal tanpa over-amplifying noise.
    
    Parameters:
    -----------
    gray_image : np.ndarray
        Gambar grayscale
    clip_limit : float
        Batas clipping untuk CLAHE (default: 2.0)
        Nilai lebih tinggi = kontras lebih tinggi, tapi bisa menambah noise
    
    Returns:
    --------
    np.ndarray
        Gambar dengan kontras yang sudah dinormalisasi
    """
    # CLAHE dengan tile size 8x8 untuk adaptasi lokal yang baik
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    return enhanced


def remove_noise(gray_image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Menghilangkan noise dari gambar sambil mempertahankan edges.
    
    Parameters:
    -----------
    gray_image : np.ndarray
        Gambar grayscale
    method : str
        Metode denoising: 'bilateral', 'gaussian', atau 'nlmeans'
        - bilateral: Bagus untuk preserve edges, cepat
        - gaussian: Simple blur, sangat cepat
        - nlmeans: Paling bagus tapi lambat
    
    Returns:
    --------
    np.ndarray
        Gambar yang sudah dihilangkan noise-nya
    """
    if method == 'bilateral':
        # Bilateral filter: Smooth noise tapi preserve edges
        # Parameters: diameter=11, sigmaColor=17, sigmaSpace=17
        denoised = cv2.bilateralFilter(gray_image, 11, 17, 17)
        
    elif method == 'gaussian':
        # Gaussian blur: Simple dan cepat
        denoised = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
    elif method == 'nlmeans':
        # Non-local means denoising: Paling bagus tapi lambat
        denoised = cv2.fastNlMeansDenoising(gray_image, None, h=10, 
                                            templateWindowSize=7, 
                                            searchWindowSize=21)
    else:
        # Default: bilateral
        denoised = cv2.bilateralFilter(gray_image, 11, 17, 17)
    
    return denoised


def apply_adaptive_threshold(gray_image: np.ndarray, 
                             is_light_background: bool,
                             block_size: int = 31,
                             C: int = 5) -> np.ndarray:
    """
    Menerapkan adaptive thresholding yang disesuaikan dengan tipe background.
    Adaptive threshold lebih baik daripada global threshold untuk gambar dengan
    pencahayaan tidak merata.
    
    Parameters:
    -----------
    gray_image : np.ndarray
        Gambar grayscale yang sudah di-preprocess
    is_light_background : bool
        True jika background terang, False jika gelap
    block_size : int
        Ukuran neighborhood untuk adaptive threshold (harus ganjil)
        Nilai lebih besar = lebih smooth, lebih kecil = lebih detail
    C : int
        Konstanta yang dikurangi dari weighted mean
        Nilai lebih tinggi = threshold lebih strict
    
    Returns:
    --------
    np.ndarray
        Binary image (0 atau 255)
    """
    # Pastikan block_size ganjil
    if block_size % 2 == 0:
        block_size += 1
    
    if is_light_background:
        # Background terang (putih), teks gelap (hitam)
        # Gunakan THRESH_BINARY untuk menghasilkan teks hitam (0) di background putih (255)
        # Kemudian invert agar teks jadi putih (255) untuk OCR
        binary = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )
        # Invert: teks hitam → putih, background putih → hitam
        binary = cv2.bitwise_not(binary)
        
    else:
        # Background gelap, teks terang
        # Gunakan THRESH_BINARY_INV untuk langsung mendapat teks putih
        binary = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            C
        )
    
    return binary


def apply_otsu_threshold(gray_image: np.ndarray,
                        is_light_background: bool) -> np.ndarray:
    """
    Menerapkan Otsu's thresholding untuk binarisasi otomatis.
    Otsu's method secara otomatis menentukan threshold optimal berdasarkan histogram.
    
    Parameters:
    -----------
    gray_image : np.ndarray
        Gambar grayscale yang sudah di-preprocess
    is_light_background : bool
        True jika background terang, False jika gelap
    
    Returns:
    --------
    np.ndarray
        Binary image (0 atau 255)
    """
    if is_light_background:
        # Background terang: gunakan THRESH_BINARY + OTSU, lalu invert
        _, binary = cv2.threshold(
            gray_image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Invert agar teks jadi putih
        binary = cv2.bitwise_not(binary)
        
    else:
        # Background gelap: gunakan THRESH_BINARY_INV + OTSU
        _, binary = cv2.threshold(
            gray_image,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    
    return binary


def apply_morphological_operations(binary_image: np.ndarray,
                                   operation: str = 'open_close',
                                   kernel_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """
    Menerapkan operasi morfologi untuk membersihkan noise kecil dan mengisi gaps.
    
    Parameters:
    -----------
    binary_image : np.ndarray
        Binary image (0 atau 255)
    operation : str
        Jenis operasi: 'open', 'close', 'open_close', atau 'none'
        - open: Menghilangkan noise kecil (erode lalu dilate)
        - close: Mengisi gaps kecil (dilate lalu erode)
        - open_close: Kombinasi keduanya (open dulu, lalu close)
    kernel_size : Tuple[int, int]
        Ukuran kernel (width, height)
        Kernel lebih besar = efek lebih kuat
    
    Returns:
    --------
    np.ndarray
        Binary image yang sudah di-clean
    """
    if operation == 'none':
        return binary_image
    
    # Buat kernel rectangular
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    if operation == 'open':
        # Opening: Remove noise (erode → dilate)
        result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        
    elif operation == 'close':
        # Closing: Fill gaps (dilate → erode)
        result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        
    elif operation == 'open_close':
        # Kombinasi: open dulu untuk hapus noise, lalu close untuk isi gaps
        temp = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    
    else:
        result = binary_image
    
    return result


def preprocess_for_ocr(image: np.ndarray,
                       denoise_method: str = 'bilateral',
                       threshold_method: str = 'adaptive',
                       morphology: str = 'open_close',
                       auto_resize: bool = True,
                       min_width: int = 1200) -> Dict[str, any]:
    """
    Pipeline preprocessing lengkap untuk OCR.
    Fungsi ini menggabungkan semua langkah preprocessing untuk hasil OCR optimal.
    
    Pipeline:
    1. Resize gambar jika terlalu kecil (opsional)
    2. Konversi ke grayscale
    3. Deteksi tipe background (terang/gelap)
    4. Penghilangan noise (bilateral/gaussian/nlmeans)
    5. Normalisasi kontras (CLAHE)
    6. Gaussian blur ringan untuk smoothing
    7. Binarisasi dengan adaptive/otsu threshold
    8. Morphological operations untuk cleanup
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (BGR atau RGB format dari cv2.imread atau PIL)
    denoise_method : str
        Metode denoising: 'bilateral', 'gaussian', atau 'nlmeans'
    threshold_method : str
        Metode thresholding: 'adaptive', 'otsu', atau 'both'
        - adaptive: Lebih bagus untuk pencahayaan tidak merata
        - otsu: Lebih cepat, bagus untuk pencahayaan merata
        - both: Coba keduanya dan pilih yang terbaik
    morphology : str
        Operasi morfologi: 'open', 'close', 'open_close', atau 'none'
    auto_resize : bool
        Otomatis resize jika gambar terlalu kecil
    min_width : int
        Lebar minimum untuk resize (default: 1200px)
    
    Returns:
    --------
    Dict dengan keys:
        - 'binary': Binary image siap untuk OCR (teks putih, background hitam)
        - 'grayscale': Grayscale image setelah preprocessing
        - 'is_light_background': Boolean, True jika background terang
        - 'mean_intensity': Float, rata-rata intensitas
        - 'threshold_method_used': String, metode threshold yang digunakan
        - 'original_size': Tuple (width, height) ukuran asli
        - 'processed_size': Tuple (width, height) ukuran setelah resize
    """
    result = {}
    
    # Simpan ukuran asli
    original_height, original_width = image.shape[:2]
    result['original_size'] = (original_width, original_height)
    
    # Langkah 1: Resize jika terlalu kecil (OCR lebih akurat pada gambar yang lebih besar)
    if auto_resize and original_width < min_width:
        scale = min_width / original_width
        new_width = min_width
        new_height = int(original_height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"✓ Resized from {original_width}x{original_height} to {new_width}x{new_height}")
    
    result['processed_size'] = (image.shape[1], image.shape[0])
    
    # Langkah 2: Konversi ke grayscale
    if len(image.shape) == 3:
        # Gambar berwarna, konversi ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Sudah grayscale
        gray = image.copy()
    
    print("✓ Converted to grayscale")
    
    # Langkah 3: Deteksi tipe background
    is_light_bg, mean_intensity = detect_background_type(gray)
    result['is_light_background'] = is_light_bg
    result['mean_intensity'] = mean_intensity
    
    bg_type = "TERANG (putih)" if is_light_bg else "GELAP (hitam)"
    print(f"✓ Background detected: {bg_type} (intensity: {mean_intensity:.1f})")
    
    # Langkah 4: Penghilangan noise
    denoised = remove_noise(gray, method=denoise_method)
    print(f"✓ Noise removed using {denoise_method} filter")
    
    # Langkah 5: Normalisasi kontras dengan CLAHE
    enhanced = normalize_contrast(denoised, clip_limit=2.0)
    print("✓ Contrast normalized with CLAHE")
    
    # Langkah 6: Gaussian blur ringan untuk smoothing final
    smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)
    print("✓ Final smoothing applied")
    
    result['grayscale'] = smoothed
    
    # Langkah 7: Binarisasi dengan threshold
    if threshold_method == 'adaptive':
        binary = apply_adaptive_threshold(smoothed, is_light_bg, block_size=31, C=5)
        result['threshold_method_used'] = 'adaptive'
        print("✓ Applied adaptive threshold")
        
    elif threshold_method == 'otsu':
        binary = apply_otsu_threshold(smoothed, is_light_bg)
        result['threshold_method_used'] = 'otsu'
        print("✓ Applied Otsu threshold")
        
    elif threshold_method == 'both':
        # Coba kedua metode dan pilih yang menghasilkan lebih banyak teks
        binary_adaptive = apply_adaptive_threshold(smoothed, is_light_bg, block_size=31, C=5)
        binary_otsu = apply_otsu_threshold(smoothed, is_light_bg)
        
        # Hitung white pixel ratio (asumsi: teks = putih)
        white_adaptive = np.sum(binary_adaptive == 255) / binary_adaptive.size
        white_otsu = np.sum(binary_otsu == 255) / binary_otsu.size
        
        # Pilih yang lebih balance (biasanya teks sekitar 10-30% dari gambar)
        # Terlalu banyak putih = over-threshold, terlalu sedikit = under-threshold
        target_ratio = 0.20
        diff_adaptive = abs(white_adaptive - target_ratio)
        diff_otsu = abs(white_otsu - target_ratio)
        
        if diff_adaptive < diff_otsu:
            binary = binary_adaptive
            result['threshold_method_used'] = 'adaptive'
            print(f"✓ Selected adaptive threshold (white ratio: {white_adaptive:.2%})")
        else:
            binary = binary_otsu
            result['threshold_method_used'] = 'otsu'
            print(f"✓ Selected Otsu threshold (white ratio: {white_otsu:.2%})")
    
    else:
        # Default: adaptive
        binary = apply_adaptive_threshold(smoothed, is_light_bg, block_size=31, C=5)
        result['threshold_method_used'] = 'adaptive'
        print("✓ Applied adaptive threshold (default)")
    
    # Langkah 8: Morphological operations untuk cleanup
    if morphology != 'none':
        binary = apply_morphological_operations(binary, operation=morphology, kernel_size=(2, 2))
        print(f"✓ Applied morphological operation: {morphology}")
    
    result['binary'] = binary
    
    print("=" * 60)
    print("PREPROCESSING SELESAI - Gambar siap untuk OCR")
    print("=" * 60)
    
    return result


def preprocess_roi_for_ocr(roi: np.ndarray,
                           is_light_background: bool,
                           skip_resize: bool = True) -> np.ndarray:
    """
    Preprocessing khusus untuk Region of Interest (ROI) yang sudah di-crop.
    Pipeline lebih ringan karena ROI biasanya sudah kecil dan fokus pada area teks.
    
    Parameters:
    -----------
    roi : np.ndarray
        Region of Interest (bagian gambar yang sudah di-crop)
    is_light_background : bool
        True jika background terang
    skip_resize : bool
        Skip resize untuk ROI (biasanya tidak perlu)
    
    Returns:
    --------
    np.ndarray
        Binary image siap untuk OCR
    """
    # Konversi ke grayscale jika perlu
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    # Denoise ringan
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # CLAHE untuk kontras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Threshold
    binary = apply_adaptive_threshold(enhanced, is_light_background, block_size=15, C=3)
    
    # Morphology ringan
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary


# Contoh penggunaan
if __name__ == "__main__":
    # Test dengan gambar sample
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_preprocessing.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    print("=" * 60)
    
    # Preprocess
    result = preprocess_for_ocr(
        image,
        denoise_method='bilateral',
        threshold_method='both',
        morphology='open_close',
        auto_resize=True,
        min_width=1200
    )
    
    # Simpan hasil
    output_path = image_path.replace('.', '_preprocessed.')
    cv2.imwrite(output_path, result['binary'])
    print(f"\n✓ Preprocessed image saved to: {output_path}")
    
    # Tampilkan info
    print(f"\nInfo:")
    print(f"  - Original size: {result['original_size']}")
    print(f"  - Processed size: {result['processed_size']}")
    print(f"  - Background: {'Light' if result['is_light_background'] else 'Dark'}")
    print(f"  - Mean intensity: {result['mean_intensity']:.1f}")
    print(f"  - Threshold method: {result['threshold_method_used']}")
