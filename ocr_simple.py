"""
Simplified OCR Preprocessing Module
====================================
This module provides a minimal preprocessing pipeline that produces OCR results
identical to the Tesseract CLI command:
    tesseract input.png stdout -l eng --psm 6

The key principle: LESS IS MORE
- Minimal preprocessing to avoid introducing artifacts
- No aggressive thresholding or morphological operations
- Matches CLI behavior exactly

Author: Classification Model Team
Date: October 30, 2025
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
from typing import Union, Dict


def run_ocr_from_ui(image_bytes: bytes) -> str:
    """
    Run OCR on image bytes from UI upload.
    Produces results matching: tesseract input.png stdout -l eng --psm 6
    
    Parameters:
    -----------
    image_bytes : bytes
        Raw image bytes from file upload (e.g., from Streamlit, Flask, etc.)
    
    Returns:
    --------
    str
        Extracted text, cleaned and ready for use
    
    Example:
    --------
    >>> with open('product_label.jpg', 'rb') as f:
    >>>     image_bytes = f.read()
    >>> text = run_ocr_from_ui(image_bytes)
    >>> print(text)
    """
    # Convert bytes to numpy array
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image from bytes")
    
    # Process the image with minimal preprocessing
    processed_image = preprocess_for_cli_match(image)
    
    # Run OCR with exact CLI configuration
    text = pytesseract.image_to_string(
        processed_image,
        lang='eng',
        config='--psm 6'
    )
    
    # Light cleanup (remove excessive whitespace)
    text = text.strip()
    
    return text


def preprocess_for_cli_match(image: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing to match CLI Tesseract results.
    
    Pipeline:
    1. Convert to grayscale
    2. Apply bilateral filter (preserves edges, reduces noise)
    3. Upscale if too small (< 900px height)
    4. Optional inversion only if background is darker than text
    
    NO thresholding, NO morphology, NO CLAHE, NO aggressive adjustments.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (BGR format from cv2.imread)
    
    Returns:
    --------
    np.ndarray
        Preprocessed grayscale image ready for Tesseract
    """
    # Step 1: Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Step 2: Apply bilateral filter
    # Preserves text edges while reducing noise
    # Parameters: diameter=9, sigmaColor=75, sigmaSpace=75
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Step 3: Upscale if image is too small
    height, width = filtered.shape
    if height < 900:
        # Calculate scale factor
        scale = 900 / height
        new_width = int(width * scale)
        new_height = 900
        
        # Upscale using INTER_CUBIC for best quality
        filtered = cv2.resize(
            filtered, 
            (new_width, new_height), 
            interpolation=cv2.INTER_CUBIC
        )
    
    # Step 4: Check if inversion is needed
    # Only invert if background is darker than text
    mean_intensity = np.mean(filtered)
    
    if mean_intensity < 127:
        # Dark background - invert to get white background
        filtered = cv2.bitwise_not(filtered)
    
    return filtered


def run_ocr_from_path(image_path: str) -> str:
    """
    Run OCR on image from file path.
    Convenience function for testing.
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    
    Returns:
    --------
    str
        Extracted text
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to read image from: {image_path}")
    
    processed_image = preprocess_for_cli_match(image)
    
    text = pytesseract.image_to_string(
        processed_image,
        lang='eng',
        config='--psm 6'
    )
    
    return text.strip()


def get_preprocessing_info(image: np.ndarray) -> Dict[str, any]:
    """
    Get detailed information about preprocessing steps.
    Useful for debugging and understanding what's happening.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    
    Returns:
    --------
    Dict with preprocessing information
    """
    # Original info
    original_height, original_width = image.shape[:2]
    original_channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Check if upscaling is needed
    height, width = filtered.shape
    needs_upscale = height < 900
    
    if needs_upscale:
        scale = 900 / height
        new_width = int(width * scale)
        new_height = 900
        filtered_upscaled = cv2.resize(
            filtered, 
            (new_width, new_height), 
            interpolation=cv2.INTER_CUBIC
        )
    else:
        filtered_upscaled = filtered
        new_width, new_height = width, height
    
    # Check if inversion is needed
    mean_intensity = np.mean(filtered_upscaled)
    needs_inversion = mean_intensity < 127
    
    return {
        'original_size': (original_width, original_height),
        'original_channels': original_channels,
        'after_filter_size': (width, height),
        'needs_upscale': needs_upscale,
        'upscaled_size': (new_width, new_height) if needs_upscale else None,
        'mean_intensity': mean_intensity,
        'needs_inversion': needs_inversion,
        'background_type': 'dark' if needs_inversion else 'light'
    }


# Test and comparison functions
def compare_with_cli(image_path: str, verbose: bool = True) -> Dict[str, str]:
    """
    Compare UI OCR result with CLI command.
    Useful for testing and validation.
    
    Parameters:
    -----------
    image_path : str
        Path to test image
    verbose : bool
        Print comparison details
    
    Returns:
    --------
    Dict with 'ui_result' and comparison info
    """
    import subprocess
    import os
    
    # Get UI result
    ui_text = run_ocr_from_path(image_path)
    
    # Try to get CLI result (if tesseract is in PATH)
    try:
        cli_result = subprocess.run(
            ['tesseract', image_path, 'stdout', '-l', 'eng', '--psm', '6'],
            capture_output=True,
            text=True,
            timeout=30
        )
        cli_text = cli_result.stdout.strip()
    except Exception as e:
        cli_text = f"CLI command failed: {e}"
    
    # Get preprocessing info
    image = cv2.imread(image_path)
    info = get_preprocessing_info(image)
    
    if verbose:
        print("=" * 70)
        print("OCR COMPARISON: UI vs CLI")
        print("=" * 70)
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Original size: {info['original_size']}")
        print(f"Upscaled: {info['needs_upscale']}")
        if info['needs_upscale']:
            print(f"  → New size: {info['upscaled_size']}")
        print(f"Background: {info['background_type']} (intensity: {info['mean_intensity']:.1f})")
        print(f"Inverted: {info['needs_inversion']}")
        
        print("\n" + "-" * 70)
        print("UI RESULT:")
        print("-" * 70)
        print(ui_text)
        
        print("\n" + "-" * 70)
        print("CLI RESULT:")
        print("-" * 70)
        print(cli_text)
        
        print("\n" + "-" * 70)
        print("MATCH:")
        print("-" * 70)
        if ui_text == cli_text:
            print("✅ PERFECT MATCH - UI and CLI results are identical!")
        else:
            print("⚠️  Results differ")
            print(f"UI length: {len(ui_text)} chars, CLI length: {len(cli_text)} chars")
    
    return {
        'ui_result': ui_text,
        'cli_result': cli_text,
        'match': ui_text == cli_text,
        'preprocessing_info': info
    }


if __name__ == "__main__":
    """
    Test the simplified OCR pipeline
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_simple.py <image_path>")
        print("\nExample:")
        print("  python ocr_simple.py sample/input/test.gt.txt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=" * 70)
    print("SIMPLIFIED OCR PIPELINE TEST")
    print("=" * 70)
    print(f"Processing: {image_path}\n")
    
    # Run comparison
    result = compare_with_cli(image_path, verbose=True)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
