"""
Receipt Preprocessing Module
Specialized preprocessing for thermal receipts and low-contrast documents.
Optimized for Arabic/English mixed text.
"""

import cv2
import numpy as np
from typing import Optional


def preprocess_receipt(image_path: str, enhance_contrast: bool = True) -> Optional[np.ndarray]:
    """
    Preprocess receipt image with settings optimized for thermal receipts.
    Less aggressive than standard invoice preprocessing.

    Args:
        image_path: Path to the receipt image
        enhance_contrast: Apply aggressive contrast enhancement (default: True)

    Returns:
        Preprocessed image or None on error
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error reading image: {image_path}")
        return None

    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize if image is too small (improves OCR)
    height, width = gray.shape
    if width < 1000:
        scale = 1000 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"Upscaled image from {width}x{height} to {new_width}x{new_height}")

    # Step 3: Aggressive contrast enhancement for low-contrast receipts
    if enhance_contrast:
        # CLAHE with higher clip limit for receipts
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    else:
        enhanced = gray

    # Step 4: Denoise but preserve text
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 5: Adaptive threshold with larger block size for receipts
    # Larger block size = less sensitive to local variations
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,  # Larger block size for receipts
        10  # Lower constant = more text preserved
    )

    # Step 6: Very light morphological operations
    # Receipts need less morphology than invoices
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Light dilation to thicken thin text
    result = cv2.dilate(thresh, kernel, iterations=1)

    return result


def preprocess_receipt_conservative(image_path: str) -> Optional[np.ndarray]:
    """
    Conservative preprocessing - minimal changes.
    Use this if aggressive preprocessing is hurting accuracy.

    Args:
        image_path: Path to the receipt image

    Returns:
        Preprocessed image or None on error
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error reading image: {image_path}")
        return None

    # Just grayscale and light enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize if too small
    height, width = gray.shape
    if width < 1000:
        scale = 1000 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Light contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Very light denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=5, templateWindowSize=7, searchWindowSize=15)

    return denoised


# Example usage
if __name__ == "__main__":
    import sys

    image_path = "..//tabali.jpg"

    # Test both methods
    print("Processing with aggressive enhancement...")
    result1 = preprocess_receipt(image_path, enhance_contrast=True)
    if result1 is not None:
        cv2.imwrite("receipt_aggressive.jpg", result1)
        print("Saved: receipt_aggressive.jpg")

    print("\nProcessing with conservative method...")
    result2 = preprocess_receipt_conservative(image_path)
    if result2 is not None:
        cv2.imwrite("receipt_conservative.jpg", result2)
        print("Saved: receipt_conservative.jpg")

    print("\nCompare the outputs to see which works better for your receipts!")