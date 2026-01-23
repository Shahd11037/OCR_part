"""
Receipt Testing Script
Tests different OCR configurations to find the best settings for receipts.
"""

import sys

sys.path.append('/home/claude/invoice_processor')

from ocr.invoice_ocr import InvoiceOCR
from preprocessing.preprocess import preprocess_image
from preprocessing.reciept_preprocessing import preprocess_receipt, preprocess_receipt_conservative
import cv2


def test_receipt(image_path: str):
    """Test receipt with different configurations"""

    print("=" * 70)
    print("RECEIPT OCR TESTING")
    print("=" * 70)
    print(f"Image: {image_path}\n")

    # Configuration 1: Standard with post-processing
    print("\n" + "=" * 70)
    print("TEST 1: Standard preprocessing + Post-processing")
    print("=" * 70)
    ocr1 = InvoiceOCR(use_gpu=True, enable_post_processing=True)
    results1 = ocr1.extract_text_with_boxes(image_path, use_preprocessing=True)
    print_key_results(results1, "Test 1")

    # Configuration 2: Receipt-specific preprocessing + Post-processing
    print("\n" + "=" * 70)
    print("TEST 2: Receipt preprocessing + Post-processing")
    print("=" * 70)
    # Preprocess with receipt settings
    preprocessed = preprocess_receipt(image_path)
    if preprocessed is not None:
        temp_path = "/tmp/receipt_preprocessed.jpg"
        cv2.imwrite(temp_path, preprocessed)

        ocr2 = InvoiceOCR(use_gpu=True, enable_post_processing=True)
        results2 = ocr2.extract_text_with_boxes(temp_path, use_preprocessing=False)
        print_key_results(results2, "Test 2")

    # Configuration 3: No preprocessing + Post-processing
    print("\n" + "=" * 70)
    print("TEST 3: No preprocessing + Post-processing")
    print("=" * 70)
    ocr3 = InvoiceOCR(use_gpu=True, enable_post_processing=True)
    results3 = ocr3.extract_text_with_boxes(image_path, use_preprocessing=False)
    print_key_results(results3, "Test 3")

    # Configuration 4: Conservative preprocessing + Post-processing
    print("\n" + "=" * 70)
    print("TEST 4: Conservative preprocessing + Post-processing")
    print("=" * 70)
    preprocessed_cons = preprocess_receipt_conservative(image_path)
    if preprocessed_cons is not None:
        temp_path_cons = "/tmp/receipt_conservative.jpg"
        cv2.imwrite(temp_path_cons, preprocessed_cons)

        ocr4 = InvoiceOCR(use_gpu=True, enable_post_processing=True)
        results4 = ocr4.extract_text_with_boxes(temp_path_cons, use_preprocessing=False)
        print_key_results(results4, "Test 4")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("Compare the results above and choose the configuration that works best.")
    print("For thermal receipts, Test 2 or Test 4 typically work best.")


def print_key_results(results, test_name):
    """Print key fields from OCR results"""
    print(f"\n{test_name} Results:")
    print(f"Total elements: {len(results)}\n")

    # Look for key fields
    date_found = False
    total_found = False
    vat_found = False

    for r in results:
        text = r['text']
        original = r.get('original_text', text)

        # Show corrections if post-processing was applied
        if 'original_text' in r and original != text:
            print(f"  ✓ Corrected: '{original}' → '{text}' (confidence: {r['confidence']:.2f})")

        # Highlight key fields
        if 'date' in text.lower() or '2022' in text or '2023' in text or '2024' in text:
            if not date_found:
                print(f"  📅 Date: {text}")
                date_found = True

        if 'total' in text.lower() and not 'sub' in text.lower():
            if not total_found:
                print(f"  💰 Total: {text}")
                total_found = True

        if 'vat' in text.lower() or 'tax' in text.lower():
            if not vat_found:
                print(f"  🧾 VAT: {text}")
                vat_found = True

        # Show currency
        if 'egp' in text.lower() or 'sar' in text.lower():
            print(f"  💵 Currency: {text}")

    # Show amounts
    print("\n  Amounts found:")
    for r in results:
        text = r['text']
        # Look for decimal numbers
        import re
        if re.search(r'\d+\.\d{2}', text):
            print(f"    {text}")


if __name__ == "__main__":
    image_path = "../tabali.jpg"
    test_receipt(image_path)