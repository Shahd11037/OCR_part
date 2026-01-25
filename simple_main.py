"""
Simple Receipt Processor
Extracts: date, total, and category from receipts.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr.invoice_ocr import InvoiceOCR
from preprocessing.reciept_preprocessing import preprocess_receipt_conservative
from extraction.simple_extractor import SimpleExtractor
from validation.categorizer import ReceiptCategorizer
import cv2


def process_receipt(image_path: str, save_json: bool = True) -> dict:
    """
    Process a receipt image and extract date, total, and category.

    Args:
        image_path: Path to receipt image
        save_json: Whether to save result as JSON file

    Returns:
        Dict with date, total, and category
    """
    print(f"Processing: {image_path}")
    print("=" * 60)

    # Step 1: Preprocess
    print("[1/4] Preprocessing image...")
    preprocessed = preprocess_receipt_conservative(image_path)

    if preprocessed is not None:
        # Save preprocessed image temporarily
        temp_path = "/tmp/preprocessed_receipt.jpg"
        cv2.imwrite(temp_path, preprocessed)
        ocr_image_path = temp_path
    else:
        print("  Warning: Preprocessing failed, using original image")
        ocr_image_path = image_path

    # Step 2: OCR
    print("[2/4] Running OCR...")
    ocr = InvoiceOCR(use_gpu=True, enable_post_processing=True)
    ocr_results = ocr.extract_text_with_boxes(ocr_image_path, use_preprocessing=False)
    print(f"  Extracted {len(ocr_results)} text elements")

    # Step 3: Extract fields
    print("[3/4] Extracting date and total...")
    extractor = SimpleExtractor()
    extracted = extractor.extract(ocr_results)

    print(f"  Date: {extracted['date']}")
    print(f"  Total: {extracted['total']}")
    print(f"  Line items found: {len(extracted['line_items'])}")

    # Step 4: Categorize
    print("[4/4] Categorizing receipt...")

    # Get vendor name (usually first or second text element)
    vendor_name = ""
    if len(ocr_results) > 0:
        vendor_name = ocr_results[0]['text']

    categorizer = ReceiptCategorizer()
    category = categorizer.categorize(vendor_name, extracted['line_items'])
    print(f"  Category: {category}")

    # Build final result
    result = {
        "date": extracted['date'],
        "total": extracted['total'],
        "category": category
    }

    # Save JSON
    if save_json:
        output_path = Path(image_path).stem + "_result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved result to: {output_path}")

    print("=" * 60)
    print("\n Processing complete!")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


def process_batch(image_paths: list) -> list:
    """
    Process multiple receipts.

    Args:
        image_paths: List of image paths

    Returns:
        List of results
    """
    results = []

    for i, image_path in enumerate(image_paths, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing {i}/{len(image_paths)}: {image_path}")
        print(f"{'=' * 60}\n")

        try:
            result = process_receipt(image_path, save_json=True)
            results.append({
                "file": image_path,
                "success": True,
                **result
            })
        except Exception as e:
            print(f" Error processing {image_path}: {e}")
            results.append({
                "file": image_path,
                "success": False,
                "error": str(e)
            })

    # Save batch summary
    with open("batch_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"BATCH SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('success'))}")
    print(f"Failed: {sum(1 for r in results if not r.get('success'))}")
    print(f"\nResults saved to: batch_results.json")

    return results


if __name__ == "__main__":

    image_paths = ["D:/Dying/4th year/seventh term/GP/ocr part/chicken_fila.jpg"]

    if len(image_paths) == 1:
        # Single receipt
        process_receipt(image_paths[0])