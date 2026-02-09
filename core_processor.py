"""
Core Receipt Processor
Shared logic for processing receipts.
Used by both API and simple_main
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional

from ocr.invoice_ocr import InvoiceOCR
from preprocessing.reciept_preprocessing import preprocess_receipt_conservative
from extraction.simple_extractor import SimpleExtractor
from validation.categorizer import ReceiptCategorizer
import cv2


class ReceiptProcessor:
    """
    Core receipt processing engine.
    Handles OCR, extraction, and categorization.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the processor.

        Args:
            use_gpu: Whether to use GPU for OCR (faster but requires CUDA)
        """
        print("Initializing Receipt Processor...")

        # Initialize components
        self.ocr_engine = InvoiceOCR(use_gpu=use_gpu, enable_post_processing=True)
        self.extractor = SimpleExtractor()
        self.categorizer = ReceiptCategorizer()

        print("Receipt Processor ready")

    def process(self, image_path: str) -> Dict:
        """
        Process a receipt image.

        Args:
            image_path: Path to receipt image

        Returns:
            Dict with:
            - date: Transaction date (YYYY-MM-DD)
            - total: Total amount (float)
            - category: Spending category (string)
        """

        # Step 1: Preprocess
        preprocessed = preprocess_receipt_conservative(image_path)

        if preprocessed is not None:
            # Save preprocessed image temporarily
            temp_preprocessed_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(temp_preprocessed_path, preprocessed)
            ocr_image_path = temp_preprocessed_path
        else:
            ocr_image_path = image_path

        try:
            # Step 2: OCR
            ocr_results = self.ocr_engine.extract_text_with_boxes(
                ocr_image_path,
                use_preprocessing=False  # Already preprocessed
            )

            # Step 3: Extract fields
            extracted = self.extractor.extract(ocr_results)

            # Step 4: Categorize
            vendor_name = ""
            if len(ocr_results) > 0:
                vendor_name = ocr_results[0]['text']

            category = self.categorizer.categorize(vendor_name, extracted['line_items'])

            # Build result
            result = {
                "date": extracted['date'],
                "total": extracted['total'],
                "category": category
            }

            return result

        finally:
            # Clean up temporary preprocessed file
            if preprocessed is not None and os.path.exists(temp_preprocessed_path):
                os.unlink(temp_preprocessed_path)

    def process_with_details(self, image_path: str) -> Dict:
        """
        Process receipt with additional debug information.

        Args:
            image_path: Path to receipt image

        Returns:
            Dict with date, total, category, plus debug info
        """

        # Preprocess
        preprocessed = preprocess_receipt_conservative(image_path)

        if preprocessed is not None:
            temp_preprocessed_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(temp_preprocessed_path, preprocessed)
            ocr_image_path = temp_preprocessed_path
        else:
            ocr_image_path = image_path

        try:
            # OCR
            ocr_results = self.ocr_engine.extract_text_with_boxes(
                ocr_image_path,
                use_preprocessing=False
            )

            # Extract
            extracted = self.extractor.extract(ocr_results)

            # Categorize
            vendor_name = ocr_results[0]['text'] if len(ocr_results) > 0 else ""
            category = self.categorizer.categorize(vendor_name, extracted['line_items'])

            # Calculate stats
            avg_confidence = sum(r['confidence'] for r in ocr_results) / len(ocr_results) if ocr_results else 0

            # Build detailed result
            result = {
                "date": extracted['date'],
                "total": extracted['total'],
                "category": category,
                "debug_info": {
                    "ocr_elements_found": len(ocr_results),
                    "average_confidence": round(avg_confidence, 2),
                    "line_items_detected": len(extracted['line_items']),
                    "vendor_name": vendor_name,
                    "line_items": extracted['line_items'][:5]  # First 5
                }
            }

            return result

        finally:
            if preprocessed is not None and os.path.exists(temp_preprocessed_path):
                os.unlink(temp_preprocessed_path)


# Singleton instance for reuse
_processor_instance = None


def get_processor(use_gpu: bool = True) -> ReceiptProcessor:
    """
    Get or create processor instance (singleton pattern).
    Reuses the same instance to avoid reinitializing OCR engine.

    Args:
        use_gpu: Whether to use GPU

    Returns:
        ReceiptProcessor instance
    """
    global _processor_instance

    if _processor_instance is None:
        _processor_instance = ReceiptProcessor(use_gpu=use_gpu)

    return _processor_instance


# Convenience functions for quick use
def process_receipt(image_path: str) -> Dict:
    """
    Quick function to process a receipt.

    Args:
        image_path: Path to receipt image

    Returns:
        Dict with date, total, category
    """
    processor = get_processor()
    return processor.process(image_path)


def process_receipt_detailed(image_path: str) -> Dict:
    """
    Quick function to process receipt with debug info.

    Args:
        image_path: Path to receipt image

    Returns:
        Dict with date, total, category, and debug info
    """
    processor = get_processor()
    return processor.process_with_details(image_path)


# Example usage
if __name__ == "__main__":

    image_path = ""

    print(f"Processing: {image_path}")
    result = process_receipt(image_path)

    print("\nResult:")
    print(f"  Date:     {result['date']}")
    print(f"  Total:    {result['total']}")
    print(f"  Category: {result['category']}")