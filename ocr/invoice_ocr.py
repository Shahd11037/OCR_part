"""
Invoice OCR Module
Handles text extraction from invoice images using EasyOCR.
Supports both Arabic and English text with optional preprocessing.
"""

import easyocr
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path to import preprocessing
sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.preprocess import preprocess_image

# Import post-processor if available
try:
    from ocr.ocr_post_processor import OCRPostProcessor
    POST_PROCESSOR_AVAILABLE = True
except ImportError:
    POST_PROCESSOR_AVAILABLE = False


class InvoiceOCR:
    """
    OCR engine for invoice text extraction.
    Uses EasyOCR with Arabic and English language support.
    """

    def __init__(self, use_gpu: bool = False, languages: List[str] = None, enable_post_processing: bool = True):
        """
        Initialize the OCR reader.

        Args:
            use_gpu: Whether to use GPU acceleration (default: True)
            languages: List of language codes (default: ['en', 'ar'])
            enable_post_processing: Clean and correct common OCR errors (default: True)
        """
        if languages is None:
            languages = ['en', 'ar']

        print(f"Initializing EasyOCR with languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        print("EasyOCR initialized successfully")

        # Initialize post-processor
        self.enable_post_processing = enable_post_processing and POST_PROCESSOR_AVAILABLE
        if self.enable_post_processing:
            self.post_processor = OCRPostProcessor()
            print("OCR post-processing enabled")
        else:
            self.post_processor = None
            if enable_post_processing and not POST_PROCESSOR_AVAILABLE:
                print("Warning: Post-processor not available")

    def extract_text_with_boxes(
        self,
        image_path: str,
        use_preprocessing: bool = True,
        detail: int = 1,
        paragraph: bool = False
    ) -> List[Dict]:
        """
        Extract text from invoice image with bounding box information.

        Args:
            image_path: Path to the invoice image
            use_preprocessing: Apply preprocessing pipeline (deskew, denoise, etc.)
            detail: Level of detail (0=text only, 1=text+confidence+bbox)
            paragraph: Group text into paragraphs

        Returns:
            List of dictionaries containing:
            - text: Extracted text
            - confidence: Confidence score (0-1)
            - bbox: Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            - center: Center point of bounding box (x, y)
        """
        try:
            # Load and optionally preprocess the image
            if use_preprocessing:
                print(f"Preprocessing image: {image_path}")
                img = preprocess_image(image_path)
                if img is None:
                    print("Preprocessing failed, using original image")
                    img = cv2.imread(image_path)
            else:
                img = cv2.imread(image_path)

            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")

            # Get image dimensions for normalization
            img_height, img_width = img.shape[:2]

            print("Running OCR extraction...")
            # Extract text with bounding boxes
            results = self.reader.readtext(img, detail=1, paragraph=paragraph)

            # Format results into structured dictionaries
            formatted_results = []
            for bbox, text, confidence in results:
                # Calculate center point of bounding box
                bbox_array = np.array(bbox)
                center_x = int(np.mean(bbox_array[:, 0]))
                center_y = int(np.mean(bbox_array[:, 1]))

                # Calculate normalized positions (0-1 range)
                norm_center_x = center_x / img_width
                norm_center_y = center_y / img_height

                formatted_results.append({
                    'text': text.strip(),
                    'confidence': float(confidence),
                    'bbox': bbox,
                    'center': (center_x, center_y),
                    'normalized_center': (norm_center_x, norm_center_y),
                    'image_dimensions': (img_width, img_height)
                })

            print(f"Extracted {len(formatted_results)} text elements")

            # Apply post-processing if enabled
            if self.enable_post_processing and self.post_processor:
                print("Applying OCR post-processing...")
                formatted_results = self.post_processor.process_all(formatted_results)
                print("Post-processing complete")

            return formatted_results

        except Exception as e:
            print(f"Error during OCR extraction: {e}")
            raise

    def extract_text_only(
        self,
        image_path: str,
        use_preprocessing: bool = True
    ) -> str:
        """
        Extract only the text from an invoice (no bounding boxes).

        Args:
            image_path: Path to the invoice image
            use_preprocessing: Apply preprocessing pipeline

        Returns:
            Extracted text as a single string
        """
        results = self.extract_text_with_boxes(
            image_path,
            use_preprocessing=use_preprocessing
        )

        # Combine all text elements
        text_lines = [result['text'] for result in results]
        return '\n'.join(text_lines)

    def visualize_detections(
        self,
        image_path: str,
        output_path: str = None,
        use_preprocessing: bool = False
    ) -> np.ndarray:
        """
        Visualize OCR detections with bounding boxes on the image.
        Useful for debugging and validation.

        Args:
            image_path: Path to the invoice image
            output_path: Optional path to save the visualization
            use_preprocessing: Apply preprocessing pipeline

        Returns:
            Image with bounding boxes drawn
        """
        # Read original image for visualization
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Get OCR results
        results = self.extract_text_with_boxes(
            image_path,
            use_preprocessing=use_preprocessing
        )

        # Draw bounding boxes and text
        for result in results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']

            # Convert bbox to integer coordinates
            points = np.array(bbox, dtype=np.int32)

            # Draw bounding box
            cv2.polylines(img, [points], True, (0, 255, 0), 2)

            # Draw text and confidence
            label = f"{text[:20]}... ({confidence:.2f})"
            cv2.putText(
                img,
                label,
                (points[0][0], points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Visualization saved to: {output_path}")

        return img


# Example usage and testing
if __name__ == "__main__":
    # Initialize OCR
    ocr = InvoiceOCR(use_gpu=True)

    # Example: Extract text with bounding boxes
    try:
        results = ocr.extract_text_with_boxes(
            "../tabali.jpg",
            use_preprocessing=True
        )

        print("\n" + "="*60)
        print("OCR RESULTS")
        print("="*60)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Text: {result['text']}")
            print(f"    Confidence: {result['confidence']:.3f}")
            print(f"    Center: {result['center']}")
            print(f"    Normalized: {result['normalized_center']}")

        # Visualize detections
        ocr.visualize_detections(
            "../tabali.jpg",
            output_path="ocr_visualization.jpg"
        )

    except Exception as e:
        print(f"Error: {e}")