import easyocr
import pytesseract
import cv2
import os
import json
from difflib import SequenceMatcher
import numpy as np
from tqdm import tqdm
from preprocessing.preprocess import preprocess_image as preprocess


class OCRComparator:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.ocr_dataset_dir = os.path.join(data_dir, "ocr_dataset")
        self.results = {
            "easyocr": [],
            "pytesseract": [],
            "comparison": []
        }

        # Initialize OCR engines
        self.easyocr_reader = easyocr.Reader(['en', 'ar'],gpu=True)
        print("EasyOCR initialized with English and Arabic")

    def character_error_rate(self, ground_truth, predicted):
        """
        Calculate Character Error Rate (CER)
        CER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions, N = reference length
        """
        # Use SequenceMatcher to calculate similarity
        matcher = SequenceMatcher(None, ground_truth, predicted)
        ratio = matcher.ratio()

        # CER = 1 - similarity ratio
        cer = 1 - ratio
        return cer, ratio

    def extract_text_easyocr(self, image_path):
        """Extract text using EasyOCR"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, "Failed to read image"

            results = self.easyocr_reader.readtext(img, detail=0)
            text = "\n".join(results)
            return text, None
        except Exception as e:
            return None, str(e)

    def extract_text_pytesseract(self, image_path):
        """Extract text using Pytesseract with English and Arabic"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, "Failed to read image"

            # Use both English and Arabic language models
            # 'eng' for English, 'ara' for Arabic
            text = pytesseract.image_to_string(image_path, lang='eng+ara')
            return text, None
        except Exception as e:
            return None, str(e)

    def read_ground_truth(self, text_file_path):
        """Read ground truth text from file"""
        try:
            if not os.path.exists(text_file_path):
                return None

            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text
        except Exception as e:
            print(f"Error reading {text_file_path}: {e}")
            return None

    def normalize_text(self, text):
        """Normalize text for comparison (lowercase, remove extra spaces)"""
        if text is None:
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def process_dataset(self, split="train"):
        """
        Process OCR dataset for a specific split (train/val/test)
        Images and text files are in the same directory
        """
        split_dir = os.path.join(self.ocr_dataset_dir, split)

        if not os.path.exists(split_dir):
            print(f"Directory not found: {split_dir}")
            return

        # Get all image files from the split directory
        image_files = sorted([f for f in os.listdir(split_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"\nProcessing {split} split ({len(image_files)} images)...")

        for image_file in tqdm(image_files):
            image_path = os.path.join(split_dir, image_file)

            # Get corresponding text file
            text_file_name = os.path.splitext(image_file)[0] + ".txt"
            text_file_path = os.path.join(split_dir, text_file_name)

            # Read ground truth
            ground_truth = self.read_ground_truth(text_file_path)
            if ground_truth is None:
                continue

            ground_truth_norm = self.normalize_text(ground_truth)

            # Preprocess the image
            try:
                preprocessed_img = preprocess(image_path)
                if preprocessed_img is None:
                    continue
                # Save preprocessed image temporarily
                temp_preprocessed_path = f"temp/{os.path.splitext(image_file)[0]}_preprocessed.jpg"
                cv2.imwrite(temp_preprocessed_path, preprocessed_img)
                ocr_image_path = temp_preprocessed_path
            except Exception as e:
                print(f"Preprocessing failed for {image_file}: {e}")
                ocr_image_path = image_path

            # Extract text with EasyOCR
            easyocr_text, easyocr_error = self.extract_text_easyocr(ocr_image_path)
            easyocr_text_norm = self.normalize_text(easyocr_text)

            # Extract text with Pytesseract
            pytesseract_text, pytesseract_error = self.extract_text_pytesseract(ocr_image_path)
            pytesseract_text_norm = self.normalize_text(pytesseract_text)

            # Calculate CER for both
            easyocr_cer, easyocr_ratio = self.character_error_rate(
                ground_truth_norm, easyocr_text_norm
            ) if easyocr_error is None else (1.0, 0.0)

            pytesseract_cer, pytesseract_ratio = self.character_error_rate(
                ground_truth_norm, pytesseract_text_norm
            ) if pytesseract_error is None else (1.0, 0.0)

            # Store results
            result = {
                "image": image_file,
                "split": split,
                "ground_truth": ground_truth[:100],  # First 100 chars for display
                "easyocr": {
                    "text": easyocr_text[:100] if easyocr_text else "ERROR",
                    "cer": float(easyocr_cer),
                    "accuracy": float(easyocr_ratio) * 100,
                    "error": easyocr_error
                },
                "pytesseract": {
                    "text": pytesseract_text[:100] if pytesseract_text else "ERROR",
                    "cer": float(pytesseract_cer),
                    "accuracy": float(pytesseract_ratio) * 100,
                    "error": pytesseract_error
                },
                "better": "easyocr" if easyocr_cer < pytesseract_cer else "pytesseract"
            }

            self.results["comparison"].append(result)
            self.results["easyocr"].append(easyocr_cer)
            self.results["pytesseract"].append(pytesseract_cer)

    def print_summary(self):
        """Print summary statistics"""
        if not self.results["easyocr"]:
            print("No results to summarize")
            return

        easyocr_scores = np.array(self.results["easyocr"])
        pytesseract_scores = np.array(self.results["pytesseract"])

        print("\n" + "=" * 60)
        print("OCR COMPARISON SUMMARY")
        print("=" * 60)

        print("\nEasyOCR Results:")
        print(f"  Mean CER: {easyocr_scores.mean():.4f}")
        print(f"  Median CER: {np.median(easyocr_scores):.4f}")
        print(f"  Std Dev: {easyocr_scores.std():.4f}")
        print(f"  Min CER: {easyocr_scores.min():.4f}")
        print(f"  Max CER: {easyocr_scores.max():.4f}")
        print(f"  Mean Accuracy: {(1 - easyocr_scores.mean()) * 100:.2f}%")

        print("\nPytesseract Results:")
        print(f"  Mean CER: {pytesseract_scores.mean():.4f}")
        print(f"  Median CER: {np.median(pytesseract_scores):.4f}")
        print(f"  Std Dev: {pytesseract_scores.std():.4f}")
        print(f"  Min CER: {pytesseract_scores.min():.4f}")
        print(f"  Max CER: {pytesseract_scores.max():.4f}")
        print(f"  Mean Accuracy: {(1 - pytesseract_scores.mean()) * 100:.2f}%")

        # Determine winner
        easyocr_mean = easyocr_scores.mean()
        pytesseract_mean = pytesseract_scores.mean()

        print("\n" + "-" * 60)
        if easyocr_mean < pytesseract_mean:
            print(f"WINNER: EasyOCR (lower CER: {easyocr_mean:.4f} vs {pytesseract_mean:.4f})")
        else:
            print(f"WINNER: Pytesseract (lower CER: {pytesseract_mean:.4f} vs {easyocr_mean:.4f})")
        print("=" * 60 + "\n")

    def save_results(self, output_file="ocr_results.json"):
        """Save detailed results to JSON file"""
        # Calculate statistics
        stats = {
            "total_images": len(self.results["comparison"]),
            "easyocr_mean_cer": float(np.mean(self.results["easyocr"])) if self.results["easyocr"] else 0,
            "pytesseract_mean_cer": float(np.mean(self.results["pytesseract"])) if self.results["pytesseract"] else 0,
            "results": self.results["comparison"]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_file}")


# Main execution
if __name__ == "__main__":
    # Initialize comparator
    comparator = OCRComparator(data_dir="data")

    # Process each split
    for split in ["train", "val", "test"]:
        comparator.process_dataset(split=split)

    # Print summary
    comparator.print_summary()

    # Save results
    comparator.save_results(output_file="ocr_results.json")