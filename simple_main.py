"""
Simple Receipt Processor
Extracts: date, total, and category from receipts.

pipeline: reciept_preprocessing -> easyOCR -> post_processing
-> simple data extraction -> categorization -> json exportation
"""
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core_processor import get_processor


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

    # Get processor instance
    processor = get_processor(use_gpu=True)

    # Process
    print("[1/1] Processing receipt...")
    result = processor.process(image_path)

    print(f"  Date: {result['date']}")
    print(f"  Total: {result['total']}")
    print(f"  Category: {result['category']}")

    # Save JSON
    if save_json:
        output_path = Path(image_path).stem + "_result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {output_path}")

    print("=" * 60)
    print("\nProcessing complete!")
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
            print(f"Error processing {image_path}: {e}")
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
    image_paths = ""

    process_receipt(image_paths[0])
