"""
Demo Script
Demonstrates the complete invoice processing pipeline.
This is a simplified demo - for production use, see main.py
"""

import json
from ocr.invoice_ocr import InvoiceOCR
from extraction.layout_analyzer import LayoutAnalyzer
from extraction.field_extractor import FieldExtractor
from validation.validator import InvoiceValidator
from preprocessing.reciept_preprocessing import preprocess_receipt_conservative
import cv2


def process_invoice(image_path: str, output_json_path: str = None):
    """
    Process an invoice image through the complete pipeline.

    Args:
        image_path: Path to invoice image
        output_json_path: Optional path to save JSON output

    Returns:
        Dictionary with extracted invoice data
    """
    print("=" * 70)
    print("INVOICE PROCESSING PIPELINE")
    print("=" * 70)

    # Step 1: OCR Extraction
    print("\n[1/4] Running OCR extraction...")

    # Apply conservative receipt preprocessing (Test 4 configuration)
    print("    Applying conservative preprocessing...")
    preprocessed_img = preprocess_receipt_conservative(image_path)

    if preprocessed_img is not None:
        # Save preprocessed image temporarily
        temp_preprocessed_path = "temp_preprocessed.jpg"
        cv2.imwrite(temp_preprocessed_path, preprocessed_img)

        # Run OCR on preprocessed image with post-processing enabled
        ocr = InvoiceOCR(use_gpu=True, enable_post_processing=True)
        ocr_results = ocr.extract_text_with_boxes(
            temp_preprocessed_path,
            use_preprocessing=False  # Already preprocessed
        )

        # Clean up temp file
        try:
            import os
            os.remove(temp_preprocessed_path)
        except:
            pass
    else:
        # Fallback if preprocessing fails
        print("    Preprocessing failed, using original image...")
        ocr = InvoiceOCR(use_gpu=True, enable_post_processing=True)
        ocr_results = ocr.extract_text_with_boxes(
            image_path,
            use_preprocessing=False
        )

    print(f"    ✓ Extracted {len(ocr_results)} text elements")

    # Step 2: Layout Analysis
    print("\n[2/4] Analyzing layout...")
    analyzer = LayoutAnalyzer()
    layout = analyzer.analyze(ocr_results)
    print(f"    ✓ Detected {len(layout['zones'])} zones")
    print(f"    ✓ Found {len(layout['key_value_pairs'])} key-value pairs")
    print(f"    ✓ Identified {len(layout['tables'])} tables")

    # Step 3: Field Extraction
    print("\n[3/4] Extracting invoice fields...")
    extractor = FieldExtractor()
    fields = extractor.extract_all_fields(ocr_results, layout)
    print(f"    ✓ Extracted {fields['metadata']['fields_extracted']}/{fields['metadata']['total_fields']} fields")
    print(f"    ✓ Overall confidence: {fields['metadata']['extraction_confidence']:.1%}")

    # Step 4: Validation
    print("\n[4/4] Validating extracted data...")
    validator = InvoiceValidator()
    validation = validator.validate_all(fields)
    print(f"    ✓ Validation status: {validation['overall_status']}")
    print(f"    ✓ Quality score: {validation['quality_score']:.1f}%")
    print(f"    ✓ Errors: {len(validation['errors'])}, Warnings: {len(validation['warnings'])}")

    # Display results
    print("\n" + "=" * 70)
    print("EXTRACTED INVOICE DATA")
    print("=" * 70)

    print(f"\n Invoice Number: {fields['invoice_number']['value'] or 'Not found'}")
    if fields['invoice_number']['value']:
        print(f"   Confidence: {fields['invoice_number']['confidence']:.1%}")
        print(f"   Source: {fields['invoice_number']['source']}")

    print(f"\n Invoice Date: {fields['dates']['invoice_date']['value'] or 'Not found'}")
    if fields['dates']['invoice_date']['value']:
        print(f"   Confidence: {fields['dates']['invoice_date']['confidence']:.1%}")

    print(f"\n Due Date: {fields['dates']['due_date']['value'] or 'Not found'}")
    if fields['dates']['due_date']['value']:
        print(f"   Confidence: {fields['dates']['due_date']['confidence']:.1%}")

    print(f"\n Currency: {fields['currency']['value']}")

    print(f"\n Amounts:")
    for amount_type in ['subtotal', 'tax', 'discount', 'total']:
        amount_data = fields['amounts'][amount_type]
        if amount_data['value']:
            print(f"   {amount_type.capitalize()}: {amount_data['value']:.2f} {fields['currency']['value']}")
            print(f"      (Confidence: {amount_data['confidence']:.1%}, Source: {amount_data['source']})")

    print(f"\n Vendor Information:")
    vendor = fields['vendor_info']
    if vendor['name']['value']:
        print(f"   Name: {vendor['name']['value']}")
    if vendor['phone']['value']:
        print(f"   Phone: {vendor['phone']['value']}")
    if vendor['email']['value']:
        print(f"   Email: {vendor['email']['value']}")

    print(f"\n Customer Information:")
    customer = fields['customer_info']
    if customer['name']['value']:
        print(f"   Name: {customer['name']['value']}")

    print(f"\n Line Items: {len(fields['line_items'])} items")
    for i, item in enumerate(fields['line_items'][:5], 1):  # Show first 5
        print(f"   {i}. {item['description'][:50]}")
        if item['quantity']:
            print(f"      Qty: {item['quantity']}", end="")
        if item['unit_price']:
            print(f" × {item['unit_price']:.2f}", end="")
        if item['amount']:
            print(f" = {item['amount']:.2f}")
        else:
            print()

    if len(fields['line_items']) > 5:
        print(f"   ... and {len(fields['line_items']) - 5} more items")

    print(f"\n Tax Information:")
    tax = fields['tax_info']
    if tax['tax_id']['value']:
        print(f"   Tax ID: {tax['tax_id']['value']}")
    if tax['tax_percentage']['value']:
        print(f"   Tax Rate: {tax['tax_percentage']['value']:.1f}%")

    if fields['payment_terms']['value']:
        print(f"\n Payment Terms: {fields['payment_terms']['value']}")

    print(f"\n{'=' * 70}")
    print("VALIDATION RESULTS")
    print(f"{'=' * 70}")
    print(f"\nOverall Status: {validation['overall_status']}")
    print(f"Quality Score: {validation['quality_score']:.1f}%")
    print(f"Passed Checks: {validation['summary']['passed']}/{validation['summary']['total_checks']}")

    if validation['errors']:
        print(f"\nERRORS ({len(validation['errors'])}):")
        for error in validation['errors']:
            print(f"   • {error['field']}: {error['message']}")
            if error['suggested_fix']:
                print(f"     Fix: {error['suggested_fix']}")

    if validation['warnings']:
        print(f"\nWARNINGS ({len(validation['warnings'])}):")
        for warning in validation['warnings'][:5]:  # Show first 5
            print(f"   • {warning['field']}: {warning['message']}")
        if len(validation['warnings']) > 5:
            print(f"   ... and {len(validation['warnings']) - 5} more warnings")

    if validation['requires_manual_review']:
        print(f"\nMANUAL REVIEW REQUIRED")
    else:
        print(f"\nINVOICE VALIDATED - READY FOR PROCESSING")

    # Save to JSON if requested
    if output_json_path:
        # Convert to JSON-serializable format
        output_data = {
            'invoice_number': fields['invoice_number']['value'],
            'dates': {
                'invoice_date': fields['dates']['invoice_date']['value'],
                'due_date': fields['dates']['due_date']['value']
            },
            'currency': fields['currency']['value'],
            'amounts': {
                'subtotal': fields['amounts']['subtotal']['value'],
                'tax': fields['amounts']['tax']['value'],
                'discount': fields['amounts']['discount']['value'],
                'total': fields['amounts']['total']['value']
            },
            'vendor': {
                'name': vendor['name']['value'],
                'phone': vendor['phone']['value'],
                'email': vendor['email']['value']
            },
            'customer': {
                'name': customer['name']['value']
            },
            'line_items': fields['line_items'],
            'tax_info': {
                'tax_id': tax['tax_id']['value'],
                'tax_percentage': tax['tax_percentage']['value']
            },
            'payment_terms': fields['payment_terms']['value'],
            'metadata': fields['metadata'],
            'validation': {
                'status': validation['overall_status'],
                'quality_score': validation['quality_score'],
                'requires_manual_review': validation['requires_manual_review'],
                'errors_count': len(validation['errors']),
                'warnings_count': len(validation['warnings'])
            }
        }

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_json_path}")

    print("\n" + "=" * 70)

    return fields


def batch_process_invoices(image_paths: list, output_dir: str = "output"):
    """
    Process multiple invoices in batch.

    Args:
        image_paths: List of invoice image paths
        output_dir: Directory to save JSON outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"\n🔄 Processing {len(image_paths)} invoices...\n")

    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")

        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_json = os.path.join(output_dir, f"{base_name}_extracted.json")

            # Process invoice
            fields = process_invoice(image_path, output_json)
            results.append({
                'image': image_path,
                'success': True,
                'invoice_number': fields['invoice_number']['value'],
                'total': fields['amounts']['total']['value'],
                'confidence': fields['metadata']['extraction_confidence']
            })

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                'image': image_path,
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r['success'])
    print(f"\n✓ Successfully processed: {successful}/{len(image_paths)}")
    print(f"✗ Failed: {len(image_paths) - successful}/{len(image_paths)}")

    print("\nResults:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {os.path.basename(result['image'])}")
        if result['success']:
            print(
                f"      Invoice: {result['invoice_number']}, Total: {result['total']}, Confidence: {result['confidence']:.1%}")
        else:
            print(f"      Error: {result['error']}")

    return results


if __name__ == "__main__":
    image_path = "D:/Dying/4th year/seventh term/GP/ocr part/tabali.jpg"
    process_invoice(image_path, "/invoice_extracted.json")
