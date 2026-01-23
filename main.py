"""
Main Pipeline Module
Production-ready orchestrator for invoice processing.
Handles the complete workflow from image to validated data with error handling,
logging, retry logic, and monitoring.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from ocr.invoice_ocr import InvoiceOCR
from extraction.layout_analyzer import LayoutAnalyzer
from extraction.field_extractor import FieldExtractor
from validation.validator import InvoiceValidator
from preprocessing.reciept_preprocessing import preprocess_receipt_conservative
import cv2


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('invoice_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Invoice processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    VALIDATION_WARNING = "validation_warning"
    VALIDATION_FAILED = "validation_failed"
    ERROR = "error"


@dataclass
class ProcessingResult:
    """Result of invoice processing"""
    invoice_id: str
    status: ProcessingStatus
    image_path: str
    output_path: Optional[str]
    extracted_data: Optional[Dict]
    validation_report: Optional[Dict]
    error_message: Optional[str]
    processing_time: float
    timestamp: str

    def to_dict(self):
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        return result


class InvoicePipeline:
    """
    Main pipeline for invoice processing.
    Orchestrates the complete workflow with error handling and validation.
    """

    def __init__(
        self,
        output_dir: str = "output",
        use_gpu: bool = True,
        enable_preprocessing: bool = True,
        validation_config: Optional[Dict] = None,
        save_intermediate: bool = False,
        max_retries: int = 2
    ):
        """
        Initialize the invoice processing pipeline.

        Args:
            output_dir: Directory to save processed invoices
            use_gpu: Use GPU acceleration for OCR
            enable_preprocessing: Apply image preprocessing
            validation_config: Custom validation configuration
            save_intermediate: Save intermediate processing results
            max_retries: Maximum number of retry attempts on failure
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_preprocessing = enable_preprocessing
        self.save_intermediate = save_intermediate
        self.max_retries = max_retries

        # Initialize components
        logger.info("Initializing invoice processing pipeline...")

        try:
            self.ocr = InvoiceOCR(use_gpu=use_gpu)
            self.layout_analyzer = LayoutAnalyzer()
            self.field_extractor = FieldExtractor()
            self.validator = InvoiceValidator(config=validation_config)

            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'validation_warnings': 0,
            'validation_failed': 0,
            'errors': 0,
            'total_processing_time': 0.0
        }

    def process_invoice(
        self,
        image_path: str,
        invoice_id: Optional[str] = None,
        retry_count: int = 0
    ) -> ProcessingResult:
        """
        Process a single invoice through the complete pipeline.

        Args:
            image_path: Path to invoice image
            invoice_id: Optional unique identifier for the invoice
            retry_count: Current retry attempt number

        Returns:
            ProcessingResult with status and extracted data
        """
        start_time = time.time()

        # Generate invoice ID if not provided
        if invoice_id is None:
            invoice_id = self._generate_invoice_id(image_path)

        logger.info(f"[{invoice_id}] Starting invoice processing: {image_path}")

        # Initialize result
        result = ProcessingResult(
            invoice_id=invoice_id,
            status=ProcessingStatus.PROCESSING,
            image_path=image_path,
            output_path=None,
            extracted_data=None,
            validation_report=None,
            error_message=None,
            processing_time=0.0,
            timestamp=datetime.now().isoformat()
        )

        try:
            # Step 1: OCR Extraction
            logger.info(f"[{invoice_id}] Step 1/5: OCR extraction")
            ocr_results = self._run_ocr(image_path, invoice_id)

            if not ocr_results:
                raise ValueError("OCR extraction returned no results")

            logger.info(f"[{invoice_id}] OCR completed: {len(ocr_results)} elements extracted")

            # Step 2: Layout Analysis
            logger.info(f"[{invoice_id}] Step 2/5: Layout analysis")
            layout = self._analyze_layout(ocr_results, invoice_id)

            logger.info(f"[{invoice_id}] Layout analysis completed: "
                       f"{len(layout['zones'])} zones, "
                       f"{len(layout['key_value_pairs'])} key-value pairs, "
                       f"{len(layout['tables'])} tables")

            # Step 3: Field Extraction
            logger.info(f"[{invoice_id}] Step 3/5: Field extraction")
            extracted_data = self._extract_fields(ocr_results, layout, invoice_id)

            fields_extracted = extracted_data.get('metadata', {}).get('fields_extracted', 0)
            logger.info(f"[{invoice_id}] Field extraction completed: "
                       f"{fields_extracted} fields extracted")

            # Step 4: Validation
            logger.info(f"[{invoice_id}] Step 4/5: Validation")
            validation_report = self._validate_data(extracted_data, invoice_id)

            logger.info(f"[{invoice_id}] Validation completed: "
                       f"Status={validation_report['overall_status']}, "
                       f"Quality={validation_report['quality_score']:.1f}%")

            # Step 5: Export Results
            logger.info(f"[{invoice_id}] Step 5/5: Exporting results")
            output_path = self._export_results(
                invoice_id,
                extracted_data,
                validation_report,
                ocr_results if self.save_intermediate else None,
                layout if self.save_intermediate else None
            )

            # Determine final status based on validation
            if validation_report['overall_status'] == 'PASSED':
                status = ProcessingStatus.SUCCESS
                self.stats['successful'] += 1
            elif validation_report['overall_status'] == 'WARNING':
                status = ProcessingStatus.VALIDATION_WARNING
                self.stats['validation_warnings'] += 1
            else:
                status = ProcessingStatus.VALIDATION_FAILED
                self.stats['validation_failed'] += 1

            # Update result
            result.status = status
            result.extracted_data = extracted_data
            result.validation_report = validation_report
            result.output_path = str(output_path)

            logger.info(f"[{invoice_id}] Processing completed successfully: {status.value}")

        except Exception as e:
            logger.error(f"[{invoice_id}] Processing failed: {str(e)}", exc_info=True)

            # Retry logic
            if retry_count < self.max_retries:
                logger.info(f"[{invoice_id}] Retrying... (attempt {retry_count + 1}/{self.max_retries})")
                return self.process_invoice(image_path, invoice_id, retry_count + 1)

            # Max retries exceeded
            result.status = ProcessingStatus.ERROR
            result.error_message = str(e)
            self.stats['errors'] += 1

        finally:
            # Calculate processing time
            result.processing_time = time.time() - start_time
            self.stats['total_processed'] += 1
            self.stats['total_processing_time'] += result.processing_time

            logger.info(f"[{invoice_id}] Processing time: {result.processing_time:.2f}s")

        return result

    def process_batch(
        self,
        image_paths: List[str],
        output_batch_report: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple invoices in batch.

        Args:
            image_paths: List of paths to invoice images
            output_batch_report: Generate batch processing report

        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"Starting batch processing: {len(image_paths)} invoices")

        results = []

        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing invoice {i}/{len(image_paths)}")
            logger.info(f"{'='*70}")

            try:
                result = self.process_invoice(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                # Create error result
                results.append(ProcessingResult(
                    invoice_id=self._generate_invoice_id(image_path),
                    status=ProcessingStatus.ERROR,
                    image_path=image_path,
                    output_path=None,
                    extracted_data=None,
                    validation_report=None,
                    error_message=str(e),
                    processing_time=0.0,
                    timestamp=datetime.now().isoformat()
                ))

        # Generate batch report
        if output_batch_report:
            self._generate_batch_report(results)

        logger.info(f"\nBatch processing completed: {len(image_paths)} invoices")
        self._print_statistics()

        return results

    def _run_ocr(self, image_path: str, invoice_id: str) -> List[Dict]:
        """Run OCR extraction with error handling"""
        try:
            if self.enable_preprocessing:
                # Use conservative receipt preprocessing (Test 4 configuration)
                logger.info(f"[{invoice_id}] Applying conservative receipt preprocessing...")
                preprocessed_img = preprocess_receipt_conservative(image_path)

                if preprocessed_img is not None:
                    # Save preprocessed image temporarily
                    import tempfile
                    import os
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, f"{invoice_id}_preprocessed.jpg")
                    cv2.imwrite(temp_path, preprocessed_img)

                    # Run OCR on preprocessed image
                    ocr_results = self.ocr.extract_text_with_boxes(
                        temp_path,
                        use_preprocessing=False  # Already preprocessed
                    )

                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass

                    return ocr_results
                else:
                    logger.warning(f"[{invoice_id}] Preprocessing failed, using original image")

            # Fallback: use original image with built-in preprocessing
            ocr_results = self.ocr.extract_text_with_boxes(
                image_path,
                use_preprocessing=False  # No preprocessing
            )
            return ocr_results

        except Exception as e:
            logger.error(f"[{invoice_id}] OCR extraction failed: {e}")
            raise

    def _analyze_layout(self, ocr_results: List[Dict], invoice_id: str) -> Dict:
        """Run layout analysis with error handling"""
        try:
            layout = self.layout_analyzer.analyze(ocr_results)
            return layout
        except Exception as e:
            logger.error(f"[{invoice_id}] Layout analysis failed: {e}")
            raise

    def _extract_fields(
        self,
        ocr_results: List[Dict],
        layout: Dict,
        invoice_id: str
    ) -> Dict:
        """Run field extraction with error handling"""
        try:
            extracted_data = self.field_extractor.extract_all_fields(ocr_results, layout)
            return extracted_data
        except Exception as e:
            logger.error(f"[{invoice_id}] Field extraction failed: {e}")
            raise

    def _validate_data(self, extracted_data: Dict, invoice_id: str) -> Dict:
        """Run validation with error handling"""
        try:
            validation_report = self.validator.validate_all(extracted_data)
            return validation_report
        except Exception as e:
            logger.error(f"[{invoice_id}] Validation failed: {e}")
            raise

    def _export_results(
        self,
        invoice_id: str,
        extracted_data: Dict,
        validation_report: Dict,
        ocr_results: Optional[List[Dict]] = None,
        layout: Optional[Dict] = None
    ) -> Path:
        """Export processing results to JSON"""
        try:
            # Create invoice-specific output directory
            invoice_dir = self.output_dir / invoice_id
            invoice_dir.mkdir(parents=True, exist_ok=True)

            # Main results file
            output_data = {
                'invoice_id': invoice_id,
                'timestamp': datetime.now().isoformat(),
                'extracted_data': self._clean_for_json(extracted_data),
                'validation': validation_report
            }

            output_path = invoice_dir / f"{invoice_id}_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            # Save intermediate results if enabled
            if self.save_intermediate:
                if ocr_results:
                    ocr_path = invoice_dir / f"{invoice_id}_ocr.json"
                    with open(ocr_path, 'w', encoding='utf-8') as f:
                        json.dump(self._clean_for_json(ocr_results), f, indent=2, ensure_ascii=False)

                if layout:
                    layout_path = invoice_dir / f"{invoice_id}_layout.json"
                    with open(layout_path, 'w', encoding='utf-8') as f:
                        json.dump(self._clean_for_json(layout), f, indent=2, ensure_ascii=False)

            logger.info(f"[{invoice_id}] Results exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"[{invoice_id}] Failed to export results: {e}")
            raise

    def _generate_batch_report(self, results: List[ProcessingResult]) -> None:
        """Generate batch processing report"""
        report_path = self.output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_invoices': len(results),
            'summary': {
                'successful': sum(1 for r in results if r.status == ProcessingStatus.SUCCESS),
                'warnings': sum(1 for r in results if r.status == ProcessingStatus.VALIDATION_WARNING),
                'validation_failed': sum(1 for r in results if r.status == ProcessingStatus.VALIDATION_FAILED),
                'errors': sum(1 for r in results if r.status == ProcessingStatus.ERROR)
            },
            'average_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0,
            'results': [r.to_dict() for r in results]
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Batch report saved to: {report_path}")

    def _generate_invoice_id(self, image_path: str) -> str:
        """Generate unique invoice ID from image path and timestamp"""
        filename = Path(image_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{filename}_{timestamp}"

    def _clean_for_json(self, data: Any) -> Any:
        """Clean data for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return list(data)
        elif hasattr(data, '__dict__'):
            return self._clean_for_json(data.__dict__)
        else:
            return data

    def _print_statistics(self) -> None:
        """Print processing statistics"""
        logger.info("\n" + "="*70)
        logger.info("PROCESSING STATISTICS")
        logger.info("="*70)
        logger.info(f"Total Processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']} "
                   f"({self.stats['successful']/self.stats['total_processed']*100:.1f}%)")
        logger.info(f"Warnings: {self.stats['validation_warnings']} "
                   f"({self.stats['validation_warnings']/self.stats['total_processed']*100:.1f}%)")
        logger.info(f"Validation Failed: {self.stats['validation_failed']} "
                   f"({self.stats['validation_failed']/self.stats['total_processed']*100:.1f}%)")
        logger.info(f"Errors: {self.stats['errors']} "
                   f"({self.stats['errors']/self.stats['total_processed']*100:.1f}%)")

        if self.stats['total_processed'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['total_processed']
            logger.info(f"Average Processing Time: {avg_time:.2f}s")
            logger.info(f"Total Processing Time: {self.stats['total_processing_time']:.2f}s")

        logger.info("="*70)

    def get_statistics(self) -> Dict:
        """Get current processing statistics"""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset processing statistics"""
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'validation_warnings': 0,
            'validation_failed': 0,
            'errors': 0,
            'total_processing_time': 0.0
        }
        logger.info("Statistics reset")


# Convenience function for quick processing
def process_invoice_quick(
    image_path: str,
    output_dir: str = "output",
    use_gpu: bool = True
) -> Tuple[Dict, Dict]:
    """
    Quick convenience function to process a single invoice.

    Args:
        image_path: Path to invoice image
        output_dir: Output directory
        use_gpu: Use GPU acceleration

    Returns:
        Tuple of (extracted_data, validation_report)
    """
    pipeline = InvoicePipeline(output_dir=output_dir, use_gpu=use_gpu)
    result = pipeline.process_invoice(image_path)

    if result.status == ProcessingStatus.ERROR:
        raise Exception(f"Processing failed: {result.error_message}")

    return result.extracted_data, result.validation_report


# Main execution
if __name__ == "__main__":


    # Parse arguments
    image_paths = ["/tabali.jpg"]
    output_dir = "output"
    use_gpu = True
    save_intermediate = False

    # Initialize pipeline
    print("\n" + "="*70)
    print("INVOICE PROCESSING PIPELINE")
    print("="*70)

    pipeline = InvoicePipeline(
        output_dir=output_dir,
        use_gpu=use_gpu,
        save_intermediate=save_intermediate
    )

    # Process invoices
    if len(image_paths) == 1:
        # Single invoice
        result = pipeline.process_invoice(image_paths[0])

        print("\n" + "="*70)
        print("PROCESSING RESULT")
        print("="*70)
        print(f"Status: {result.status.value}")
        print(f"Output: {result.output_path}")
        print(f"Processing Time: {result.processing_time:.2f}s")

        if result.validation_report:
            print(f"Quality Score: {result.validation_report['quality_score']:.1f}%")

        if result.error_message:
            print(f"Error: {result.error_message}")

