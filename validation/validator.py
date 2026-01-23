"""
Validator Module
Validates extracted invoice data for correctness, completeness, and consistency.
Performs business rule checks and data quality validation.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """Result of a validation check"""
    field: str
    is_valid: bool
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggested_fix: Optional[str] = None


class InvoiceValidator:
    """
    Validates extracted invoice data for quality and correctness.
    Checks for required fields, data formats, and business rules.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize validator with optional configuration.

        Args:
            config: Optional configuration dict with validation rules
        """
        self.config = config or self._default_config()
        self.validation_results = []

    def _default_config(self) -> Dict:
        """Default validation configuration"""
        return {
            'required_fields': [
                'invoice_number',
                'dates.invoice_date',
                'amounts.total',
                'currency'
            ],
            'optional_fields': [
                'dates.due_date',
                'amounts.subtotal',
                'amounts.tax',
                'vendor_info.name',
                'customer_info.name'
            ],
            'date_range': {
                'min_year': 2000,
                'max_year': 2030,
                'max_future_days': 90,  # How far in future is acceptable
                'max_past_years': 10  # How far in past is acceptable
            },
            'amount_limits': {
                'min_total': 0.01,
                'max_total': 10000000,  # 10 million
                'max_tax_percentage': 30.0
            },
            'tax_tolerance': 0.02,  # 2% tolerance for tax calculations
            'confidence_thresholds': {
                'minimum': 0.5,  # Below this is error
                'warning': 0.7,  # Below this is warning
                'good': 0.85  # Above this is good
            }
        }

    def validate_all(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform all validations on extracted invoice data.

        Args:
            extracted_data: Extracted invoice data from FieldExtractor

        Returns:
            Validation report with results and overall status
        """
        self.validation_results = []

        # 1. Required fields validation
        self._validate_required_fields(extracted_data)

        # 2. Invoice number validation
        self._validate_invoice_number(extracted_data.get('invoice_number', {}))

        # 3. Date validation
        self._validate_dates(extracted_data.get('dates', {}))

        # 4. Amount validation
        self._validate_amounts(extracted_data.get('amounts', {}))

        # 5. Tax validation
        self._validate_tax(
            extracted_data.get('amounts', {}),
            extracted_data.get('tax_info', {})
        )

        # 6. Currency validation
        self._validate_currency(extracted_data.get('currency', {}))

        # 7. Vendor/Customer validation
        self._validate_parties(
            extracted_data.get('vendor_info', {}),
            extracted_data.get('customer_info', {})
        )

        # 8. Line items validation
        self._validate_line_items(
            extracted_data.get('line_items', []),
            extracted_data.get('amounts', {})
        )

        # 9. Confidence scores validation
        self._validate_confidence_scores(extracted_data)

        # 10. Cross-field consistency checks
        self._validate_consistency(extracted_data)

        # Generate validation report
        report = self._generate_report(extracted_data)

        return report

    def _validate_required_fields(self, data: Dict) -> None:
        """Check that all required fields are present and have values"""
        for field_path in self.config['required_fields']:
            value = self._get_nested_value(data, field_path)

            if value is None or value == '' or value == {}:
                self.validation_results.append(ValidationResult(
                    field=field_path,
                    is_valid=False,
                    severity='error',
                    message=f"Required field '{field_path}' is missing or empty",
                    suggested_fix="Check OCR quality and field extraction logic"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    field=field_path,
                    is_valid=True,
                    severity='info',
                    message=f"Required field '{field_path}' is present"
                ))

    def _validate_invoice_number(self, invoice_number: Dict) -> None:
        """Validate invoice number format and presence"""
        value = invoice_number.get('value')
        confidence = invoice_number.get('confidence', 0)

        if not value:
            self.validation_results.append(ValidationResult(
                field='invoice_number',
                is_valid=False,
                severity='error',
                message="Invoice number is missing",
                suggested_fix="Check header zone extraction"
            ))
            return

        # Check length (should be at least 3 characters)
        if len(str(value)) < 3:
            self.validation_results.append(ValidationResult(
                field='invoice_number',
                is_valid=False,
                severity='warning',
                message=f"Invoice number '{value}' seems too short",
                suggested_fix="Verify this is the complete invoice number"
            ))

        # Check confidence
        if confidence < self.config['confidence_thresholds']['warning']:
            self.validation_results.append(ValidationResult(
                field='invoice_number',
                is_valid=True,
                severity='warning',
                message=f"Low confidence ({confidence:.1%}) for invoice number",
                suggested_fix="Manual verification recommended"
            ))
        else:
            self.validation_results.append(ValidationResult(
                field='invoice_number',
                is_valid=True,
                severity='info',
                message=f"Invoice number validated: {value}"
            ))

    def _validate_dates(self, dates: Dict) -> None:
        """Validate date fields"""
        invoice_date = dates.get('invoice_date', {})
        due_date = dates.get('due_date', {})

        # Validate invoice date
        inv_date_value = invoice_date.get('value')
        if inv_date_value:
            is_valid, message = self._validate_date_format(inv_date_value)
            if not is_valid:
                self.validation_results.append(ValidationResult(
                    field='dates.invoice_date',
                    is_valid=False,
                    severity='error',
                    message=message,
                    suggested_fix="Check date extraction patterns"
                ))
            else:
                # Check if date is within reasonable range
                try:
                    date_obj = datetime.strptime(inv_date_value, '%Y-%m-%d')

                    # Check year range
                    if date_obj.year < self.config['date_range']['min_year']:
                        self.validation_results.append(ValidationResult(
                            field='dates.invoice_date',
                            is_valid=False,
                            severity='warning',
                            message=f"Invoice date {inv_date_value} is very old",
                            suggested_fix="Verify the year was extracted correctly"
                        ))
                    elif date_obj.year > self.config['date_range']['max_year']:
                        self.validation_results.append(ValidationResult(
                            field='dates.invoice_date',
                            is_valid=False,
                            severity='error',
                            message=f"Invoice date {inv_date_value} is in far future",
                            suggested_fix="Check for date extraction errors"
                        ))

                    # Check if too far in future
                    max_future = datetime.now() + timedelta(days=self.config['date_range']['max_future_days'])
                    if date_obj > max_future:
                        self.validation_results.append(ValidationResult(
                            field='dates.invoice_date',
                            is_valid=False,
                            severity='warning',
                            message=f"Invoice date {inv_date_value} is far in future",
                            suggested_fix="Verify this is correct"
                        ))

                    # Check if too far in past
                    max_past = datetime.now() - timedelta(days=self.config['date_range']['max_past_years'] * 365)
                    if date_obj < max_past:
                        self.validation_results.append(ValidationResult(
                            field='dates.invoice_date',
                            is_valid=False,
                            severity='warning',
                            message=f"Invoice date {inv_date_value} is very old",
                            suggested_fix="Verify this is a current invoice"
                        ))

                    # If all checks pass
                    if all(r.is_valid for r in self.validation_results if r.field == 'dates.invoice_date'):
                        self.validation_results.append(ValidationResult(
                            field='dates.invoice_date',
                            is_valid=True,
                            severity='info',
                            message=f"Invoice date validated: {inv_date_value}"
                        ))

                except ValueError:
                    pass  # Already caught by format validation

        # Validate due date if present
        due_date_value = due_date.get('value')
        if due_date_value:
            is_valid, message = self._validate_date_format(due_date_value)
            if not is_valid:
                self.validation_results.append(ValidationResult(
                    field='dates.due_date',
                    is_valid=False,
                    severity='warning',
                    message=message,
                    suggested_fix="Check date extraction patterns"
                ))
            else:
                # Check due date is after invoice date
                if inv_date_value:
                    try:
                        inv_dt = datetime.strptime(inv_date_value, '%Y-%m-%d')
                        due_dt = datetime.strptime(due_date_value, '%Y-%m-%d')

                        if due_dt < inv_dt:
                            self.validation_results.append(ValidationResult(
                                field='dates.due_date',
                                is_valid=False,
                                severity='error',
                                message=f"Due date ({due_date_value}) is before invoice date ({inv_date_value})",
                                suggested_fix="Check which date is which"
                            ))
                        else:
                            self.validation_results.append(ValidationResult(
                                field='dates.due_date',
                                is_valid=True,
                                severity='info',
                                message=f"Due date validated: {due_date_value}"
                            ))
                    except ValueError:
                        pass

    def _validate_amounts(self, amounts: Dict) -> None:
        """Validate monetary amounts"""
        total = amounts.get('total', {}).get('value')
        subtotal = amounts.get('subtotal', {}).get('value')
        tax = amounts.get('tax', {}).get('value')
        discount = amounts.get('discount', {}).get('value')

        # Validate total amount
        if total is not None:
            if total < self.config['amount_limits']['min_total']:
                self.validation_results.append(ValidationResult(
                    field='amounts.total',
                    is_valid=False,
                    severity='error',
                    message=f"Total amount {total} is too small (< {self.config['amount_limits']['min_total']})",
                    suggested_fix="Check amount extraction"
                ))
            elif total > self.config['amount_limits']['max_total']:
                self.validation_results.append(ValidationResult(
                    field='amounts.total',
                    is_valid=False,
                    severity='warning',
                    message=f"Total amount {total} is very large (> {self.config['amount_limits']['max_total']})",
                    suggested_fix="Verify this amount is correct"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    field='amounts.total',
                    is_valid=True,
                    severity='info',
                    message=f"Total amount validated: {total}"
                ))

        # Validate subtotal is less than total
        if subtotal is not None and total is not None:
            if subtotal > total:
                self.validation_results.append(ValidationResult(
                    field='amounts.subtotal',
                    is_valid=False,
                    severity='error',
                    message=f"Subtotal ({subtotal}) is greater than total ({total})",
                    suggested_fix="Check if amounts were extracted correctly"
                ))

        # Validate tax is reasonable
        if tax is not None:
            if tax < 0:
                self.validation_results.append(ValidationResult(
                    field='amounts.tax',
                    is_valid=False,
                    severity='error',
                    message=f"Tax amount {tax} is negative",
                    suggested_fix="Check tax extraction"
                ))

            # Check tax percentage is reasonable
            if subtotal and subtotal > 0:
                tax_percentage = (tax / subtotal) * 100
                if tax_percentage > self.config['amount_limits']['max_tax_percentage']:
                    self.validation_results.append(ValidationResult(
                        field='amounts.tax',
                        is_valid=False,
                        severity='warning',
                        message=f"Tax rate ({tax_percentage:.1f}%) seems too high",
                        suggested_fix="Verify tax amount is correct"
                    ))

        # Validate discount if present
        if discount is not None:
            if discount < 0:
                self.validation_results.append(ValidationResult(
                    field='amounts.discount',
                    is_valid=False,
                    severity='warning',
                    message=f"Discount amount {discount} is negative",
                    suggested_fix="Check if this is meant to be positive"
                ))

    def _validate_tax(self, amounts: Dict, tax_info: Dict) -> None:
        """Validate tax calculations and consistency"""
        total = amounts.get('total', {}).get('value')
        subtotal = amounts.get('subtotal', {}).get('value')
        tax = amounts.get('tax', {}).get('value')
        tax_percentage = tax_info.get('tax_percentage', {}).get('value')

        # Check if subtotal + tax = total (with tolerance)
        if all(v is not None for v in [total, subtotal, tax]):
            calculated_total = subtotal + tax
            difference = abs(total - calculated_total)
            tolerance = total * self.config['tax_tolerance']

            if difference > tolerance:
                self.validation_results.append(ValidationResult(
                    field='amounts',
                    is_valid=False,
                    severity='warning',
                    message=f"Amounts don't add up: {subtotal} + {tax} ≠ {total} (diff: {difference:.2f})",
                    suggested_fix=f"Expected total: {calculated_total:.2f}"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    field='amounts',
                    is_valid=True,
                    severity='info',
                    message="Amount calculations are consistent"
                ))

        # Validate tax percentage against calculated tax
        if tax_percentage and subtotal and tax:
            expected_tax = subtotal * (tax_percentage / 100)
            tax_diff = abs(tax - expected_tax)

            if tax_diff > (expected_tax * self.config['tax_tolerance']):
                self.validation_results.append(ValidationResult(
                    field='tax_info.tax_percentage',
                    is_valid=False,
                    severity='warning',
                    message=f"Tax percentage ({tax_percentage}%) doesn't match tax amount",
                    suggested_fix=f"Expected tax: {expected_tax:.2f}, got: {tax}"
                ))

    def _validate_currency(self, currency: Dict) -> None:
        """Validate currency code"""
        value = currency.get('value')

        if not value:
            self.validation_results.append(ValidationResult(
                field='currency',
                is_valid=False,
                severity='warning',
                message="Currency code is missing",
                suggested_fix="Defaulting to USD"
            ))
            return

        # List of common currency codes
        valid_currencies = [
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'INR', 'CAD', 'AUD',
            'SAR', 'AED', 'QAR', 'KWD', 'OMR', 'BHD', 'JOD',
            'EGP', 'LBP', 'SYP', 'IQD', 'TRY'
        ]

        if value not in valid_currencies:
            self.validation_results.append(ValidationResult(
                field='currency',
                is_valid=False,
                severity='warning',
                message=f"Unusual currency code: {value}",
                suggested_fix="Verify currency is correct"
            ))
        else:
            self.validation_results.append(ValidationResult(
                field='currency',
                is_valid=True,
                severity='info',
                message=f"Currency validated: {value}"
            ))

    def _validate_parties(self, vendor: Dict, customer: Dict) -> None:
        """Validate vendor and customer information"""
        vendor_name = vendor.get('name', {}).get('value')
        customer_name = customer.get('name', {}).get('value')

        # Check vendor has at least a name
        if not vendor_name:
            self.validation_results.append(ValidationResult(
                field='vendor_info.name',
                is_valid=False,
                severity='warning',
                message="Vendor name is missing",
                suggested_fix="Check header/vendor zone extraction"
            ))

        # Validate email format if present
        vendor_email = vendor.get('email', {}).get('value')
        if vendor_email:
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Z|a-z]{2,}$')
            if not email_pattern.match(vendor_email):
                self.validation_results.append(ValidationResult(
                    field='vendor_info.email',
                    is_valid=False,
                    severity='warning',
                    message=f"Email format looks invalid: {vendor_email}",
                    suggested_fix="Verify email extraction"
                ))

        # Validate phone format if present
        vendor_phone = vendor.get('phone', {}).get('value')
        if vendor_phone:
            # Basic check: should have at least 7 digits
            digits = re.sub(r'\D', '', vendor_phone)
            if len(digits) < 7:
                self.validation_results.append(ValidationResult(
                    field='vendor_info.phone',
                    is_valid=False,
                    severity='warning',
                    message=f"Phone number seems incomplete: {vendor_phone}",
                    suggested_fix="Verify phone extraction"
                ))

    def _validate_line_items(self, line_items: List[Dict], amounts: Dict) -> None:
        """Validate line items"""
        if not line_items:
            self.validation_results.append(ValidationResult(
                field='line_items',
                is_valid=False,
                severity='warning',
                message="No line items found",
                suggested_fix="Check if invoice has a table structure"
            ))
            return

        # Check if line items total matches subtotal/total
        total_from_items = sum(
            item.get('amount', 0) or 0
            for item in line_items
            if item.get('amount')
        )

        subtotal = amounts.get('subtotal', {}).get('value')
        total = amounts.get('total', {}).get('value')

        if total_from_items > 0:
            # Compare with subtotal first
            if subtotal:
                diff = abs(total_from_items - subtotal)
                if diff > (subtotal * 0.05):  # 5% tolerance
                    self.validation_results.append(ValidationResult(
                        field='line_items',
                        is_valid=False,
                        severity='warning',
                        message=f"Line items total ({total_from_items:.2f}) doesn't match subtotal ({subtotal:.2f})",
                        suggested_fix="Check line item extraction or calculations"
                    ))
            # Or compare with total
            elif total:
                diff = abs(total_from_items - total)
                if diff > (total * 0.05):
                    self.validation_results.append(ValidationResult(
                        field='line_items',
                        is_valid=False,
                        severity='info',
                        message=f"Line items total ({total_from_items:.2f}) differs from invoice total ({total:.2f})",
                        suggested_fix="This is normal if tax/discounts are applied"
                    ))

        self.validation_results.append(ValidationResult(
            field='line_items',
            is_valid=True,
            severity='info',
            message=f"Found {len(line_items)} line items"
        ))

    def _validate_confidence_scores(self, data: Dict) -> None:
        """Validate OCR confidence scores"""
        overall_confidence = data.get('metadata', {}).get('extraction_confidence', 0)

        if overall_confidence < self.config['confidence_thresholds']['minimum']:
            self.validation_results.append(ValidationResult(
                field='metadata.confidence',
                is_valid=False,
                severity='error',
                message=f"Very low extraction confidence ({overall_confidence:.1%})",
                suggested_fix="Image quality is poor, manual review required"
            ))
        elif overall_confidence < self.config['confidence_thresholds']['warning']:
            self.validation_results.append(ValidationResult(
                field='metadata.confidence',
                is_valid=True,
                severity='warning',
                message=f"Low extraction confidence ({overall_confidence:.1%})",
                suggested_fix="Manual verification recommended"
            ))
        elif overall_confidence < self.config['confidence_thresholds']['good']:
            self.validation_results.append(ValidationResult(
                field='metadata.confidence',
                is_valid=True,
                severity='info',
                message=f"Moderate extraction confidence ({overall_confidence:.1%})"
            ))
        else:
            self.validation_results.append(ValidationResult(
                field='metadata.confidence',
                is_valid=True,
                severity='info',
                message=f"Good extraction confidence ({overall_confidence:.1%})"
            ))

    def _validate_consistency(self, data: Dict) -> None:
        """Cross-field consistency checks"""
        # Check if we have both vendor and customer but they're the same
        vendor_name = data.get('vendor_info', {}).get('name', {}).get('value')
        customer_name = data.get('customer_info', {}).get('name', {}).get('value')

        if vendor_name and customer_name and vendor_name == customer_name:
            self.validation_results.append(ValidationResult(
                field='consistency',
                is_valid=False,
                severity='warning',
                message="Vendor and customer names are identical",
                suggested_fix="Check if parties were extracted correctly"
            ))

    def _validate_date_format(self, date_str: str) -> Tuple[bool, str]:
        """Validate date format (YYYY-MM-DD)"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True, "Valid date format"
        except ValueError:
            return False, f"Invalid date format: {date_str} (expected YYYY-MM-DD)"

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dict using dot notation"""
        keys = path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

            if value is None:
                return None

        # Handle the case where value is a dict with 'value' key
        if isinstance(value, dict) and 'value' in value:
            return value['value']

        return value

    def _generate_report(self, data: Dict) -> Dict[str, Any]:
        """Generate validation report"""
        # Count results by severity
        errors = [r for r in self.validation_results if r.severity == 'error' and not r.is_valid]
        warnings = [r for r in self.validation_results if r.severity == 'warning' and not r.is_valid]
        infos = [r for r in self.validation_results if r.severity == 'info']

        # Determine overall status
        if errors:
            overall_status = 'FAILED'
        elif warnings:
            overall_status = 'WARNING'
        else:
            overall_status = 'PASSED'

        # Calculate data quality score
        total_checks = len(self.validation_results)
        passed_checks = len([r for r in self.validation_results if r.is_valid])
        quality_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0

        report = {
            'overall_status': overall_status,
            'quality_score': quality_score,
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'failed': total_checks - passed_checks,
                'errors': len(errors),
                'warnings': len(warnings),
                'info': len(infos)
            },
            'errors': [self._format_result(r) for r in errors],
            'warnings': [self._format_result(r) for r in warnings],
            'info': [self._format_result(r) for r in infos],
            'all_results': [self._format_result(r) for r in self.validation_results],
            'requires_manual_review': len(errors) > 0 or len(warnings) > 2,
            'recommended_actions': self._get_recommended_actions(errors, warnings)
        }

        return report

    def _format_result(self, result: ValidationResult) -> Dict:
        """Format validation result as dict"""
        return {
            'field': result.field,
            'is_valid': result.is_valid,
            'severity': result.severity,
            'message': result.message,
            'suggested_fix': result.suggested_fix
        }

    def _get_recommended_actions(self, errors: List, warnings: List) -> List[str]:
        """Generate recommended actions based on validation results"""
        actions = []

        if errors:
            actions.append("    Critical errors found - manual review required")
            actions.append(f"   Fix {len(errors)} error(s) before processing")

        if warnings:
            actions.append(f" {len(warnings)} warning(s) found - verify accuracy")

        if not errors and not warnings:
            actions.append("   All validation checks passed")
            actions.append("   Invoice data appears to be accurate")

        return actions


# Example usage
if __name__ == "__main__":
    # Mock extracted data for testing
    mock_data = {
        'invoice_number': {'value': 'INV-2024-001', 'confidence': 0.95},
        'dates': {
            'invoice_date': {'value': '2024-01-15', 'confidence': 0.92},
            'due_date': {'value': '2024-02-15', 'confidence': 0.88}
        },
        'amounts': {
            'subtotal': {'value': 1000.00, 'confidence': 0.96},
            'tax': {'value': 150.00, 'confidence': 0.94},
            'total': {'value': 1150.00, 'confidence': 0.98},
            'discount': {'value': None, 'confidence': 0.0}
        },
        'currency': {'value': 'SAR', 'confidence': 0.85},
        'tax_info': {
            'tax_id': {'value': 'TAX123456789', 'confidence': 0.90},
            'tax_percentage': {'value': 15.0, 'confidence': 0.92}
        },
        'vendor_info': {
            'name': {'value': 'ABC Company', 'confidence': 0.91},
            'email': {'value': 'info@abc.com', 'confidence': 0.87},
            'phone': {'value': '+966501234567', 'confidence': 0.89}
        },
        'customer_info': {
            'name': {'value': 'XYZ Corp', 'confidence': 0.86}
        },
        'line_items': [
            {'description': 'Product A', 'quantity': 10, 'unit_price': 50.0, 'amount': 500.0},
            {'description': 'Product B', 'quantity': 5, 'unit_price': 100.0, 'amount': 500.0}
        ],
        'payment_terms': {'value': 'Net 30', 'confidence': 0.80},
        'metadata': {
            'extraction_confidence': 0.91,
            'fields_extracted': 8,
            'total_fields': 9
        }
    }

    # Validate
    validator = InvoiceValidator()
    report = validator.validate_all(mock_data)

    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    print(f"\nOverall Status: {report['overall_status']}")
    print(f"Quality Score: {report['quality_score']:.1f}%")

    print(f"\nSummary:")
    print(f"  Total Checks: {report['summary']['total_checks']}")
    print(f"  Passed: {report['summary']['passed']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Errors: {report['summary']['errors']}")
    print(f"  Warnings: {report['summary']['warnings']}")

    if report['errors']:
        print(f"\n ERRORS ({len(report['errors'])}):")
        for error in report['errors']:
            print(f"  • {error['field']}: {error['message']}")
            if error['suggested_fix']:
                print(f"    Fix: {error['suggested_fix']}")

    if report['warnings']:
        print(f"\n WARNINGS ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"  • {warning['field']}: {warning['message']}")
            if warning['suggested_fix']:
                print(f"    Fix: {warning['suggested_fix']}")

    print(f"\n Recommended Actions:")
    for action in report['recommended_actions']:
        print(f"  {action}")

    print(f"\nRequires Manual Review: {'Yes' if report['requires_manual_review'] else 'No'}")