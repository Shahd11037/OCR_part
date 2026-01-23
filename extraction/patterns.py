"""
Patterns Module
Contains regex patterns and keywords for extracting invoice fields.
Supports both Arabic and English text.
"""

import re
from typing import Dict, List, Pattern
from dataclasses import dataclass


@dataclass
class FieldPattern:
    """Represents a pattern for extracting a specific field"""
    name: str
    patterns: List[Pattern]
    keywords_en: List[str]
    keywords_ar: List[str]
    zone_preference: str  # Preferred zone to search in


class InvoicePatterns:
    """
    Collection of regex patterns and keywords for invoice field extraction.
    Handles both Arabic and English text.
    """

    def __init__(self):
        # Compile all patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns"""

        # ============================================================
        # DATE PATTERNS
        # ============================================================
        self.date_patterns = {
            # Standard formats: YYYY-MM-DD, DD-MM-YYYY, MM/DD/YYYY
            'iso_date': re.compile(r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b'),
            'dmy_slash': re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'),
            'dmy_dash': re.compile(r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b'),
            'mdy_slash': re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'),

            # Written formats: Jan 15, 2024 / 15 Jan 2024
            'month_name': re.compile(
                r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b',
                re.IGNORECASE
            ),
            'name_month': re.compile(
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})\b',
                re.IGNORECASE
            ),

            # Arabic/Middle Eastern formats
            'arabic_date': re.compile(r'[\u0660-\u0669]{1,2}[-/][\u0660-\u0669]{1,2}[-/][\u0660-\u0669]{4}'),

            # Compact formats: 20240115
            'compact_date': re.compile(r'\b(20\d{2})(\d{2})(\d{2})\b'),
        }

        # ============================================================
        # NUMBER & AMOUNT PATTERNS
        # ============================================================
        self.number_patterns = {
            # Standard numbers with optional thousand separators and decimals
            'decimal_comma': re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2,}|\.\d{1,2})?\b'),
            'decimal_space': re.compile(r'\b\d{1,3}(?:\s\d{3})*(?:\.\d{2,}|\.\d{1,2})?\b'),
            'decimal_dot': re.compile(r'\b\d{1,3}(?:\.\d{3})*(?:,\d{2,}|,\d{1,2})?\b'),  # European format

            # Simple numbers
            'simple_number': re.compile(r'\b\d+(?:\.\d{1,2})?\b'),

            # Arabic-Indic digits (٠-٩)
            'arabic_number': re.compile(r'[\u0660-\u0669]+(?:\.[\u0660-\u0669]+)?'),

            # Numbers with currency symbols
            'with_currency': re.compile(
                r'(?:[$£€¥₹₪﷼])\s*(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)|'
                r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{2})?)\s*(?:[$£€¥₹₪﷼SAR|EGP|USD|AED|QAR|KWD|OMR|BHD|JOD|LBP|SYP|IQD|ريال|جنيه|دينار|درهم])',
                re.IGNORECASE
            ),
        }

        # ============================================================
        # INVOICE NUMBER PATTERNS
        # ============================================================
        self.invoice_number_patterns = {
            # INV-12345, Invoice #12345, INV/2024/001
            'standard': re.compile(
                r'\b(?:INV|INVOICE|INV#|NO|#|N°)[-/#\s]*(\d{4,})\b',
                re.IGNORECASE
            ),

            # More complex formats: INV-2024-001, A-2024-12345
            'with_year': re.compile(
                r'\b([A-Z]{2,4})[-/#\s]*(20\d{2})[-/#\s]*(\d{3,})\b',
                re.IGNORECASE
            ),

            # Simple sequential: 12345, 001234
            'sequential': re.compile(r'\b\d{5,8}\b'),

            # Arabic invoice number markers
            'arabic_marker': re.compile(
                r'(?:فاتورة|رقم|رقم الفاتورة)\s*[:#\s]*(\d+)',
                re.IGNORECASE
            ),
        }

        # ============================================================
        # TAX/VAT PATTERNS
        # ============================================================
        self.tax_patterns = {
            # Tax number: TAX123456789, VAT-GB123456789
            'tax_id': re.compile(
                r'\b(?:TAX|VAT|TIN|EIN|TRN)[-\s#:]*([A-Z0-9]{8,15})\b',
                re.IGNORECASE
            ),

            # Tax percentage: 15%, VAT 5%
            'tax_percentage': re.compile(
                r'(?:VAT|TAX|ضريبة)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*%',
                re.IGNORECASE
            ),

            # Arabic tax number
            'arabic_tax': re.compile(r'(?:الرقم الضريبي|رقم ضريبي)\s*[:]*\s*(\d+)'),
        }

        # ============================================================
        # PHONE NUMBER PATTERNS
        # ============================================================
        self.phone_patterns = {
            # International: +966 50 123 4567, +20-10-1234-5678
            'international': re.compile(r'\+\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{4}'),

            # Local formats: (050) 123-4567, 050-123-4567
            'local': re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),

            # Simple: 0501234567
            'simple': re.compile(r'\b0\d{9}\b'),
        }

        # ============================================================
        # EMAIL PATTERNS
        # ============================================================
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # ============================================================
        # CURRENCY PATTERNS
        # ============================================================
        self.currency_patterns = {
            'symbols': re.compile(r'[$£€¥₹₪﷼]'),
            'codes': re.compile(
                r'\b(SAR|EGP|USD|EUR|GBP|AED|QAR|KWD|OMR|BHD|JOD|LBP|SYP|IQD)\b',
                re.IGNORECASE
            ),
            'arabic': re.compile(r'(ريال|جنيه|دينار|درهم|ليرة)'),
        }

    # ============================================================
    # KEYWORD DEFINITIONS
    # ============================================================

    @property
    def invoice_number_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate invoice number"""
        return {
            'en': [
                'invoice number', 'invoice no', 'invoice #', 'inv no', 'inv#',
                'number', 'no.', 'reference', 'ref', 'document number', 'bill no'
            ],
            'ar': [
                'رقم الفاتورة', 'رقم فاتورة', 'فاتورة رقم', 'رقم', 'المرجع'
            ]
        }

    @property
    def date_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate dates"""
        return {
            'en': [
                'date', 'invoice date', 'issue date', 'issued', 'due date',
                'payment date', 'billing date', 'created', 'dated'
            ],
            'ar': [
                'تاريخ', 'تاريخ الفاتورة', 'تاريخ الإصدار', 'تاريخ الاستحقاق',
                'التاريخ', 'بتاريخ'
            ]
        }

    @property
    def total_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate total amount"""
        return {
            'en': [
                'total', 'grand total', 'total amount', 'amount due',
                'total due', 'balance due', 'net total', 'final total',
                'total payable', 'amount payable'
            ],
            'ar': [
                'المجموع', 'الإجمالي', 'المبلغ الإجمالي', 'الإجمالي الكلي',
                'المبلغ المستحق', 'المجموع الكلي', 'إجمالي المبلغ'
            ]
        }

    @property
    def subtotal_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate subtotal"""
        return {
            'en': [
                'subtotal', 'sub-total', 'sub total', 'amount before tax',
                'net amount', 'before tax'
            ],
            'ar': [
                'المجموع الفرعي', 'قبل الضريبة', 'المبلغ قبل الضريبة',
                'المجموع قبل الضريبة'
            ]
        }

    @property
    def tax_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate tax/VAT"""
        return {
            'en': [
                'tax', 'vat', 'sales tax', 'tax amount', 'vat amount',
                'value added tax', 'gst', 'taxation'
            ],
            'ar': [
                'ضريبة', 'الضريبة', 'ضريبة القيمة المضافة', 'قيمة الضريبة',
                'مبلغ الضريبة', 'ض.ق.م'
            ]
        }

    @property
    def discount_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate discount"""
        return {
            'en': [
                'discount', 'reduction', 'deduction', 'rebate', 'off',
                'discount amount', 'total discount'
            ],
            'ar': [
                'خصم', 'تخفيض', 'حسم', 'الخصم', 'قيمة الخصم'
            ]
        }

    @property
    def vendor_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate vendor/seller information"""
        return {
            'en': [
                'vendor', 'seller', 'from', 'supplier', 'company',
                'sold by', 'merchant', 'provider', 'business name'
            ],
            'ar': [
                'المورد', 'البائع', 'الشركة', 'من', 'مقدم الخدمة',
                'اسم الشركة', 'التاجر'
            ]
        }

    @property
    def customer_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate customer/buyer information"""
        return {
            'en': [
                'customer', 'buyer', 'to', 'bill to', 'billed to',
                'client', 'customer name', 'purchaser', 'sold to'
            ],
            'ar': [
                'العميل', 'المشتري', 'إلى', 'العميل', 'اسم العميل',
                'المستفيد', 'الزبون'
            ]
        }

    @property
    def quantity_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate quantity"""
        return {
            'en': [
                'quantity', 'qty', 'amount', 'units', 'pieces', 'pcs',
                'count', 'no. of items'
            ],
            'ar': [
                'الكمية', 'كمية', 'عدد', 'الوحدات', 'القطع'
            ]
        }

    @property
    def unit_price_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate unit price"""
        return {
            'en': [
                'unit price', 'price', 'rate', 'cost', 'price per unit',
                'unit cost', 'each'
            ],
            'ar': [
                'سعر الوحدة', 'السعر', 'التكلفة', 'سعر', 'سعر القطعة'
            ]
        }

    @property
    def description_keywords(self) -> Dict[str, List[str]]:
        """Keywords that indicate item description"""
        return {
            'en': [
                'description', 'item', 'product', 'service', 'details',
                'particulars', 'goods'
            ],
            'ar': [
                'الوصف', 'البند', 'المنتج', 'الخدمة', 'التفاصيل',
                'الصنف', 'السلعة', 'البيان'
            ]
        }

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def get_all_keywords(self, field_type: str) -> List[str]:
        """Get both English and Arabic keywords for a field type"""
        keyword_dict = getattr(self, f'{field_type}_keywords', {})
        return keyword_dict.get('en', []) + keyword_dict.get('ar', [])

    def find_currency(self, text: str) -> str:
        """Extract currency from text"""
        # Check for currency symbols
        symbol_match = self.currency_patterns['symbols'].search(text)
        if symbol_match:
            symbol_map = {
                '$': 'USD', '£': 'GBP', '€': 'EUR', '¥': 'JPY',
                '₹': 'INR', '₪': 'ILS', '﷼': 'SAR'
            }
            return symbol_map.get(symbol_match.group(), 'USD')

        # Check for currency codes
        code_match = self.currency_patterns['codes'].search(text)
        if code_match:
            return code_match.group(1).upper()

        # Check for Arabic currency names
        arabic_match = self.currency_patterns['arabic'].search(text)
        if arabic_match:
            arabic_map = {
                'ريال': 'SAR', 'جنيه': 'EGP', 'دينار': 'KWD',
                'درهم': 'AED', 'ليرة': 'LBP'
            }
            return arabic_map.get(arabic_match.group(1), 'SAR')

        return 'USD'  # Default fallback

    def normalize_arabic_numbers(self, text: str) -> str:
        """Convert Arabic-Indic digits (٠-٩) to Western digits (0-9)"""
        arabic_to_western = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        for arabic, western in arabic_to_western.items():
            text = text.replace(arabic, western)
        return text

    def clean_amount(self, amount_str: str) -> float:
        """Clean and convert amount string to float"""
        if not amount_str:
            return 0.0

        # Normalize Arabic numbers
        amount_str = self.normalize_arabic_numbers(amount_str)

        # Remove currency symbols and codes
        amount_str = re.sub(r'[$£€¥₹₪﷼]', '', amount_str)
        amount_str = re.sub(r'\b(?:SAR|EGP|USD|EUR|GBP|AED|QAR|KWD)\b', '', amount_str, flags=re.IGNORECASE)

        # Remove spaces
        amount_str = amount_str.strip().replace(' ', '')

        # Handle different decimal separators
        # If there are multiple separators, the last one is the decimal
        if ',' in amount_str and '.' in amount_str:
            # Determine which is decimal based on position
            last_comma_pos = amount_str.rfind(',')
            last_dot_pos = amount_str.rfind('.')

            if last_comma_pos > last_dot_pos:
                # Comma is decimal: 1.234,56 -> 1234.56
                amount_str = amount_str.replace('.', '').replace(',', '.')
            else:
                # Dot is decimal: 1,234.56 -> 1234.56
                amount_str = amount_str.replace(',', '')
        elif ',' in amount_str:
            # Only comma: could be thousand separator or decimal
            if amount_str.count(',') == 1 and len(amount_str.split(',')[1]) <= 2:
                # Likely decimal: 123,45 -> 123.45
                amount_str = amount_str.replace(',', '.')
            else:
                # Likely thousand separator: 1,234,567 -> 1234567
                amount_str = amount_str.replace(',', '')

        try:
            return float(amount_str)
        except ValueError:
            return 0.0


# Create a global instance for easy importing
patterns = InvoicePatterns()

# Example usage
if __name__ == "__main__":
    patterns = InvoicePatterns()

    print("=" * 60)
    print("INVOICE PATTERNS TEST")
    print("=" * 60)

    # Test date extraction
    test_dates = [
        "Date: 2024-01-15",
        "Invoice Date: 15/01/2024",
        "تاريخ: ١٥-٠١-٢٠٢٤",
        "Jan 15, 2024"
    ]

    print("\n1. DATE EXTRACTION:")
    for test_date in test_dates:
        for pattern_name, pattern in patterns.date_patterns.items():
            match = pattern.search(test_date)
            if match:
                print(f"   {test_date} → {match.group()} ({pattern_name})")
                break

    # Test amount extraction
    test_amounts = [
        "Total: $1,234.56",
        "المجموع: 1,250.00 ريال",
        "Amount: 1.234,56 EUR",
        "Total: ١٢٣٤٫٥٦"
    ]

    print("\n2. AMOUNT EXTRACTION:")
    for test_amount in test_amounts:
        normalized = patterns.normalize_arabic_numbers(test_amount)
        for pattern_name, pattern in patterns.number_patterns.items():
            match = pattern.search(normalized)
            if match:
                cleaned = patterns.clean_amount(match.group())
                currency = patterns.find_currency(test_amount)
                print(f"   {test_amount} → {cleaned} {currency}")
                break

    # Test invoice number
    test_invoice_nums = [
        "Invoice Number: INV-2024-001",
        "رقم الفاتورة: 12345",
        "Invoice #: 987654"
    ]

    print("\n3. INVOICE NUMBER EXTRACTION:")
    for test_inv in test_invoice_nums:
        for pattern_name, pattern in patterns.invoice_number_patterns.items():
            match = pattern.search(test_inv)
            if match:
                print(f"   {test_inv} → {match.group(1) if match.lastindex else match.group()}")
                break

    print("\n4. KEYWORDS:")
    print(f"   Total keywords (EN): {patterns.total_keywords['en'][:3]}")
    print(f"   Total keywords (AR): {patterns.total_keywords['ar'][:3]}")