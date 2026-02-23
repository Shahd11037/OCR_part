"""
Simplified Field Extractor
Extracts: date, total, and detects line items for categorization.
"""

import re
from typing import Dict, List, Optional
from datetime import datetime


class SimpleExtractor:
    """Extract date and total from OCR results"""

    def __init__(self):
        """Initialize with minimal patterns"""

        # Date keywords (English + Arabic)
        self.date_keywords = [
            'date', 'dated', 'invoice date', 'bill date',
            'تاريخ', 'بتاريخ', 'التاريخ','تاريخ الطباعة'
        ]

        # Total keywords (English + Arabic)
        self.total_keywords = [
            'total', 'grand total', 'net total', 'amount due',
            'المجموع', 'الإجمالي', 'المبلغ', 'الكلي','المطلوب', 'المبلغ المستحق','الصافي'
        ]

        # Date patterns
        self.date_patterns = [
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',  # 2022-09-27, 2022/09/27
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',  # 27-09-2022, 27/09/2022
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2}\b',  # 27-09-22
        ]

        # Amount pattern (decimal numbers)
        self.amount_pattern = r'\b\d{1,3}(?:[,\s]\d{3})*(?:[.,]\d{2})?\b'

    def extract(self, ocr_results: List[Dict]) -> Dict:
        """
        Extract date, total, and line items from OCR results.

        Args:
            ocr_results: List of OCR text elements with positions

        Returns:
            Dict with date, total, and line_items for categorization
        """
        result = {
            'date': None,
            'total': None,
            'line_items': []  # For categorization
        }

        # Extract date
        result['date'] = self._extract_date(ocr_results)

        # Extract total
        result['total'] = self._extract_total(ocr_results)

        # Extract line items (for categorization)
        result['line_items'] = self._extract_line_items(ocr_results)

        return result

    def _extract_date(self, ocr_results: List[Dict]) -> Optional[str]:
        """Extract and normalize date"""

        # Strategy 1: Look for date keywords + nearby dates
        for i, element in enumerate(ocr_results):
            text = element['text'].lower()

            # Check if this element has a date keyword
            if any(keyword in text for keyword in self.date_keywords):
                # Look in nearby elements (within 3 positions)
                for j in range(max(0, i - 1), min(len(ocr_results), i + 4)):
                    nearby_text = ocr_results[j]['text']
                    date = self._parse_date(nearby_text)
                    if date:
                        return date

        # Strategy 2: Find any date pattern in the document
        for element in ocr_results:
            date = self._parse_date(element['text'])
            if date:
                return date

        return None

    def _parse_date(self, text: str) -> Optional[str]:
        """Parse and normalize date to YYYY-MM-DD"""

        for pattern in self.date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group()
                return self._normalize_date(date_str)

        return None

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to YYYY-MM-DD format"""

        # Replace / with -
        date_str = date_str.replace('/', '-')

        # Try different formats
        formats = [
            '%Y-%m-%d',  # 2022-09-27
            '%d-%m-%Y',  # 27-09-2022
            '%m-%d-%Y',  # 09-27-2022
            '%d-%m-%y',  # 27-09-22
            '%y-%m-%d',  # 22-09-27
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # Return in YYYY-MM-DD format
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue

        return None

    def _extract_total(self, ocr_results: List[Dict]) -> Optional[float]:
        candidates = []
        for i, element in enumerate(ocr_results):
            text = element['text'].lower()
            if any(keyword in text for keyword in self.total_keywords):
                # Scan a small window around the keyword
                for j in range(max(0, i - 1), min(len(ocr_results), i + 5)):
                    amount = self._parse_amount(ocr_results[j]['text'])
                    if amount:
                        # Generic confidence: Prefer numbers with decimals
                        conf = 1.0 if ('.' in ocr_results[j]['text']) else 0.6
                        candidates.append({'amount': amount, 'conf': conf})

        if candidates:
            # Pick the most confident, then the largest (The true "Grand Total")
            best = sorted(candidates, key=lambda x: (x['conf'], x['amount']), reverse=True)
            return best[0]['amount']

        # Universal Fallback: Max value within a reasonable price range
        all_vals = [self._parse_amount(e['text']) for e in ocr_results if self._parse_amount(e['text'])]
        return max([v for v in all_vals if 0.5 < v < 50000]) if all_vals else None

    def _parse_amount(self, text: str) -> Optional[float]:
        """Parse amount from text"""

        # Clean the text
        text = text.strip()

        # Remove currency symbols and codes
        text = re.sub(r'[EGP$£€¥₹₪﷼]', '', text, flags=re.IGNORECASE)
        text = text.strip()

        # Find decimal number
        match = re.search(self.amount_pattern, text)
        if not match:
            return None

        amount_str = match.group()

        # Clean: remove spaces and commas (thousand separators)
        amount_str = amount_str.replace(' ', '').replace(',', '')

        # Handle European format (123.456,78 → 123456.78)
        if '.' in amount_str and ',' in amount_str:
            # If both present, last one is decimal
            if amount_str.rindex(',') > amount_str.rindex('.'):
                # Format: 1.234,56 → 1234.56
                amount_str = amount_str.replace('.', '').replace(',', '.')
            else:
                # Format: 1,234.56 → 1234.56
                amount_str = amount_str.replace(',', '')

        try:
            return float(amount_str)
        except ValueError:
            return None

    def _extract_line_items(self, ocr_results: List[Dict]) -> List[str]:
        """
        Extract line item text for categorization.
        Returns list of product/service names found.
        """
        line_items = []

        # Look for common item patterns
        # Typically items are followed by amounts
        for i, element in enumerate(ocr_results):
            text = element['text'].strip()

            # Skip if it's a keyword, number, or currency
            if len(text) < 3:
                continue
            if any(kw in text.lower() for kw in self.total_keywords + self.date_keywords):
                continue
            if re.match(r'^[\d.,\s]+$', text):
                continue
            if re.match(r'^[EGP$£€]+', text, re.IGNORECASE):
                continue

            # Check if next element is an amount (indicates this is an item)
            if i + 1 < len(ocr_results):
                next_text = ocr_results[i + 1]['text']
                if self._parse_amount(next_text):
                    line_items.append(text)

        return line_items


# Quick test
if __name__ == "__main__":
    # Test with sample OCR results
    sample_ocr = [
        {'text': 'Tabali', 'confidence': 0.95},
        {'text': 'Date: 27-09-2022', 'confidence': 0.90},
        {'text': 'Pepsi', 'confidence': 0.85},
        {'text': '19.95', 'confidence': 0.95},
        {'text': 'Total', 'confidence': 0.92},
        {'text': '22.75', 'confidence': 0.98},
    ]

    extractor = SimpleExtractor()
    result = extractor.extract(sample_ocr)

    print("Extracted:")
    print(f"  Date: {result['date']}")
    print(f"  Total: {result['total']}")
    print(f"  Line Items: {result['line_items']}")