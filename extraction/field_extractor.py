"""
Field Extractor Module
Extracts specific fields from invoices using regex patterns,
layout analysis, and spatial heuristics.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from extraction.patterns import InvoicePatterns
from extraction.layout_analyzer import LayoutAnalyzer


class FieldExtractor:
    """
    Extracts structured fields from invoice OCR results.
    Combines regex patterns, keyword matching, and spatial layout analysis.
    """

    def __init__(self):
        """Initialize field extractor with patterns and layout analyzer"""
        self.patterns = InvoicePatterns()
        self.layout_analyzer = LayoutAnalyzer()

    def extract_all_fields(
            self,
            ocr_results: List[Dict],
            layout: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract all invoice fields from OCR results.

        Args:
            ocr_results: List of OCR results from InvoiceOCR
            layout: Optional pre-computed layout analysis

        Returns:
            Dictionary containing all extracted fields with confidence scores
        """
        # Perform layout analysis if not provided
        if layout is None:
            layout = self.layout_analyzer.analyze(ocr_results)

        # Extract each field
        extracted = {
            'invoice_number': self.extract_invoice_number(ocr_results, layout),
            'dates': self.extract_dates(ocr_results, layout),
            'amounts': self.extract_amounts(ocr_results, layout),
            'tax_info': self.extract_tax_info(ocr_results, layout),
            'vendor_info': self.extract_vendor_info(ocr_results, layout),
            'customer_info': self.extract_customer_info(ocr_results, layout),
            'line_items': self.extract_line_items(ocr_results, layout),
            'currency': self.extract_currency(ocr_results),
            'payment_terms': self.extract_payment_terms(ocr_results, layout),
        }

        # Add metadata
        extracted['metadata'] = {
            'extraction_confidence': self._calculate_overall_confidence(extracted),
            'fields_extracted': sum(1 for v in extracted.values() if v and v != {}),
            'total_fields': 9
        }

        return extracted

    def extract_invoice_number(
            self,
            ocr_results: List[Dict],
            layout: Dict
    ) -> Dict[str, Any]:
        """
        Extract invoice number from header zone.

        Returns:
            Dict with 'value', 'confidence', 'source'
        """
        # Search primarily in header zone
        header_elements = layout['zones'].get('header', [])

        # Strategy 1: Look for keyword + nearby value
        for element in header_elements:
            text_lower = element['text'].lower()

            # Check for English keywords
            for keyword in self.patterns.invoice_number_keywords['en']:
                if keyword in text_lower:
                    # Found label, look for value nearby
                    nearby = self.layout_analyzer.find_elements_near(
                        header_elements,
                        element['text'],
                        direction='right',
                        max_distance=0.4
                    )

                    for near_element in nearby:
                        # Try to extract invoice number using patterns
                        for pattern_name, pattern in self.patterns.invoice_number_patterns.items():
                            match = pattern.search(near_element['text'])
                            if match:
                                inv_num = match.group(1) if match.lastindex else match.group()
                                return {
                                    'value': inv_num.strip(),
                                    'confidence': (element['confidence'] + near_element['confidence']) / 2,
                                    'source': 'keyword_proximity',
                                    'zone': 'header'
                                }

            # Check for Arabic keywords
            for keyword in self.patterns.invoice_number_keywords['ar']:
                if keyword in element['text']:
                    nearby = self.layout_analyzer.find_elements_near(
                        header_elements,
                        element['text'],
                        direction='right',
                        max_distance=0.4
                    )

                    for near_element in nearby:
                        # Extract numbers from nearby element
                        numbers = re.findall(r'\d+', near_element['text'])
                        if numbers and len(numbers[0]) >= 4:
                            return {
                                'value': numbers[0],
                                'confidence': (element['confidence'] + near_element['confidence']) / 2,
                                'source': 'arabic_keyword_proximity',
                                'zone': 'header'
                            }

        # Strategy 2: Pattern matching on all header text
        for element in header_elements:
            for pattern_name, pattern in self.patterns.invoice_number_patterns.items():
                if pattern_name == 'sequential':
                    continue  # Skip too generic pattern for now

                match = pattern.search(element['text'])
                if match:
                    inv_num = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group()
                    return {
                        'value': inv_num.strip(),
                        'confidence': element['confidence'],
                        'source': f'pattern_{pattern_name}',
                        'zone': 'header'
                    }

        # Strategy 3: Look in key-value pairs
        for pair in layout.get('key_value_pairs', []):
            if any(kw in pair['label'].lower() for kw in self.patterns.invoice_number_keywords['en']):
                return {
                    'value': pair['value'],
                    'confidence': pair['value_confidence'],
                    'source': 'key_value_pair',
                    'zone': 'header'
                }

        return {'value': None, 'confidence': 0.0, 'source': None, 'zone': None}

    def extract_dates(
            self,
            ocr_results: List[Dict],
            layout: Dict
    ) -> Dict[str, Any]:
        """
        Extract dates (invoice date, due date, etc.)

        Returns:
            Dict with 'invoice_date', 'due_date', etc.
        """
        dates = {
            'invoice_date': {'value': None, 'confidence': 0.0},
            'due_date': {'value': None, 'confidence': 0.0},
            'other_dates': []
        }

        # Search primarily in header and vendor zones
        search_elements = (
                layout['zones'].get('header', []) +
                layout['zones'].get('vendor', [])
        )

        # Strategy 1: Keyword-based extraction
        for element in search_elements:
            text = element['text']
            text_lower = text.lower()

            # Check for date keywords
            is_invoice_date = any(
                kw in text_lower for kw in ['invoice date', 'issue date', 'date', 'dated', 'تاريخ الفاتورة', 'تاريخ'])
            is_due_date = any(kw in text_lower for kw in ['due date', 'payment date', 'due', 'تاريخ الاستحقاق'])

            if is_invoice_date or is_due_date:
                # Look for date in current element or nearby
                date_value = self._extract_date_from_text(text)

                if not date_value:
                    # Check nearby elements
                    nearby = self.layout_analyzer.find_elements_near(
                        search_elements,
                        text,
                        direction='right',
                        max_distance=0.3
                    )
                    for near_element in nearby:
                        date_value = self._extract_date_from_text(near_element['text'])
                        if date_value:
                            break

                if date_value:
                    date_dict = {
                        'value': date_value,
                        'confidence': element['confidence'],
                        'source': 'keyword_proximity'
                    }

                    if is_due_date:
                        dates['due_date'] = date_dict
                    elif is_invoice_date and not dates['invoice_date']['value']:
                        dates['invoice_date'] = date_dict

        # Strategy 2: Pattern-based extraction from all elements
        if not dates['invoice_date']['value']:
            for element in search_elements:
                date_value = self._extract_date_from_text(element['text'])
                if date_value:
                    dates['invoice_date'] = {
                        'value': date_value,
                        'confidence': element['confidence'],
                        'source': 'pattern_match'
                    }
                    break

        # Strategy 3: Check key-value pairs
        for pair in layout.get('key_value_pairs', []):
            label_lower = pair['label'].lower()

            if 'date' in label_lower or 'تاريخ' in pair['label']:
                date_value = self._extract_date_from_text(pair['value'])
                if date_value:
                    if 'due' in label_lower or 'استحقاق' in pair['label']:
                        dates['due_date'] = {
                            'value': date_value,
                            'confidence': pair['value_confidence'],
                            'source': 'key_value_pair'
                        }
                    elif not dates['invoice_date']['value']:
                        dates['invoice_date'] = {
                            'value': date_value,
                            'confidence': pair['value_confidence'],
                            'source': 'key_value_pair'
                        }

        return dates

    def extract_amounts(
            self,
            ocr_results: List[Dict],
            layout: Dict
    ) -> Dict[str, Any]:
        """
        Extract monetary amounts (total, subtotal, tax, discount).

        Returns:
            Dict with 'total', 'subtotal', 'tax', 'discount'
        """
        amounts = {
            'total': {'value': None, 'confidence': 0.0},
            'subtotal': {'value': None, 'confidence': 0.0},
            'tax': {'value': None, 'confidence': 0.0},
            'discount': {'value': None, 'confidence': 0.0}
        }

        # Search primarily in totals zone
        totals_elements = layout['zones'].get('totals', [])

        # Also check items zone for subtotal
        items_elements = layout['zones'].get('items', [])

        # Strategy 1: Keyword + nearby amount in totals zone
        amount_types = [
            ('total', self.patterns.total_keywords),
            ('subtotal', self.patterns.subtotal_keywords),
            ('tax', self.patterns.tax_keywords),
            ('discount', self.patterns.discount_keywords)
        ]

        for amount_type, keywords_dict in amount_types:
            keywords = keywords_dict['en'] + keywords_dict['ar']

            # Search in totals zone
            for element in totals_elements:
                text_lower = element['text'].lower()

                # Check if element contains amount keyword
                if any(kw in text_lower or kw in element['text'] for kw in keywords):
                    # Try to extract amount from same element
                    amount = self._extract_amount_from_text(element['text'])

                    if not amount:
                        # Look for amount in nearby element (to the right)
                        nearby = self.layout_analyzer.find_elements_near(
                            totals_elements,
                            element['text'],
                            direction='right',
                            max_distance=0.4
                        )

                        for near_element in nearby:
                            amount = self._extract_amount_from_text(near_element['text'])
                            if amount:
                                amounts[amount_type] = {
                                    'value': amount,
                                    'confidence': (element['confidence'] + near_element['confidence']) / 2,
                                    'source': 'keyword_proximity',
                                    'zone': 'totals'
                                }
                                break
                    else:
                        amounts[amount_type] = {
                            'value': amount,
                            'confidence': element['confidence'],
                            'source': 'keyword_same_element',
                            'zone': 'totals'
                        }

        # Strategy 2: Validate and fill missing amounts
        # If we have total and tax, we can calculate subtotal
        if amounts['total']['value'] and amounts['tax']['value'] and not amounts['subtotal']['value']:
            calculated_subtotal = amounts['total']['value'] - amounts['tax']['value']
            if calculated_subtotal > 0:
                amounts['subtotal'] = {
                    'value': calculated_subtotal,
                    'confidence': min(amounts['total']['confidence'], amounts['tax']['confidence']),
                    'source': 'calculated',
                    'zone': 'totals'
                }

        # Strategy 3: Check key-value pairs
        for pair in layout.get('key_value_pairs', []):
            label_lower = pair['label'].lower()

            for amount_type, keywords_dict in amount_types:
                keywords = keywords_dict['en'] + keywords_dict['ar']
                if any(kw in label_lower or kw in pair['label'] for kw in keywords):
                    if not amounts[amount_type]['value']:
                        amount = self._extract_amount_from_text(pair['value'])
                        if amount:
                            amounts[amount_type] = {
                                'value': amount,
                                'confidence': pair['value_confidence'],
                                'source': 'key_value_pair',
                                'zone': 'totals'
                            }

        return amounts

    def extract_tax_info(
            self,
            ocr_results: List[Dict],
            layout: Dict
    ) -> Dict[str, Any]:
        """Extract tax/VAT information"""
        tax_info = {
            'tax_id': {'value': None, 'confidence': 0.0},
            'tax_percentage': {'value': None, 'confidence': 0.0}
        }

        # Search all zones for tax ID
        for element in ocr_results:
            # Extract tax ID number
            for pattern_name, pattern in self.patterns.tax_patterns.items():
                if pattern_name == 'tax_id':
                    match = pattern.search(element['text'])
                    if match:
                        tax_info['tax_id'] = {
                            'value': match.group(1),
                            'confidence': element['confidence'],
                            'source': f'pattern_{pattern_name}'
                        }

                # Extract tax percentage
                elif pattern_name == 'tax_percentage':
                    match = pattern.search(element['text'])
                    if match:
                        tax_info['tax_percentage'] = {
                            'value': float(match.group(1)),
                            'confidence': element['confidence'],
                            'source': f'pattern_{pattern_name}'
                        }

        return tax_info

    def extract_vendor_info(
            self,
            ocr_results: List[Dict],
            layout: Dict
    ) -> Dict[str, Any]:
        """Extract vendor/seller information"""
        vendor_info = {
            'name': {'value': None, 'confidence': 0.0},
            'address': {'value': [], 'confidence': 0.0},
            'phone': {'value': None, 'confidence': 0.0},
            'email': {'value': None, 'confidence': 0.0}
        }

        # Search in header and vendor zones
        search_elements = (
                layout['zones'].get('header', []) +
                layout['zones'].get('vendor', [])
        )

        # Extract phone numbers
        for element in search_elements:
            for pattern_name, pattern in self.patterns.phone_patterns.items():
                match = pattern.search(element['text'])
                if match:
                    vendor_info['phone'] = {
                        'value': match.group(),
                        'confidence': element['confidence'],
                        'source': f'pattern_{pattern_name}'
                    }
                    break

        # Extract email
        for element in search_elements:
            match = self.patterns.email_pattern.search(element['text'])
            if match:
                vendor_info['email'] = {
                    'value': match.group(),
                    'confidence': element['confidence'],
                    'source': 'pattern_email'
                }
                break

        # Extract company name (usually in top portion, first few lines)
        if search_elements:
            # Take first non-date, non-number element as potential company name
            for element in search_elements[:5]:
                text = element['text'].strip()
                # Skip if it's just a number or date
                if len(text) > 3 and not text.replace('-', '').replace('/', '').isdigit():
                    if not any(pattern.search(text) for pattern in self.patterns.date_patterns.values()):
                        vendor_info['name'] = {
                            'value': text,
                            'confidence': element['confidence'],
                            'source': 'position_heuristic'
                        }
                        break

        return vendor_info

    def extract_customer_info(
            self,
            ocr_results: List[Dict],
            layout: Dict
    ) -> Dict[str, Any]:
        """Extract customer/buyer information"""
        customer_info = {
            'name': {'value': None, 'confidence': 0.0},
            'address': {'value': [], 'confidence': 0.0}
        }

        # Search in vendor zone (customer info often below vendor info)
        vendor_elements = layout['zones'].get('vendor', [])

        # Look for customer keywords
        for i, element in enumerate(vendor_elements):
            text_lower = element['text'].lower()

            if any(kw in text_lower for kw in
                   self.patterns.customer_keywords['en'] + self.patterns.customer_keywords['ar']):
                # Found customer label, next elements might be customer info
                if i + 1 < len(vendor_elements):
                    customer_info['name'] = {
                        'value': vendor_elements[i + 1]['text'],
                        'confidence': vendor_elements[i + 1]['confidence'],
                        'source': 'keyword_proximity'
                    }
                break

        return customer_info

    def extract_line_items(
            self,
            ocr_results: List[Dict],
            layout: Dict
    ) -> List[Dict]:
        """Extract line items from table"""
        line_items = []

        # Get detected tables
        tables = layout.get('tables', [])

        if not tables:
            return line_items

        # Process first table (main items table)
        table = tables[0]

        for row in table['rows']:
            # Extract information from each row
            row_elements = row['elements']

            if len(row_elements) < 2:
                continue  # Skip header or invalid rows

            item = {
                'description': '',
                'quantity': None,
                'unit_price': None,
                'amount': None,
                'confidence': 0.0
            }

            # Simple heuristic:
            # - First column: description
            # - Last column: amount
            # - Middle columns: quantity, unit price

            if row_elements:
                item['description'] = row_elements[0]['text']
                item['confidence'] = row_elements[0]['confidence']

            if len(row_elements) >= 2:
                # Try to extract amount from last column
                last_text = row_elements[-1]['text']
                amount = self._extract_amount_from_text(last_text)
                if amount:
                    item['amount'] = amount

            if len(row_elements) >= 3:
                # Try to extract quantity from middle columns
                for el in row_elements[1:-1]:
                    # Look for simple numbers (quantity)
                    if re.match(r'^\d+$', el['text'].strip()):
                        item['quantity'] = int(el['text'].strip())
                        break

            line_items.append(item)

        return line_items

    def extract_currency(self, ocr_results: List[Dict]) -> Dict[str, Any]:
        """Extract currency code/symbol"""
        # Concatenate all text
        full_text = ' '.join([el['text'] for el in ocr_results])

        currency = self.patterns.find_currency(full_text)

        return {
            'value': currency,
            'confidence': 0.8,  # Moderate confidence
            'source': 'pattern_match'
        }

    def extract_payment_terms(
            self,
            ocr_results: List[Dict],
            layout: Dict
    ) -> Dict[str, Any]:
        """Extract payment terms if available"""
        payment_terms = {
            'value': None,
            'confidence': 0.0
        }

        # Search in footer zone
        footer_elements = layout['zones'].get('footer', [])

        payment_keywords = ['payment', 'terms', 'net', 'due', 'days', 'شروط الدفع']

        for element in footer_elements:
            if any(kw in element['text'].lower() for kw in payment_keywords):
                payment_terms = {
                    'value': element['text'],
                    'confidence': element['confidence'],
                    'source': 'keyword_match'
                }
                break

        return payment_terms

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _extract_date_from_text(self, text: str) -> Optional[str]:
        """Extract and normalize date from text"""
        # Normalize Arabic numbers
        text = self.patterns.normalize_arabic_numbers(text)

        # Try each date pattern
        for pattern_name, pattern in self.patterns.date_patterns.items():
            match = pattern.search(text)
            if match:
                try:
                    # Parse based on pattern type
                    if pattern_name == 'iso_date':
                        year, month, day = match.groups()
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

                    elif pattern_name in ['dmy_slash', 'dmy_dash']:
                        day, month, year = match.groups()
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

                    elif pattern_name == 'month_name':
                        day, month_name, year = match.groups()
                        month_map = {
                            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                        }
                        month = month_map.get(month_name.lower()[:3], '01')
                        return f"{year}-{month}-{day.zfill(2)}"

                    elif pattern_name == 'name_month':
                        month_name, day, year = match.groups()
                        month_map = {
                            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                        }
                        month = month_map.get(month_name.lower()[:3], '01')
                        return f"{year}-{month}-{day.zfill(2)}"

                    elif pattern_name == 'compact_date':
                        year, month, day = match.groups()
                        return f"{year}-{month}-{day}"

                    else:
                        # Return as-is for other formats
                        return match.group()

                except (ValueError, AttributeError):
                    continue

        return None

    def _extract_amount_from_text(self, text: str) -> Optional[float]:
        """Extract and clean amount from text"""
        # Normalize Arabic numbers
        text = self.patterns.normalize_arabic_numbers(text)

        # Try each number pattern
        for pattern_name, pattern in self.patterns.number_patterns.items():
            match = pattern.search(text)
            if match:
                # Extract the matched amount
                amount_str = match.group(1) if match.lastindex else match.group()

                # Clean and convert to float
                amount = self.patterns.clean_amount(amount_str)

                # Validate (amounts should be positive and reasonable)
                if amount > 0 and amount < 1000000000:  # Less than 1 billion
                    return amount

        return None

    def _calculate_overall_confidence(self, extracted: Dict) -> float:
        """Calculate overall extraction confidence"""
        confidences = []

        # Collect all confidence scores
        for field_name, field_value in extracted.items():
            if isinstance(field_value, dict):
                if 'confidence' in field_value:
                    confidences.append(field_value['confidence'])
                else:
                    # Nested dict (like amounts, dates)
                    for sub_field_value in field_value.values():
                        if isinstance(sub_field_value, dict) and 'confidence' in sub_field_value:
                            confidences.append(sub_field_value['confidence'])

        return sum(confidences) / len(confidences) if confidences else 0.0


# Example usage
if __name__ == "__main__":
    # Mock OCR results for testing
    mock_results = [
        {'text': 'Invoice Number: INV-2024-001', 'confidence': 0.95, 'bbox': [[10, 10], [200, 10], [200, 30], [10, 30]],
         'center': (105, 20), 'normalized_center': (0.1, 0.02), 'image_dimensions': (1000, 1000)},
        {'text': 'Date: 2024-01-15', 'confidence': 0.96, 'bbox': [[10, 50], [150, 50], [150, 70], [10, 70]],
         'center': (80, 60), 'normalized_center': (0.08, 0.06), 'image_dimensions': (1000, 1000)},
        {'text': 'Total:', 'confidence': 0.97, 'bbox': [[10, 800], [80, 800], [80, 820], [10, 820]],
         'center': (45, 810), 'normalized_center': (0.05, 0.81), 'image_dimensions': (1000, 1000)},
        {'text': '$1,250.00', 'confidence': 0.99, 'bbox': [[150, 800], [250, 800], [250, 820], [150, 820]],
         'center': (200, 810), 'normalized_center': (0.2, 0.81), 'image_dimensions': (1000, 1000)},
    ]

    # Extract fields
    extractor = FieldExtractor()
    fields = extractor.extract_all_fields(mock_results)

    print("=" * 60)
    print("FIELD EXTRACTION RESULTS")
    print("=" * 60)

    print(f"\nInvoice Number: {fields['invoice_number']}")
    print(f"\nDates: {fields['dates']}")
    print(f"\nAmounts: {fields['amounts']}")
    print(f"\nCurrency: {fields['currency']}")
    print(f"\nMetadata: {fields['metadata']}")