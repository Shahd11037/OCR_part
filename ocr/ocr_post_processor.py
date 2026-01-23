"""
OCR Post-Processor Module
Cleans and corrects common OCR errors, especially for mixed Arabic/English receipts.
"""

import re
from typing import List, Dict, Optional
import unicodedata


class OCRPostProcessor:
    """
    Post-processes OCR results to fix common errors.
    Handles Arabic/English character confusion, date fixes, and text cleanup.
    """

    def __init__(self):
        """Initialize the post-processor with correction mappings"""

        # Common Arabic-to-English character confusions
        self.arabic_to_english = {
            'ه': 'a',  # Often confused in "Total" → "Totهl"
            '٧': 'V',  # Arabic 7 looks like V (VAT)
            '٤': 'E',  # Arabic 4 looks like backwards E (EGP)
            'ل': 'l',  # Arabic lam looks like lowercase L
            'ا': 'I',  # Arabic alif looks like I
            '؟': ')',  # Arabic question mark
            'G': 'E',  # Common G/E confusion in EGP
        }

        # Common English-to-Arabic confusions (for Arabic text)
        self.english_to_arabic = {
            'a': 'ا',
            'I': 'ا',
            'V': '٧',
            'E': '٤',
        }

        # Known keywords that should be English
        self.english_keywords = [
            'total', 'subtotal', 'vat', 'tax', 'item', 'sales',
            'cash', 'paid', 'change', 'receipt', 'order', 'date',
            'print', 'server', 'pepsi', 'cola', 'opened', 'closed'
        ]

        # Known keywords that should be Arabic
        self.arabic_keywords = [
            'المجموع', 'الإجمالي', 'ضريبة', 'نقدي', 'مدفوع'
        ]

        # Currency code fixes
        self.currency_fixes = {
            'EGE': 'EGP',
            'EG': 'EGP',
            'EGت': 'EGP',
            'EG؟': 'EGP',
            '٤G؟': 'EGP',
            'EGف': 'EGP',
            'EEP': 'EGP',  # Common misread
            'E6P': 'EGP',  # 6 instead of G
            'E0P': 'EGP',  # 0 instead of G
            'EOP': 'EGP',  # O instead of G
            '[6P': 'EGP',  # [ and 6
            '[0P': 'EGP',  # [ and 0
            'EP': 'EGP',   # Missing G
        }

        # Arabic-Indic digit to Western digit mapping
        self.arabic_digits = {
            '٠': '0',
            '١': '1',
            '٢': '2',
            '٣': '3',
            '٤': '4',
            '٥': '5',
            '٦': '6',
            '٧': '7',
            '٨': '8',
            '٩': '9',
        }

    def process_all(self, ocr_results: List[Dict]) -> List[Dict]:
        """
        Process all OCR results to clean and correct errors.

        Args:
            ocr_results: List of OCR results from InvoiceOCR

        Returns:
            Cleaned OCR results
        """
        cleaned_results = []

        for result in ocr_results:
            text = result['text']

            # Apply all cleaning steps
            cleaned_text = self.clean_text(text)
            cleaned_text = self.fix_mixed_script(cleaned_text)
            cleaned_text = self.fix_dates(cleaned_text)
            cleaned_text = self.fix_numbers(cleaned_text)
            cleaned_text = self.fix_currency(cleaned_text)
            cleaned_text = self.fix_common_words(cleaned_text)

            # Create cleaned result
            cleaned_result = result.copy()
            cleaned_result['text'] = cleaned_text
            cleaned_result['original_text'] = text  # Keep original for reference

            cleaned_results.append(cleaned_result)

        return cleaned_results

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner

        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def fix_mixed_script(self, text: str) -> str:
        """
        Fix mixed Arabic/English characters in words.
        Determines if word should be Arabic or English based on context.
        """
        # If text is mostly English, convert Arabic characters
        if self._is_mostly_latin(text):
            for arabic, english in self.arabic_to_english.items():
                text = text.replace(arabic, english)

        # If text is mostly Arabic, convert English characters
        elif self._is_mostly_arabic(text):
            for english, arabic in self.english_to_arabic.items():
                text = text.replace(english, arabic)

        return text

    def fix_dates(self, text: str) -> str:
        """Fix common date OCR errors"""
        # Fix incomplete years: 202& → 2022, 202- → 2022, etc.
        text = re.sub(r'202[&\-_=]', '2022', text)
        text = re.sub(r'202[&\-_=]\b', '2023', text)

        # Fix date separators
        text = re.sub(r'(\d{2})[-–—](\d{2})[-–—](\d{4})', r'\1-\2-\3', text)

        # Fix common date digit confusions
        text = re.sub(r'(\d{2})-(\d{2})-202([0-9o&\-_])',
                     lambda m: f"{m.group(1)}-{m.group(2)}-202{self._fix_digit(m.group(3))}",
                     text)

        return text

    def fix_numbers(self, text: str) -> str:
        """Fix common number OCR errors"""
        # First, convert Arabic-Indic digits to Western
        for arabic, western in self.arabic_digits.items():
            text = text.replace(arabic, western)

        # Only fix if text looks like a number (contains digits and decimal separators)
        if not re.search(r'\d', text):
            return text

        # Fix decimal/thousand separators
        # Keep format consistent: use . for decimals, , for thousands

        # Fix misread digits in numbers
        replacements = {
            'O': '0',  # Letter O → Zero
            'o': '0',
            'l': '1',  # Lowercase L → One
            'I': '1',  # Uppercase I → One (in numbers)
            'S': '5',  # S → 5 (in numbers)
            'B': '8',  # B → 8 (in numbers)
        }

        # Only replace in number context
        for old, new in replacements.items():
            # Replace if surrounded by digits
            text = re.sub(f'(?<=\\d){old}(?=\\d)', new, text)
            text = re.sub(f'^{old}(?=\\d)', new, text)
            text = re.sub(f'(?<=\\d){old}$', new, text)

        return text

    def fix_currency(self, text: str) -> str:
        """Fix currency code errors"""
        # Convert Arabic-Indic digits first
        for arabic, western in self.arabic_digits.items():
            text = text.replace(arabic, western)

        text_upper = text.upper()

        # Check all known incorrect patterns
        for wrong, correct in self.currency_fixes.items():
            if wrong.upper() in text_upper:
                # Replace while preserving case
                pattern = re.escape(wrong)
                text = re.sub(pattern, correct, text, flags=re.IGNORECASE)

        # Fix specific patterns with regex
        # Matches: E + (any char that could be G) + P
        text = re.sub(r'E[0O6oG\[]P', 'EGP', text, flags=re.IGNORECASE)

        # Fix: [number]P at end of amounts (missing EG)
        text = re.sub(r'\[(\d)', r'EGP \1', text)

        # Fix common misreads in "EGP"
        text = re.sub(r'E+P(?=\s*\d)', 'EGP', text)  # EEP, EP → EGP

        # Fix Arabic patterns
        text = re.sub(r'[٤E][G][؟\)]', 'EGP', text)

        return text

    def fix_common_words(self, text: str) -> str:
        """Fix commonly misread words"""
        text_lower = text.lower()

        # Check if it's a known English keyword with errors
        for keyword in self.english_keywords:
            # Calculate similarity
            if self._fuzzy_match(text_lower, keyword, threshold=0.7):
                # Replace with correct keyword, preserving case
                if text.isupper():
                    text = keyword.upper()
                elif text[0].isupper():
                    text = keyword.capitalize()
                else:
                    text = keyword
                break

        # Specific common fixes
        fixes = {
            'totهl': 'total',
            'tothal': 'total',
            'totاl': 'total',
            'subt0tal': 'subtotal',
            'sub-total': 'subtotal',
            '٧at': 'vat',
            'vاt': 'vat',
            'ilei': 'q',  # Column header "Q"
            'ileii': 'q',
        }

        for wrong, correct in fixes.items():
            if text.lower() == wrong.lower():
                # Preserve case
                if text.isupper():
                    text = correct.upper()
                elif text[0].isupper():
                    text = correct.capitalize()
                else:
                    text = correct
                break

        return text

    def _is_mostly_latin(self, text: str) -> bool:
        """Check if text is mostly Latin characters"""
        if not text:
            return True

        latin_count = sum(1 for c in text if ord(c) < 0x0600)
        return latin_count > len(text) / 2

    def _is_mostly_arabic(self, text: str) -> bool:
        """Check if text is mostly Arabic characters"""
        if not text:
            return False

        arabic_count = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)
        return arabic_count > len(text) / 2

    def _fix_digit(self, char: str) -> str:
        """Fix a single digit character"""
        digit_map = {
            'o': '0', 'O': '0',
            'l': '1', 'I': '1',
            'z': '2', 'Z': '2',
            's': '5', 'S': '5',
            'b': '8', 'B': '8',
            '&': '8',
            '-': '1', '_': '1', '=': '2'
        }
        return digit_map.get(char, char)

    def _fuzzy_match(self, str1: str, str2: str, threshold: float = 0.7) -> bool:
        """Simple fuzzy string matching"""
        if len(str1) == 0 or len(str2) == 0:
            return False

        # Calculate character overlap
        matches = sum(1 for a, b in zip(str1, str2) if a == b)
        max_len = max(len(str1), len(str2))

        similarity = matches / max_len
        return similarity >= threshold


# Convenience function
def clean_ocr_results(ocr_results: List[Dict]) -> List[Dict]:
    """
    Quick function to clean OCR results.

    Args:
        ocr_results: Raw OCR results

    Returns:
        Cleaned OCR results
    """
    processor = OCRPostProcessor()
    return processor.process_all(ocr_results)


# Example usage
if __name__ == "__main__":
    # Test cases from your receipt
    test_cases = [
        "27-09-202&",
        "Totهl",
        "٧AT",
        "٤G؟)",
        "ILEI",
        "(22.75",
        "Receipt dac27- TW:2022-1=222-_",
    ]

    processor = OCRPostProcessor()

    print("="*60)
    print("OCR POST-PROCESSING TEST")
    print("="*60)

    for test in test_cases:
        cleaned = processor.clean_text(test)
        cleaned = processor.fix_mixed_script(cleaned)
        cleaned = processor.fix_dates(cleaned)
        cleaned = processor.fix_numbers(cleaned)
        cleaned = processor.fix_currency(cleaned)
        cleaned = processor.fix_common_words(cleaned)

        print(f"\nOriginal: {test}")
        print(f"Cleaned:  {cleaned}")