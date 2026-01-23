"""
Layout Analyzer Module
Analyzes invoice layout by grouping text into spatial zones,
detecting tables, and identifying key-value pairs.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Zone:
    """Represents a spatial zone in the invoice"""
    name: str
    y_start: float  # Normalized (0-1)
    y_end: float  # Normalized (0-1)
    elements: List[Dict]

    def contains(self, normalized_y: float) -> bool:
        """Check if a y-coordinate falls within this zone"""
        return self.y_start <= normalized_y <= self.y_end


class LayoutAnalyzer:
    """
    Analyzes invoice layout to group related text elements.
    Handles spatial grouping, table detection, and key-value pairing.
    """

    def __init__(self):
        """Initialize layout analyzer with default zone definitions"""
        # Define standard invoice zones (normalized coordinates 0-1)
        self.zones = {
            'header': Zone('header', 0.0, 0.20, []),  # Top 20%: Logo, Invoice#, Date
            'vendor': Zone('vendor', 0.20, 0.35, []),  # Vendor/buyer info
            'items': Zone('items', 0.35, 0.75, []),  # Line items table
            'totals': Zone('totals', 0.75, 0.95, []),  # Subtotal, tax, total
            'footer': Zone('footer', 0.95, 1.0, [])  # Payment info, notes
        }

        # Threshold for considering elements as "nearby" (normalized distance)
        self.proximity_threshold = 0.05  # 5% of image width/height

        # Threshold for horizontal alignment (same line)
        self.horizontal_alignment_threshold = 0.02  # 2% of image height

    def analyze(self, ocr_results: List[Dict]) -> Dict:
        """
        Analyze the layout of OCR results.

        Args:
            ocr_results: List of OCR results from InvoiceOCR

        Returns:
            Dictionary containing:
            - zones: Text elements grouped by spatial zones
            - lines: Text elements grouped by horizontal lines
            - key_value_pairs: Detected label-value pairs
            - tables: Detected table structures
        """
        if not ocr_results:
            return {
                'zones': {},
                'lines': [],
                'key_value_pairs': [],
                'tables': []
            }

        # Step 1: Assign elements to zones
        zones = self._assign_to_zones(ocr_results)

        # Step 2: Group elements by horizontal lines
        lines = self._group_by_lines(ocr_results)

        # Step 3: Detect key-value pairs
        key_value_pairs = self._detect_key_value_pairs(ocr_results)

        # Step 4: Detect table structures (in items zone)
        tables = self._detect_tables(zones.get('items', []))

        return {
            'zones': zones,
            'lines': lines,
            'key_value_pairs': key_value_pairs,
            'tables': tables,
            'all_elements': ocr_results
        }

    def _assign_to_zones(self, ocr_results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Assign each OCR element to a spatial zone based on its position.

        Args:
            ocr_results: List of OCR results

        Returns:
            Dictionary mapping zone names to lists of elements
        """
        zones = defaultdict(list)

        for element in ocr_results:
            norm_y = element['normalized_center'][1]

            # Find which zone this element belongs to
            assigned = False
            for zone_name, zone in self.zones.items():
                if zone.contains(norm_y):
                    zones[zone_name].append(element)
                    assigned = True
                    break

            if not assigned:
                # Fallback to closest zone
                zones['unknown'].append(element)

        # Sort elements within each zone by position (top to bottom, left to right)
        for zone_name in zones:
            zones[zone_name].sort(
                key=lambda x: (x['normalized_center'][1], x['normalized_center'][0])
            )

        return dict(zones)

    def _group_by_lines(self, ocr_results: List[Dict]) -> List[List[Dict]]:
        """
        Group text elements that appear on the same horizontal line.

        Args:
            ocr_results: List of OCR results

        Returns:
            List of lines, where each line is a list of elements
        """
        if not ocr_results:
            return []

        # Sort by y-coordinate
        sorted_elements = sorted(
            ocr_results,
            key=lambda x: x['normalized_center'][1]
        )

        lines = []
        current_line = [sorted_elements[0]]

        for element in sorted_elements[1:]:
            # Check if this element is on the same line as the current line
            last_y = current_line[-1]['normalized_center'][1]
            current_y = element['normalized_center'][1]

            if abs(current_y - last_y) < self.horizontal_alignment_threshold:
                # Same line - add to current line
                current_line.append(element)
            else:
                # New line - sort current line by x-coordinate and start new line
                current_line.sort(key=lambda x: x['normalized_center'][0])
                lines.append(current_line)
                current_line = [element]

        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda x: x['normalized_center'][0])
            lines.append(current_line)

        return lines

    def _detect_key_value_pairs(self, ocr_results: List[Dict]) -> List[Dict]:
        """
        Detect key-value pairs (e.g., "Invoice Number: 12345").
        Looks for labels followed by values based on proximity.

        Args:
            ocr_results: List of OCR results

        Returns:
            List of detected key-value pairs
        """
        pairs = []

        # Common label patterns (both Arabic and English)
        label_keywords = [
            # English
            'invoice', 'number', 'date', 'total', 'subtotal', 'tax', 'vat',
            'customer', 'vendor', 'supplier', 'amount', 'due', 'payment',
            'address', 'phone', 'email', 'description', 'quantity', 'qty',
            'price', 'unit', 'discount',
            # Arabic
            'فاتورة', 'رقم', 'تاريخ', 'المجموع', 'الإجمالي', 'ضريبة',
            'عميل', 'مورد', 'مبلغ', 'دفع', 'عنوان', 'هاتف', 'بريد',
            'وصف', 'كمية', 'سعر', 'خصم'
        ]

        for i, element in enumerate(ocr_results):
            text_lower = element['text'].lower()

            # Check if this element contains a label keyword
            is_label = any(keyword in text_lower for keyword in label_keywords)

            # Also consider elements ending with ':' or containing ':'
            has_colon = ':' in element['text']

            if is_label or has_colon:
                # Look for nearby elements to the right (potential values)
                label_x = element['normalized_center'][0]
                label_y = element['normalized_center'][1]

                # Find candidates to the right and roughly on the same line
                candidates = []
                for j, other in enumerate(ocr_results):
                    if i == j:
                        continue

                    other_x = other['normalized_center'][0]
                    other_y = other['normalized_center'][1]

                    # Check if to the right and on same line
                    is_right = other_x > label_x
                    is_same_line = abs(other_y - label_y) < self.horizontal_alignment_threshold
                    is_close = abs(other_x - label_x) < 0.3  # Within 30% of width

                    if is_right and is_same_line and is_close:
                        distance = other_x - label_x
                        candidates.append((distance, other))

                # Take the closest candidate as the value
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    closest_distance, value_element = candidates[0]

                    pairs.append({
                        'label': element['text'],
                        'value': value_element['text'],
                        'label_bbox': element['bbox'],
                        'value_bbox': value_element['bbox'],
                        'label_confidence': element['confidence'],
                        'value_confidence': value_element['confidence'],
                        'distance': float(closest_distance)
                    })

        return pairs

    def _detect_tables(self, items_zone_elements: List[Dict]) -> List[Dict]:
        """
        Detect table structures in the items zone.
        Groups rows and attempts to identify columns.

        Args:
            items_zone_elements: Elements in the items/middle zone

        Returns:
            List of detected tables with rows and columns
        """
        if not items_zone_elements:
            return []

        # Group elements by horizontal lines (table rows)
        lines = self._group_by_lines(items_zone_elements)

        if len(lines) < 2:
            # Need at least 2 lines to form a table
            return []

        # Analyze column structure
        # Find consistent x-positions across multiple rows
        x_positions = defaultdict(list)
        for line in lines:
            for element in line:
                x = element['normalized_center'][0]
                x_positions[round(x, 2)].append(x)

        # Find x-positions that appear in multiple rows (column boundaries)
        column_positions = []
        for rounded_x, x_values in x_positions.items():
            if len(x_values) >= len(lines) * 0.5:  # Appears in at least 50% of rows
                column_positions.append(np.mean(x_values))

        column_positions.sort()

        # Build table structure
        table = {
            'num_rows': len(lines),
            'num_columns': len(column_positions) if column_positions else 0,
            'column_positions': column_positions,
            'rows': []
        }

        for line in lines:
            row = {
                'elements': line,
                'text': ' | '.join([el['text'] for el in line]),
                'y_position': np.mean([el['normalized_center'][1] for el in line])
            }
            table['rows'].append(row)

        return [table] if table['num_rows'] > 1 else []

    def get_zone_text(self, zones: Dict[str, List[Dict]], zone_name: str) -> str:
        """
        Get all text from a specific zone as a single string.

        Args:
            zones: Zone dictionary from analyze()
            zone_name: Name of the zone to extract

        Returns:
            Concatenated text from the zone
        """
        if zone_name not in zones:
            return ""

        elements = zones[zone_name]
        return '\n'.join([el['text'] for el in elements])

    def find_elements_near(
            self,
            ocr_results: List[Dict],
            target_text: str,
            direction: str = 'right',
            max_distance: float = 0.3
    ) -> List[Dict]:
        """
        Find elements near a target text element.

        Args:
            ocr_results: List of OCR results
            target_text: Text to search for (case-insensitive)
            direction: Direction to search ('right', 'left', 'below', 'above')
            max_distance: Maximum normalized distance to search

        Returns:
            List of elements found near the target
        """
        target_elements = [
            el for el in ocr_results
            if target_text.lower() in el['text'].lower()
        ]

        if not target_elements:
            return []

        # Use the first matching element as reference
        target = target_elements[0]
        target_x, target_y = target['normalized_center']

        nearby = []
        for element in ocr_results:
            if element == target:
                continue

            el_x, el_y = element['normalized_center']

            # Check direction and distance
            if direction == 'right':
                if el_x > target_x and abs(el_y - target_y) < self.horizontal_alignment_threshold:
                    if el_x - target_x < max_distance:
                        nearby.append(element)

            elif direction == 'left':
                if el_x < target_x and abs(el_y - target_y) < self.horizontal_alignment_threshold:
                    if target_x - el_x < max_distance:
                        nearby.append(element)

            elif direction == 'below':
                if el_y > target_y and abs(el_x - target_x) < self.proximity_threshold:
                    if el_y - target_y < max_distance:
                        nearby.append(element)

            elif direction == 'above':
                if el_y < target_y and abs(el_x - target_x) < self.proximity_threshold:
                    if target_y - el_y < max_distance:
                        nearby.append(element)

        return nearby


# Example usage
if __name__ == "__main__":
    # Mock OCR results for testing
    mock_results = [
        {'text': 'Invoice Number:', 'confidence': 0.95, 'bbox': [[10, 10], [100, 10], [100, 30], [10, 30]],
         'center': (55, 20), 'normalized_center': (0.1, 0.05), 'image_dimensions': (1000, 1000)},
        {'text': 'INV-12345', 'confidence': 0.98, 'bbox': [[150, 10], [250, 10], [250, 30], [150, 30]],
         'center': (200, 20), 'normalized_center': (0.2, 0.05), 'image_dimensions': (1000, 1000)},
        {'text': 'Date:', 'confidence': 0.92, 'bbox': [[10, 50], [60, 50], [60, 70], [10, 70]],
         'center': (35, 60), 'normalized_center': (0.05, 0.1), 'image_dimensions': (1000, 1000)},
        {'text': '2024-01-15', 'confidence': 0.96, 'bbox': [[150, 50], [250, 50], [250, 70], [150, 70]],
         'center': (200, 60), 'normalized_center': (0.2, 0.1), 'image_dimensions': (1000, 1000)},
        {'text': 'Item', 'confidence': 0.94, 'bbox': [[10, 400], [80, 400], [80, 420], [10, 420]],
         'center': (45, 410), 'normalized_center': (0.05, 0.4), 'image_dimensions': (1000, 1000)},
        {'text': 'Total:', 'confidence': 0.97, 'bbox': [[10, 800], [80, 800], [80, 820], [10, 820]],
         'center': (45, 810), 'normalized_center': (0.05, 0.8), 'image_dimensions': (1000, 1000)},
        {'text': '1,250.00', 'confidence': 0.99, 'bbox': [[150, 800], [250, 800], [250, 820], [150, 820]],
         'center': (200, 810), 'normalized_center': (0.2, 0.8), 'image_dimensions': (1000, 1000)},
    ]

    # Analyze layout
    analyzer = LayoutAnalyzer()
    layout = analyzer.analyze(mock_results)

    print("=" * 60)
    print("LAYOUT ANALYSIS RESULTS")
    print("=" * 60)

    print("\n1. ZONES:")
    for zone_name, elements in layout['zones'].items():
        print(f"\n   {zone_name.upper()} ({len(elements)} elements):")
        for el in elements:
            print(f"      - {el['text']}")

    print("\n2. KEY-VALUE PAIRS:")
    for pair in layout['key_value_pairs']:
        print(f"   {pair['label']} → {pair['value']}")

    print("\n3. TABLES:")
    for i, table in enumerate(layout['tables'], 1):
        print(f"   Table {i}: {table['num_rows']} rows × {table['num_columns']} columns")