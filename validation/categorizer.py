"""
Receipt Categorizer
Categorizes receipts into spending categories based on vendor name and line items.
"""

import re
from typing import List, Optional


class ReceiptCategorizer:
    """Categorize receipts into spending categories"""

    def __init__(self):
        """Initialize with category keywords"""

        # Category keywords (English + Arabic + brand names)
        self.categories = {
            "Food & Groceries": {
                "keywords": [
                    # Stores
                    "carrefour", "spinneys", "metro", "kheir zaman", "oscar",
                    "سبينس", "كارفور", "ميترو", "خير زمان",
                    # Products
                    "grocery", "supermarket", "market", "bread", "milk", "eggs",
                    "vegetables", "fruits", "meat", "chicken", "fish",
                    "خضار", "فواكه", "لحم", "دجاج", "سمك", "خبز", "لبن"
                ],
                "brands": ["carrefour", "spinneys", "metro", "kheir zaman"]
            },

            "Dining Out": {
                "keywords": [
                    # Types
                    "restaurant", "cafe", "coffee", "pizza", "burger", "kfc",
                    "mcdonalds", "hardees", "pizza hut", "dominos", "cook door",
                    "مطعم", "كافيه", "قهوة", "بيتزا", "برجر",
                    # Items
                    "pepsi", "cola", "sandwich", "meal", "fries", "chicken"
                ],
                "brands": ["kfc", "mcdonalds", "pizza hut", "tabali", "chicken fila", "cook door"]
            },

            "Transportation": {
                "keywords": [
                    # Services
                    "uber", "careem", "taxi", "gas", "fuel", "petrol", "diesel",
                    "parking", "toll", "metro", "bus",
                    "تاكسي", "بنزين", "وقود", "مواصلات", "اوبر", "كريم",
                    # Stations
                    "gas station", "petrol station", "محطة بنزين"
                ],
                "brands": ["uber", "careem", "misr petroleum", "total", "wataniya"]
            },

            "Housing / Rent": {
                "keywords": [
                    "rent", "landlord", "apartment", "flat", "house",
                    "إيجار", "شقة", "منزل", "سكن",
                    "real estate", "property"
                ],
                "brands": []
            },

            "Utilities": {
                "keywords": [
                    "electricity", "water", "gas", "internet", "phone", "mobile",
                    "we", "vodafone", "orange", "etisalat",
                    "كهرباء", "مياه", "غاز", "انترنت", "تليفون", "موبايل",
                    "bill", "utility"
                ],
                "brands": ["we", "vodafone", "orange", "etisalat", "te data","fawry","فوري" ,"فورى"]
            },

            "Health & Medical": {
                "keywords": [
                    "pharmacy", "doctor", "clinic", "hospital", "medical",
                    "medicine", "drug", "prescription", "lab", "laboratory",
                    "صيدلية", "طبيب", "عيادة", "مستشفى", "دواء", "علاج",
                    "seif", "19011", "al ezaby"
                ],
                "brands": ["seif", "19011", "al ezaby", "el ezaby"]
            },

            "Entertainment": {
                "keywords": [
                    "cinema", "movie", "theatre", "theater", "game", "gaming",
                    "playstation", "xbox", "netflix", "spotify", "subscription",
                    "سينما", "فيلم", "العاب", "لعبة",
                    "vox", "galaxy", "cinema"
                ],
                "brands": ["vox", "galaxy", "netflix", "spotify", "steam"]
            },

            "Shopping": {
                "keywords": [
                    "clothing", "clothes", "shoes", "fashion", "accessories",
                    "electronics", "mobile", "laptop", "computer", "phone",
                    "ملابس", "احذية", "موضة", "الكترونيات", "موبايل",
                    "zara", "h&m", "noon", "amazon", "jumia", "souq"
                ],
                "brands": ["zara", "h&m", "noon", "amazon", "jumia", "souq"]
            },

            "Education": {
                "keywords": [
                    "school", "university", "college", "course", "tuition",
                    "books", "library", "study", "education", "learning",
                    "مدرسة", "جامعة", "كلية", "دراسة", "تعليم", "كتب",
                    "coursera", "udemy", "udacity"
                ],
                "brands": ["coursera", "udemy", "udacity"]
            },

            "Personal Care": {
                "keywords": [
                    "salon", "barber", "haircut", "spa", "beauty", "cosmetics",
                    "makeup", "perfume", "fragrance", "shampoo",
                    "صالون", "حلاق", "تجميل", "عطر", "شامبو",
                    "loreal", "nivea", "dove"
                ],
                "brands": ["loreal", "nivea", "dove"]
            },

            "Other": {
                "keywords": [],  # Fallback category
                "brands": []
            }
        }

    def categorize(self, vendor_name: str = "", line_items: List[str] = None) -> str:
        """
        Categorize receipt based on vendor name and line items.

        Args:
            vendor_name: Vendor/merchant name from receipt
            line_items: List of product/service names

        Returns:
            Category name (one of the 11 categories)
        """
        if line_items is None:
            line_items = []

        # Combine all text for matching
        all_text = vendor_name.lower() + " " + " ".join(line_items).lower()

        # Score each category
        scores = {}
        for category, data in self.categories.items():
            if category == "Other":
                continue  # Skip Other for now

            score = 0

            # Check keywords
            for keyword in data["keywords"]:
                if keyword.lower() in all_text:
                    score += 1

            # Check brands (higher weight)
            for brand in data["brands"]:
                if brand.lower() in all_text:
                    score += 3  # Brands are strong indicators

            scores[category] = score

        # Find best match
        if scores:
            best_category = max(scores, key=scores.get)
            if scores[best_category] > 0:
                return best_category

        # Default to "Other"
        return "Other"


# Quick test
if __name__ == "__main__":
    categorizer = ReceiptCategorizer()

    # Test cases
    tests = [
        ("Tabali", ["Pepsi"], "Dining Out"),
        ("Chicken Fila", ["Spicy Chicken Ranch Pizzawich", "Cheese Fries"], "Dining Out"),
        ("Carrefour", ["Bread", "Milk", "Eggs"], "Food & Groceries"),
        ("Uber", [], "Transportation"),
        ("Seif Pharmacy", ["Medicine"], "Health & Medical"),
        ("Unknown Store", [], "Other"),
    ]

    print("Categorization Tests:")
    print("=" * 60)

    for vendor, items, expected in tests:
        result = categorizer.categorize(vendor, items)
        status = "✅" if result == expected else "❌"
        print(f"{status} Vendor: {vendor:20s} → {result:20s} (expected: {expected})")

    print("=" * 60)