"""
Test: Currency Pattern Execution Order
Verifies that Arabic characters in currency codes are handled correctly
BEFORE digit conversion.
"""

import sys

sys.path.append('/mnt/user-data/outputs/invoice_processor')

from ocr.ocr_post_processor import OCRPostProcessor


def test_currency_pattern_order():
    """
    Test that execution order is correct:
    1. Fix Arabic patterns FIRST (٤GP → EGP)
    2. Then convert digits (٩ → 9)

    If order is wrong, ٤GP becomes 4GP and won't match patterns!
    """

    processor = OCRPostProcessor()

    print("=" * 70)
    print("CURRENCY PATTERN EXECUTION ORDER TEST")
    print("=" * 70)
    print("\nThis test verifies the fix for your execution order concern.\n")

    test_cases = [
        # The critical case you asked about!
        ("٤GP 123.45", "EGP 123.45", "Arabic ٤ instead of E"),
        ("٤G؟ 99.99", "EGP 99.99", "Arabic ٤ and Arabic ؟"),
        ("٤GP", "EGP", "Just ٤GP"),

        # Other Arabic variations
        ("٤GP 56.14", "EGP 56.14", "From Chicken Fila receipt"),
        ("٤6P 66.58", "EGP 66.58", "Arabic ٤ + digit 6"),

        # English variations (should still work)
        ("E6P 123", "EGP 123", "E with digit 6"),
        ("E0P 456", "EGP 456", "E with digit 0"),
        ("[6P 789", "EGP 789", "Bracket instead of E"),
        ("EEP 100", "EGP 100", "Double E"),
        ("EP 200", "EGP 200", "Missing G"),

        # Mixed cases
        ("EGP 302.64", "EGP 302.64", "Already correct"),
        ("egp 150", "EGP 150", "Lowercase"),
    ]

    all_passed = True
    critical_passed = True

    for i, (input_text, expected, description) in enumerate(test_cases, 1):
        result = processor.fix_currency(input_text)
        passed = result == expected

        # First 3 tests are critical (the ones you asked about)
        is_critical = i <= 3

        if is_critical and not passed:
            critical_passed = False
            status = "❌ CRITICAL FAIL"
        elif passed:
            status = "✅ PASS"
        else:
            status = "⚠️  FAIL"
            all_passed = False

        marker = "🔴" if is_critical else "  "

        print(f"{marker} Test {i}: {description}")
        print(f"   Input:    '{input_text}'")
        print(f"   Expected: '{expected}'")
        print(f"   Got:      '{result}'")
        print(f"   Status:   {status}")
        print()

    print("=" * 70)

    if critical_passed:
        print("✅ CRITICAL TESTS PASSED!")
        print("   The execution order issue is FIXED.")
        print("   Arabic ٤ in currency codes is handled correctly.")
    else:
        print("❌ CRITICAL TESTS FAILED!")
        print("   Execution order is still wrong.")
        print("   ٤GP is being converted to 4GP before pattern matching.")

    print()

    if all_passed:
        print("✅ ALL TESTS PASSED - Currency patterns work perfectly!")
    else:
        print("⚠️  Some non-critical tests failed (might need pattern tuning)")

    print("=" * 70)

    return critical_passed


def demonstrate_the_issue():
    """
    Demonstrates WHY execution order matters.
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Why Execution Order Matters")
    print("=" * 70)

    test_input = "٤GP 123.45"

    print(f"\nOriginal OCR output: '{test_input}'")
    print("\n--- WRONG ORDER (Original Bug) ---")
    print("Step 1: Convert ٤ → 4")
    print(f"  Result: '4GP 123.45'")
    print("Step 2: Try to match pattern [٤E][G][؟\\)]")
    print(f"  Result: ❌ No match! (looking for ٤ or E, but we have 4)")
    print(f"  Final: '4GP 123.45' ❌ WRONG!")

    print("\n--- CORRECT ORDER (After Fix) ---")
    print("Step 1: Match pattern [٤][G][Pp؟\\)]")
    print(f"  Result: ✅ Match found! Replace with 'EGP'")
    print(f"  After: 'EGP 123.45'")
    print("Step 2: Convert remaining Arabic digits (if any)")
    print(f"  Result: 'EGP 123.45' (no digits to convert in 'EGP')")
    print(f"  Final: 'EGP 123.45' ✅ CORRECT!")

    processor = OCRPostProcessor()
    actual_result = processor.fix_currency(test_input)

    print(f"\n--- ACTUAL TEST ---")
    print(f"Input:  '{test_input}'")
    print(f"Output: '{actual_result}'")
    print(f"Status: {'✅ CORRECT' if actual_result == 'EGP 123.45' else '❌ WRONG'}")
    print("=" * 70)


if __name__ == "__main__":
    # Demonstrate the issue
    demonstrate_the_issue()

    # Run tests
    print("\n")
    passed = test_currency_pattern_order()

    if passed:
        print("\n🎉 Your execution order concern is RESOLVED!")
        print("   ٤GP will correctly become EGP (not 4GP)")
    else:
        print("\n⚠️  The fix needs more work.")