import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sympy import false


def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def getSkewAngle(cvImage) -> float:
    """
    Detect skew angle using edge detection.
    Focuses on horizontal text lines.
    """
    gray = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Use HoughLines to detect line orientations
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        # Normalize to -45 to 45
        if angle > 45:
            angle = angle - 180
        angles.append(angle)

    # Filter to keep only near-horizontal angles (text lines)
    # Exclude near-vertical angles (noise, barcodes, edges)
    text_angles = [a for a in angles if abs(a) < 30]

    if len(text_angles) == 0:
        return 0.0

    # Use median for robustness
    dominant_angle = np.median(text_angles)

    if abs(dominant_angle) < 2.0:
        return 0.0

    return -1.0 * dominant_angle


def rotateImage(cvImage, angle: float):
    if angle == 0:
        return cvImage

    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, angle)


def remove_borders(image):
    """
    Remove white/black borders more conservatively.
    Only removes obvious borders, keeps content.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape

    # Scan from edges inward to find content boundary
    # Scan from top
    top = 0
    for i in range(h):
        if np.mean(image[i, :]) < 240:  # Not mostly white
            top = i
            break

    # Scan from bottom
    bottom = h
    for i in range(h - 1, -1, -1):
        if np.mean(image[i, :]) < 240:
            bottom = i + 1
            break

    # Scan from left
    left = 0
    for j in range(w):
        if np.mean(image[:, j]) < 240:
            left = j
            break

    # Scan from right
    right = w
    for j in range(w - 1, -1, -1):
        if np.mean(image[:, j]) < 240:
            right = j + 1
            break

    # Add small padding but be conservative
    padding = 2
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(h, bottom + padding)
    right = min(w, right + padding)

    # Sanity check - if crop is too small, return original
    crop_h = bottom - top
    crop_w = right - left
    if crop_h < h * 0.3 or crop_w < w * 0.3:
        return image

    crop = image[top:bottom, left:right]
    return crop


def enhance_contrast(image):
    """
    Enhance contrast for washed/faded images using CLAHE
    (Contrast Limited Adaptive Histogram Equalization)
    """
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced


def preprocess_image(image_path, use_clahe=True):
    """
    Preprocess invoice image for OCR with robust deskewing and border removal.
    Optimized for mixed Arabic/English text.

    Args:
        image_path: Path to the image
        use_clahe: Use CLAHE contrast enhancement (good for washed images)
    """
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error reading image: {image_path}")
        return None

    # Step 1: Deskew
    deskewed = deskew(img)

    # Step 2: Grayscale
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    # Step 3: Border removal (on grayscale, not binary)
    no_borders = remove_borders(gray)

    # Step 4: Enhance contrast for washed images
    if use_clahe:
        enhanced = enhance_contrast(no_borders)
    else:
        enhanced = no_borders

    # Step 5: Denoise using Non-Local Means (preserves text better than bilateral)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 6: Balanced adaptive threshold - sweet spot for Arabic + English
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19, 8  # blockSize=19, constant=8 (less noise)
    )

    # Step 7: Morphological processing to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Light erosion to remove noise
    eroded = cv2.erode(thresh, kernel, iterations=1)

    # Light dilation to restore text
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    return dilated


# Main execution
if __name__ == "__main__":
    processed_image = preprocess_image("../tabali.jpg", use_clahe=True)

    if processed_image is not None:
        cv2.imwrite("../temp/final_output.jpg", processed_image)
        display("../temp/final_output.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()