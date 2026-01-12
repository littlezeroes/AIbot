"""
Image Difference Detection Utilities
Pinpoints and highlights differences between two images.
"""

import cv2
import numpy as np
from PIL import Image
import io
import logging


def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format (BGR)."""
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    numpy_image = np.array(pil_image)
    # Convert RGB to BGR for OpenCV
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert OpenCV image (BGR) to PIL Image."""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def resize_to_match(img1, img2):
    """Resize img2 to match img1's dimensions."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)

    return img1, img2


def create_diff_image(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO,
                      sensitivity: int = 30, min_area: int = 100) -> io.BytesIO:
    """
    Compare two images and create a difference visualization with bounding boxes.

    Args:
        image1_bytes: First image (DEV - to be checked)
        image2_bytes: Second image (DESIGN - reference)
        sensitivity: Threshold for detecting differences (lower = more sensitive)
        min_area: Minimum contour area to be considered a difference

    Returns:
        BytesIO containing the annotated difference image
    """
    try:
        # Load images
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes)
        pil_img2 = Image.open(image2_bytes)

        # Convert to OpenCV format
        cv_img1 = pil_to_cv2(pil_img1)
        cv_img2 = pil_to_cv2(pil_img2)

        # Resize to match dimensions
        cv_img1, cv_img2 = resize_to_match(cv_img1, cv_img2)

        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Apply Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # Threshold the difference
        _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)

        # Dilate to fill gaps
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours (areas of difference)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create output image (copy of first image)
        output = cv_img1.copy()

        # Draw bounding boxes around differences
        diff_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                diff_count += 1
                x, y, w, h = cv2.boundingRect(contour)

                # Add padding
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(output.shape[1] - x, w + 2 * padding)
                h = min(output.shape[0] - y, h + 2 * padding)

                # Draw red rectangle
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Add label number
                cv2.putText(output, str(diff_count), (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add summary text
        h, w = output.shape[:2]
        cv2.putText(output, f"Found {diff_count} difference(s)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Convert back to PIL and save to BytesIO
        pil_output = cv2_to_pil(output)
        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        logging.info(f"Image diff completed: found {diff_count} differences")
        return output_bytes, diff_count

    except Exception as e:
        logging.error(f"Error creating diff image: {e}")
        return None, 0


def create_overlay_image(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO,
                         alpha: float = 0.5) -> io.BytesIO:
    """
    Create an overlay blend of two images to visually spot misalignments.

    Args:
        image1_bytes: First image (DEV)
        image2_bytes: Second image (DESIGN)
        alpha: Blend factor (0.5 = equal mix)

    Returns:
        BytesIO containing the blended overlay image
    """
    try:
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes)
        pil_img2 = Image.open(image2_bytes)

        # Convert to OpenCV
        cv_img1 = pil_to_cv2(pil_img1)
        cv_img2 = pil_to_cv2(pil_img2)

        # Resize to match
        cv_img1, cv_img2 = resize_to_match(cv_img1, cv_img2)

        # Blend images
        blended = cv2.addWeighted(cv_img1, alpha, cv_img2, 1 - alpha, 0)

        # Convert back to PIL
        pil_output = cv2_to_pil(blended)
        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        return output_bytes

    except Exception as e:
        logging.error(f"Error creating overlay image: {e}")
        return None


def create_heatmap_diff(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO) -> io.BytesIO:
    """
    Create a heatmap showing intensity of differences between images.
    Red = major differences, Blue = minor differences.

    Args:
        image1_bytes: First image (DEV)
        image2_bytes: Second image (DESIGN)

    Returns:
        BytesIO containing the heatmap image
    """
    try:
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes)
        pil_img2 = Image.open(image2_bytes)

        # Convert to OpenCV
        cv_img1 = pil_to_cv2(pil_img1)
        cv_img2 = pil_to_cv2(pil_img2)

        # Resize to match
        cv_img1, cv_img2 = resize_to_match(cv_img1, cv_img2)

        # Convert to grayscale
        gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)

        # Compute difference
        diff = cv2.absdiff(gray1, gray2)

        # Apply color map (JET: blue -> green -> yellow -> red)
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

        # Blend with original for context
        output = cv2.addWeighted(cv_img1, 0.3, heatmap, 0.7, 0)

        # Convert back to PIL
        pil_output = cv2_to_pil(output)
        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        return output_bytes

    except Exception as e:
        logging.error(f"Error creating heatmap: {e}")
        return None


def create_side_by_side_diff(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO,
                              sensitivity: int = 30, min_area: int = 100) -> io.BytesIO:
    """
    Create a side-by-side comparison with differences highlighted on both images.

    Args:
        image1_bytes: First image (DEV)
        image2_bytes: Second image (DESIGN)
        sensitivity: Threshold for detecting differences
        min_area: Minimum contour area

    Returns:
        BytesIO containing the side-by-side comparison
    """
    try:
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes)
        pil_img2 = Image.open(image2_bytes)

        # Convert to OpenCV
        cv_img1 = pil_to_cv2(pil_img1)
        cv_img2 = pil_to_cv2(pil_img2)

        # Resize to match
        cv_img1, cv_img2 = resize_to_match(cv_img1, cv_img2)

        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)

        # Compute difference
        diff = cv2.absdiff(gray1, gray2)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw on both images
        output1 = cv_img1.copy()
        output2 = cv_img2.copy()

        diff_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                diff_count += 1
                x, y, w, h = cv2.boundingRect(contour)

                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(output1.shape[1] - x, w + 2 * padding)
                h = min(output1.shape[0] - y, h + 2 * padding)

                # Red on DEV, Blue on DESIGN
                cv2.rectangle(output1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(output2, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.putText(output1, str(diff_count), (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(output2, str(diff_count), (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Add labels
        cv2.putText(output1, "DEV", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(output2, "DESIGN", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Create side-by-side image
        h, w = output1.shape[:2]
        combined = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        combined[:, :w] = output1
        combined[:, w+10:] = output2

        # Convert back to PIL
        pil_output = cv2_to_pil(combined)
        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        return output_bytes, diff_count

    except Exception as e:
        logging.error(f"Error creating side-by-side diff: {e}")
        return None, 0
