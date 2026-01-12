"""
Image Difference Detection Utilities
Pinpoints and highlights differences between two images.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import logging
from skimage.metrics import structural_similarity as ssim


def draw_bugs_on_image(image_bytes: io.BytesIO, bugs: list) -> io.BytesIO:
    """
    Draw bounding boxes on image based on bug coordinates from Claude.

    Args:
        image_bytes: The DEV image to annotate
        bugs: List of bugs with x, y, w, h (0.0-1.0 scale) and bug description

    Returns:
        BytesIO containing the annotated image
    """
    try:
        image_bytes.seek(0)
        pil_img = Image.open(image_bytes).convert('RGB')
        img_width, img_height = pil_img.size

        # Convert to OpenCV
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Colors for different bug types
        colors = {
            'SPACING': (0, 165, 255),    # Orange
            'ALIGNMENT': (0, 0, 255),     # Red
            'COLOR': (255, 0, 255),       # Magenta
            'COMPONENT': (255, 0, 0),     # Blue
        }
        default_color = (0, 0, 255)  # Red

        for i, bug in enumerate(bugs):
            # Get coordinates (0.0-1.0 scale)
            x = bug.get('x', 0)
            y = bug.get('y', 0)
            w = bug.get('w', 0.1)
            h = bug.get('h', 0.1)
            bug_type = bug.get('type', 'UNKNOWN')
            description = bug.get('bug', 'Bug')

            # Convert to pixel coordinates
            x1 = int(x * img_width)
            y1 = int(y * img_height)
            x2 = int((x + w) * img_width)
            y2 = int((y + h) * img_height)

            # Ensure valid bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(x1 + 10, min(x2, img_width))
            y2 = max(y1 + 10, min(y2, img_height))

            # Get color
            color = colors.get(bug_type, default_color)

            # Draw rectangle
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 3)

            # Draw bug number
            label = f"#{i+1}"
            cv2.putText(cv_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert back to PIL
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_output = Image.fromarray(rgb_img)

        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        return output_bytes

    except Exception as e:
        logging.error(f"Error drawing bugs on image: {e}")
        return None


def format_bug_report(bugs: list) -> str:
    """
    Format bugs list into readable text report with fun comments.
    """
    import random

    if not bugs:
        comments = [
            "✅ 0 bug! Dev hôm nay uống thuốc gì ngon vậy? 🔥",
            "✅ Perfect! Cho dev tăng lương đi sếp ơi! 💰",
            "✅ Không có bug! Dev đẹp trai/xinh gái quá đi! 😍",
            "✅ Clean như nước suối! Dev hôm nay ngủ đủ giấc hen 👏",
        ]
        return random.choice(comments)

    report = []
    for i, bug in enumerate(bugs):
        bug_type = bug.get('type', 'UNKNOWN')
        description = bug.get('bug', 'Không rõ')
        report.append(f"🔴 #{i+1} [{bug_type}]: {description}")

    report.append(f"\n📊 Tổng: {len(bugs)} lỗi")

    # Add fun comment based on bug count
    if len(bugs) > 5:
        comments = [
            "💀 Dev ơi về học lại code đi!",
            "🔥 Đuổi việc hết cho rồi!",
            "😭 Designer đang khóc trong toilet!",
            "👀 Mắt dev để ở nhà hả?",
        ]
    elif len(bugs) > 2:
        comments = [
            "😏 Gần ngon rồi, cố lên dev ơi!",
            "🤔 Tạm được, nhưng cần fix gấp!",
            "💪 Còn vài chỗ thôi, ráng lên!",
        ]
    else:
        comments = [
            "👍 Ít bug, dev có tiến bộ đó!",
            "😎 Gần perfect rồi, fix nốt đi!",
            "✨ Còn tí xíu thôi!",
        ]

    report.append(random.choice(comments))
    return "\n".join(report)


def analyze_color_differences(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO) -> str:
    """
    Analyze color differences between two images.
    Returns a text description of color differences for Claude.
    """
    try:
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes).convert('RGB')
        pil_img2 = Image.open(image2_bytes).convert('RGB')

        cv_img1 = cv2.cvtColor(np.array(pil_img1), cv2.COLOR_RGB2BGR)
        cv_img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)

        # Resize to match
        h1, w1 = cv_img1.shape[:2]
        h2, w2 = cv_img2.shape[:2]
        if (h1, w1) != (h2, w2):
            cv_img2 = cv2.resize(cv_img2, (w1, h1), interpolation=cv2.INTER_AREA)

        # Find color differences (not just grayscale)
        diff = cv2.absdiff(cv_img1, cv_img2)

        # Sum across color channels to find any color difference
        diff_sum = np.sum(diff, axis=2)

        # Threshold to find significant color differences
        _, mask = cv2.threshold(diff_sum.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

        # Find contours of different colored regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_diffs = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)

                # Get average color in this region from both images
                region1 = cv_img1[y:y+h, x:x+w]
                region2 = cv_img2[y:y+h, x:x+w]

                avg_color1 = np.mean(region1, axis=(0,1)).astype(int)
                avg_color2 = np.mean(region2, axis=(0,1)).astype(int)

                # Convert to position percentage
                px = round(x / w1, 2)
                py = round(y / h1, 2)

                # BGR to RGB for human readable
                rgb1 = f"rgb({avg_color1[2]},{avg_color1[1]},{avg_color1[0]})"
                rgb2 = f"rgb({avg_color2[2]},{avg_color2[1]},{avg_color2[0]})"

                color_diffs.append(f"- Vị trí ({px}, {py}): DEV={rgb1}, DESIGN={rgb2}")

        if color_diffs:
            return "PHÁT HIỆN KHÁC BIỆT MÀU SẮC:\n" + "\n".join(color_diffs[:10])  # Max 10
        return ""

    except Exception as e:
        logging.error(f"Error analyzing colors: {e}")
        return ""


def create_overlay_for_comparison(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO) -> io.BytesIO:
    """
    Create an overlay image combining DEV and DESIGN for easier comparison.
    DEV in red channel, DESIGN in green channel - differences show as color shifts.

    Returns:
        BytesIO containing the overlay image
    """
    try:
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes).convert('RGB')
        pil_img2 = Image.open(image2_bytes).convert('RGB')

        # Convert to OpenCV
        cv_img1 = cv2.cvtColor(np.array(pil_img1), cv2.COLOR_RGB2BGR)
        cv_img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)

        # Resize to match
        h1, w1 = cv_img1.shape[:2]
        h2, w2 = cv_img2.shape[:2]
        if (h1, w1) != (h2, w2):
            cv_img2 = cv2.resize(cv_img2, (w1, h1), interpolation=cv2.INTER_AREA)

        # Method: Blend with difference highlight
        # Where images differ, you'll see color shifts
        gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)

        # Create diff mask
        diff = cv2.absdiff(gray1, gray2)
        _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

        # Create overlay: base is DEV, highlight differences in red
        overlay = cv_img1.copy()
        overlay[mask > 0] = [0, 0, 255]  # Red for differences

        # Blend overlay with original
        result = cv2.addWeighted(cv_img1, 0.7, overlay, 0.3, 0)

        # Add label
        cv2.putText(result, "RED = Khac biet", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Convert back to PIL
        rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_output = Image.fromarray(rgb_result)

        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        return output_bytes

    except Exception as e:
        logging.error(f"Error creating overlay: {e}")
        return None


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


def create_ssim_diff(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO) -> tuple:
    """
    Create SSIM (Structural Similarity Index) difference visualization.
    SSIM is better than pixel-by-pixel because it considers:
    - Luminance (brightness)
    - Contrast
    - Structure

    This catches meaningful visual differences while ignoring noise.

    Args:
        image1_bytes: DEV image (to check)
        image2_bytes: DESIGN image (reference)

    Returns:
        Tuple of (diff_image_bytes, ssim_score, diff_regions)
        - diff_image_bytes: Visual diff highlighting structural differences
        - ssim_score: 0-1 similarity score (1 = identical)
        - diff_regions: List of regions with low similarity
    """
    try:
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes).convert('RGB')
        pil_img2 = Image.open(image2_bytes).convert('RGB')

        cv_img1 = cv2.cvtColor(np.array(pil_img1), cv2.COLOR_RGB2BGR)
        cv_img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)

        # Resize to match
        h1, w1 = cv_img1.shape[:2]
        h2, w2 = cv_img2.shape[:2]
        if (h1, w1) != (h2, w2):
            cv_img2 = cv2.resize(cv_img2, (w1, h1), interpolation=cv2.INTER_AREA)

        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)

        # Compute SSIM with full output
        # full=True returns the SSIM image showing local differences
        score, ssim_image = ssim(gray1, gray2, full=True)

        # Convert SSIM image to 0-255 range
        # SSIM values are 0-1 where 1=identical, so we invert: 1-ssim shows differences
        diff_map = (1 - ssim_image) * 255
        diff_map = diff_map.astype(np.uint8)

        # Threshold to find significant structural differences
        _, thresh = cv2.threshold(diff_map, 50, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours of different regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create colorful diff visualization
        # Use JET colormap: blue (similar) -> red (different)
        diff_colored = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)

        # Blend with original DEV image for context
        output = cv2.addWeighted(cv_img1, 0.5, diff_colored, 0.5, 0)

        # Draw bounding boxes around significant differences
        diff_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)

                # Add padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(w1 - x, w + 2 * padding)
                h = min(h1 - y, h + 2 * padding)

                # Draw white box with black outline for visibility
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 0), 4)
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 2)

                # Store region info (normalized 0-1)
                diff_regions.append({
                    'x': round(x / w1, 3),
                    'y': round(y / h1, 3),
                    'w': round(w / w1, 3),
                    'h': round(h / h1, 3)
                })

        # Add legend
        cv2.putText(output, f"SSIM Score: {score:.2%}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output, "RED = Khac biet | BLUE = Giong", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Found {len(diff_regions)} region(s)", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert to PIL and save
        rgb_output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_output = Image.fromarray(rgb_output)

        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        logging.info(f"SSIM analysis complete: score={score:.2%}, regions={len(diff_regions)}")

        return output_bytes, score, diff_regions

    except Exception as e:
        logging.error(f"Error creating SSIM diff: {e}")
        return None, 0, []


def create_edge_comparison(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO) -> tuple:
    """
    Create edge detection comparison to highlight alignment differences.
    Uses Canny edge detection to find element boundaries.
    Overlays edges from both images: GREEN=DEV, RED=DESIGN, WHITE=match

    This helps detect:
    - Vertical alignment issues (elements not lined up vertically)
    - Horizontal alignment issues
    - Padding differences on left/right sides
    - Element boundary mismatches

    Returns:
        Tuple of (edge_diff_bytes, alignment_info)
    """
    try:
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes).convert('RGB')
        pil_img2 = Image.open(image2_bytes).convert('RGB')

        cv_img1 = cv2.cvtColor(np.array(pil_img1), cv2.COLOR_RGB2BGR)
        cv_img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)

        # Resize to match
        h1, w1 = cv_img1.shape[:2]
        h2, w2 = cv_img2.shape[:2]
        if (h1, w1) != (h2, w2):
            cv_img2 = cv2.resize(cv_img2, (w1, h1), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
        gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)

        # Canny edge detection
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)

        # Dilate edges slightly for better visibility
        kernel = np.ones((2, 2), np.uint8)
        edges1 = cv2.dilate(edges1, kernel, iterations=1)
        edges2 = cv2.dilate(edges2, kernel, iterations=1)

        # Create color overlay image
        # Start with dark background
        output = np.zeros((h1, w1, 3), dtype=np.uint8)
        output[:] = (30, 30, 30)  # Dark gray background

        # GREEN = DEV edges only (not in DESIGN)
        dev_only = cv2.bitwise_and(edges1, cv2.bitwise_not(edges2))
        output[dev_only > 0] = (0, 255, 0)  # Green

        # RED = DESIGN edges only (not in DEV) - these are MISSING in dev
        design_only = cv2.bitwise_and(edges2, cv2.bitwise_not(edges1))
        output[design_only > 0] = (0, 0, 255)  # Red

        # WHITE = matching edges (both have)
        matching = cv2.bitwise_and(edges1, edges2)
        output[matching > 0] = (255, 255, 255)  # White

        # Analyze vertical alignment by checking left/right edges
        # Split image into left and right halves
        mid_x = w1 // 2
        left_dev = edges1[:, :mid_x]
        left_design = edges2[:, :mid_x]
        right_dev = edges1[:, mid_x:]
        right_design = edges2[:, mid_x:]

        # Count edge differences in left/right sides
        left_diff = np.sum(cv2.absdiff(left_dev, left_design)) / 255
        right_diff = np.sum(cv2.absdiff(right_dev, right_design)) / 255

        # Detect vertical lines (for alignment check)
        # Use Hough Line Transform on both images
        lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

        vertical_diff_count = 0
        if lines1 is not None and lines2 is not None:
            # Extract x positions of vertical lines
            def get_vertical_x_positions(lines):
                positions = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is mostly vertical (angle close to 90 degrees)
                    if abs(x2 - x1) < 10:  # Nearly vertical
                        positions.append((x1 + x2) // 2)
                return sorted(set(positions))

            v_pos1 = get_vertical_x_positions(lines1)
            v_pos2 = get_vertical_x_positions(lines2)

            # Compare vertical line positions
            for pos1 in v_pos1:
                has_match = any(abs(pos1 - pos2) < 5 for pos2 in v_pos2)
                if not has_match:
                    vertical_diff_count += 1

        # Build alignment info
        alignment_info = {
            'left_padding_diff': int(left_diff),
            'right_padding_diff': int(right_diff),
            'vertical_alignment_issues': vertical_diff_count,
            'total_edge_diff': int(np.sum(cv2.absdiff(edges1, edges2)) / 255)
        }

        # Add legend and info to output
        cv2.putText(output, "EDGE COMPARISON", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(output, "WHITE=Match | GREEN=DEV only | RED=DESIGN only", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(output, f"Left padding diff: {left_diff:.0f}px", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(output, f"Right padding diff: {right_diff:.0f}px", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(output, f"Vertical align issues: {vertical_diff_count}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw center line for reference
        cv2.line(output, (mid_x, 0), (mid_x, h1), (100, 100, 100), 1)

        # Convert to PIL and save
        rgb_output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_output = Image.fromarray(rgb_output)

        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        logging.info(f"Edge comparison: left_diff={left_diff:.0f}, right_diff={right_diff:.0f}, vertical_issues={vertical_diff_count}")

        return output_bytes, alignment_info

    except Exception as e:
        logging.error(f"Error creating edge comparison: {e}")
        return None, {}


def create_pixelmatch_diff(image1_bytes: io.BytesIO, image2_bytes: io.BytesIO,
                           threshold: float = 0.1) -> tuple:
    """
    Use pixelmatch for accurate pixel-level comparison with smart region grouping.

    Args:
        image1_bytes: DEV image
        image2_bytes: DESIGN image
        threshold: Matching threshold 0-1 (lower = more sensitive). Default 0.1

    Returns:
        Tuple of (diff_image_bytes, diff_count, grouped_regions, shift_analysis)
        - diff_image_bytes: Visual diff image
        - diff_count: Total different pixels
        - grouped_regions: List of grouped difference regions
        - shift_analysis: Analysis of vertical/horizontal shifts
    """
    from pixelmatch import pixelmatch

    try:
        image1_bytes.seek(0)
        image2_bytes.seek(0)

        pil_img1 = Image.open(image1_bytes).convert('RGBA')
        pil_img2 = Image.open(image2_bytes).convert('RGBA')

        # Resize to match
        w1, h1 = pil_img1.size
        w2, h2 = pil_img2.size
        if (w1, h1) != (w2, h2):
            pil_img2 = pil_img2.resize((w1, h1), Image.Resampling.LANCZOS)

        # Create output diff image
        diff_img = Image.new('RGBA', (w1, h1))

        # Run pixelmatch with STRICT settings
        diff_count = pixelmatch(
            pil_img1, pil_img2, diff_img,
            threshold=threshold,
            includeAA=True,  # Also catch anti-aliasing differences
            alpha=0.1
        )

        logging.info(f"Pixelmatch found {diff_count} different pixels")

        # Only return no-diff if TRULY identical (0 pixels different)
        if diff_count == 0:
            logging.info("Images are truly identical")
            return None, 0, [], {}

        # Convert diff image to numpy for region analysis
        diff_array = np.array(diff_img)

        # Create binary mask of differences (red pixels in diff)
        # Pixelmatch outputs red for differences
        red_channel = diff_array[:, :, 0]
        green_channel = diff_array[:, :, 1]
        diff_mask = ((red_channel > 200) & (green_channel < 100)).astype(np.uint8) * 255

        # Group nearby differences into regions using morphological operations
        # Use smaller kernel to preserve individual differences better
        kernel = np.ones((5, 5), np.uint8)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
        diff_mask = cv2.dilate(diff_mask, kernel, iterations=1)  # Less dilation

        # Find contours (regions of difference)
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Group and analyze regions
        grouped_regions = []
        all_y_positions = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 30:  # Lower minimum to catch small differences
                x, y, w, h = cv2.boundingRect(contour)

                # Store region info
                region = {
                    'x': round(x / w1, 3),
                    'y': round(y / h1, 3),
                    'w': round(w / w1, 3),
                    'h': round(h / h1, 3),
                    'area': area,
                    'pixel_x': x,
                    'pixel_y': y,
                    'pixel_w': w,
                    'pixel_h': h
                }
                grouped_regions.append(region)
                all_y_positions.append(y)

        # Sort regions by y position (top to bottom)
        grouped_regions.sort(key=lambda r: r['pixel_y'])

        # Analyze for vertical shift pattern (cascade effect)
        shift_analysis = analyze_shift_pattern(grouped_regions, h1)

        # Create annotated diff image
        cv_diff = cv2.cvtColor(np.array(diff_img.convert('RGB')), cv2.COLOR_RGB2BGR)

        for i, region in enumerate(grouped_regions):
            x, y = region['pixel_x'], region['pixel_y']
            w, h = region['pixel_w'], region['pixel_h']

            # Color based on whether it's part of a shift pattern
            if shift_analysis.get('is_cascade'):
                if i == 0:
                    color = (0, 0, 255)  # Red for root cause
                else:
                    color = (0, 165, 255)  # Orange for cascade effect
            else:
                color = (0, 0, 255)  # Red

            cv2.rectangle(cv_diff, (x, y), (x + w, y + h), color, 2)
            cv2.putText(cv_diff, f"#{i+1}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add summary info
        cv2.putText(cv_diff, f"Diff: {diff_count}px ({diff_percentage:.2f}%)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(cv_diff, f"Regions: {len(grouped_regions)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if shift_analysis.get('is_cascade'):
            cv2.putText(cv_diff, "DETECTED: Vertical shift cascade", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Convert back to PIL and save
        rgb_output = cv2.cvtColor(cv_diff, cv2.COLOR_BGR2RGB)
        pil_output = Image.fromarray(rgb_output)

        output_bytes = io.BytesIO()
        pil_output.save(output_bytes, format='PNG')
        output_bytes.seek(0)

        return output_bytes, diff_count, grouped_regions, shift_analysis

    except Exception as e:
        logging.error(f"Error in pixelmatch diff: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, [], {}


def analyze_shift_pattern(regions: list, img_height: int) -> dict:
    """
    Analyze if differences form a vertical shift pattern (cascade effect).

    When one element has spacing issue, everything below shifts.
    This detects that pattern to report only the ROOT CAUSE.

    Returns:
        dict with:
        - is_cascade: True if vertical cascade detected
        - root_region_idx: Index of the likely root cause region
        - shift_direction: 'down' or 'up'
        - estimated_shift_px: Estimated pixel shift amount
    """
    if len(regions) < 2:
        return {'is_cascade': False}

    # Check if regions are stacked vertically
    y_positions = [r['pixel_y'] for r in regions]
    x_positions = [r['pixel_x'] for r in regions]

    # Calculate vertical spread vs horizontal spread
    y_spread = max(y_positions) - min(y_positions) if y_positions else 0
    x_spread = max(x_positions) - min(x_positions) if x_positions else 0

    # If regions span more than 30% of image height and are vertically aligned
    if y_spread > img_height * 0.3:
        # Check if x positions are similar (vertically aligned)
        x_variance = np.std(x_positions) if len(x_positions) > 1 else 0

        # If x positions are similar (low variance), it's likely a vertical shift
        if x_variance < 50:  # Threshold for "same column"
            # The topmost region is likely the root cause
            return {
                'is_cascade': True,
                'root_region_idx': 0,  # First region (topmost) is root cause
                'shift_direction': 'down',
                'estimated_shift_px': y_spread // len(regions),
                'affected_regions': len(regions) - 1
            }

    # Check for horizontal shift pattern
    if x_spread > 100 and y_spread < img_height * 0.2:
        return {
            'is_cascade': True,
            'root_region_idx': 0,
            'shift_direction': 'horizontal',
            'estimated_shift_px': x_spread // len(regions),
            'affected_regions': len(regions) - 1
        }

    return {'is_cascade': False}


def merge_overlapping_regions(regions: list, overlap_threshold: float = 0.3) -> list:
    """
    Merge regions that overlap significantly to avoid duplicate reports.

    Args:
        regions: List of region dicts with x, y, w, h (normalized 0-1)
        overlap_threshold: Minimum overlap ratio to merge (0-1)

    Returns:
        List of merged regions
    """
    if len(regions) <= 1:
        return regions

    merged = []
    used = set()

    for i, r1 in enumerate(regions):
        if i in used:
            continue

        # Start with this region
        merged_region = r1.copy()
        used.add(i)

        # Find overlapping regions
        for j, r2 in enumerate(regions):
            if j in used or j == i:
                continue

            # Calculate overlap
            x1_min, x1_max = r1['x'], r1['x'] + r1['w']
            y1_min, y1_max = r1['y'], r1['y'] + r1['h']
            x2_min, x2_max = r2['x'], r2['x'] + r2['w']
            y2_min, y2_max = r2['y'], r2['y'] + r2['h']

            # Intersection
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

            intersection = x_overlap * y_overlap
            area1 = r1['w'] * r1['h']
            area2 = r2['w'] * r2['h']

            # Check if overlap is significant
            overlap_ratio = intersection / min(area1, area2) if min(area1, area2) > 0 else 0

            if overlap_ratio > overlap_threshold:
                # Merge regions - expand to cover both
                new_x = min(r1['x'], r2['x'])
                new_y = min(r1['y'], r2['y'])
                new_x2 = max(x1_max, x2_max)
                new_y2 = max(y1_max, y2_max)

                merged_region = {
                    'x': new_x,
                    'y': new_y,
                    'w': new_x2 - new_x,
                    'h': new_y2 - new_y
                }
                used.add(j)

        merged.append(merged_region)

    return merged
