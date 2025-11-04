import cv2
import numpy as np
import sys
import os
import re
from collections import Counter

def extract_exam_stats_visual(image_path):
    """
    Extract exam stats by analyzing visual patterns instead of OCR
    """
    try:
        print(f"Processing: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not load image. Check the file path.")
            return
            
        # Display image info
        height, width = img.shape[:2]
        
        
        # Method 1: Count colored circles/buttons (most reliable)
        # stats_circles = count_by_circles(img.copy())
        # print(f"Circle counting method: {stats_circles}")
        
        # Method 2: Try OCR on specific regions
        stats_ocr = extract_stats_regions(img.copy())
        
        
        # Method 3: Look for text patterns in bottom area
        stats_bottom = analyze_bottom_text(img.copy())
        
        
        # Choose best result
        best_stats = choose_best_result([stats_ocr, stats_bottom])
        
        print(f"Attempted: {best_stats['attempted']}")
        print(f"Not Attempted: {best_stats['not_attempted']}")
        print(f"Not Saved: {best_stats['not_saved']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

def count_by_circles(img):
    """
    Count question status by analyzing colored circles/buttons
    """
    stats = {'attempted': 0, 'not_attempted': 0, 'not_saved': 0}
    
    try:
        # Preprocessing
        height, width = img.shape[:2]
        # Focus on the grid area
        roi = img[int(height*0.1):int(height*0.85), int(width*0.05):int(width*0.95)]
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Refined color ranges
        color_ranges = {
            'blue': [(110, 100, 100), (130, 255, 255)],    # Attempted (blue)
            'gray': [(0, 0, 150), (180, 30, 255)],         # Not attempted (gray/white)
            'red': [
                (0, 100, 100), (10, 255, 255),             # Red lower range
                (160, 100, 100), (180, 255, 255)           # Red upper range
            ]
        }
        
        circles_found = {'blue': 0, 'gray': 0, 'red': 0}
        
        # Process each color
        for color_name, ranges in color_ranges.items():
            if color_name == 'red':
                # Special handling for red (two ranges)
                mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
            
            # Clean up the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find circles
            circles = cv2.HoughCircles(
                mask,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=25,          # Minimum distance between circles
                param1=50,
                param2=12,           # Accumulator threshold
                minRadius=10,
                maxRadius=20
            )
            
            if circles is not None:
                # Filter out weak detections
                circles = np.uint16(np.around(circles))
                valid_circles = 0
                for circle in circles[0, :]:
                    # Check if circle has enough color pixels in its area
                    x, y, r = circle
                    circle_mask = np.zeros_like(mask)
                    cv2.circle(circle_mask, (x, y), r, 255, -1)
                    if cv2.countNonZero(cv2.bitwise_and(mask, circle_mask)) > 0.5 * np.pi * r * r:
                        valid_circles += 1
                
                circles_found[color_name] = valid_circles
                
            print(f"Found {circles_found[color_name]} {color_name} circles")
        
        # Map colors to stats
        stats['attempted'] = circles_found['blue']
        stats['not_attempted'] = circles_found['gray']
        stats['not_saved'] = circles_found['red']
        
        # Validation
        total = sum(stats.values())
        if total < 20 or total > 50:  # Typical range for number of questions
            print("Warning: Unusual number of circles detected")
            
    except Exception as e:
        print(f"Circle counting error: {e}")
    
    return stats

def extract_stats_regions(img):
    """
    Focus OCR on likely text regions at bottom of image with improved parsing
    """
    stats = {'attempted': 0, 'not_attempted': 0, 'not_saved': 0}
    
    try:
        import pytesseract
        
        height, width = img.shape[:2]
        
        # Focus on bottom 20% of image where stats usually appear
        bottom_region = img[int(height * 0.8):, :]
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Scale up
        h, w = enhanced.shape
        scaled = cv2.resize(enhanced, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        
        # Extract text with custom config
        text = pytesseract.image_to_string(scaled, config='--oem 3 --psm 6')
        
        
        # --- Primary Extraction with Improved Logic ---
        # Handle "Attempted: O | Not Attempted: 35" case explicitly
        if "Attempted: O" in text and "Not Attempted:" in text:
            na_match = re.search(r'Not\s*Attempted\s*[:\-]?\s*(\d+)', text)
            if na_match:
                stats['attempted'] = 0
                stats['not_attempted'] = int(na_match.group(1))
                return stats
        
        # General case extraction
        attempted_match = re.search(r'Attempted\s*[:\-]?\s*([O0]|\d+)', text, re.IGNORECASE)
        not_attempted_match = re.search(r'Not\s*Attempted\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)
        not_saved_match = re.search(r'Not\s*Saved\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)

        if attempted_match:
            val = attempted_match.group(1)
            stats['attempted'] = 0 if val in ['O', '0'] else int(val)
        if not_attempted_match:
            stats['not_attempted'] = int(not_attempted_match.group(1))
        if not_saved_match:
            stats['not_saved'] = int(not_saved_match.group(1))

        # --- Fallback Logic ---
        numbers = re.findall(r'\d+', text)
        if stats['attempted'] == 0 and stats['not_attempted'] == 0 and len(numbers) >= 1:
            # If we found numbers but regex failed, use context
            if "Not Attempted" in text:
                stats['not_attempted'] = int(numbers[0])
            elif "Attempted" in text:
                stats['attempted'] = int(numbers[0])
                
        # Special case for when OCR misreads "0" as "O"
        if stats['attempted'] == 0 and "Attempted: O" in text:
            stats['attempted'] = 0
            if len(numbers) == 1:
                stats['not_attempted'] = int(numbers[0])

    except ImportError:
        print("pytesseract not available, skipping OCR method")
    except Exception as e:
        print(f"OCR region error: {e}")
    
    return stats

def analyze_bottom_text(img):
    """
    Look for text patterns in the bottom area without OCR
    """
    stats = {'attempted': 0, 'not_attempted': 0, 'not_saved': 0}
    
    try:
        height, width = img.shape[:2]
        
        # Get bottom portion
        bottom_img = img[int(height * 0.7):, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2GRAY)
        
        # Look for dark text on light background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (potential text)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that might be text
        text_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter for text-like shapes
            if 10 < area < 5000 and 0.2 < aspect_ratio < 5:
                text_contours.append(contour)
        
        
        
    except Exception as e:
        print(f"Bottom text analysis error: {e}")
    
    return stats

def choose_best_result(results_list):
    """
    Choose the best result from multiple methods
    """
    # Filter out results with all zeros
    valid_results = []
    for result in results_list:
        total = sum(result.values())
        if total > 0:
            valid_results.append(result)
    
    if not valid_results:
        return {'attempted': 0, 'not_attempted': 0, 'not_saved': 0}
    
    # If we have valid results, pick the one with highest total
    # (assuming more detected = more accurate)
    best = max(valid_results, key=lambda x: sum(x.values()))
    return best

def manual_input_fallback():
    """
    If automated methods fail, allow manual input
    """
    print("\nAutomated extraction failed. Please enter values manually:")
    try:
        attempted = int(input("Attempted: "))
        not_attempted = int(input("Not Attempted: "))
        not_saved = int(input("Not Saved: "))
        
        print(f"\nManual input:")
        print(f"Attempted: {attempted}")
        print(f"Not Attempted: {not_attempted}")
        print(f"Not Saved: {not_saved}")
        
    except ValueError:
        print("Invalid input. Please enter numbers only.")

# Main execution
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "issue12.png"
    
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        sys.exit(1)
    
    extract_exam_stats_visual(image_path)
    
    # Uncomment if you want manual input fallback
    # manual_input_fallback()