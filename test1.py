import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from collections import Counter
import matplotlib.pyplot as plt



class TemplateBasedOlympiadExtractor:
    def __init__(self, main_image_path, template_folder_path=None):
        self.main_image_path = main_image_path
        self.template_folder_path = template_folder_path
        self.main_image = None
        self.templates = {}
        self.min_confidence = 0.6  # Minimum matching confidence
        
    def load_main_image(self):
        """Load the main answer sheet image"""
        self.main_image = cv2.imread(self.main_image_path)
        if self.main_image is None:
            raise ValueError(f"Could not load main image from {self.main_image_path}")
        
        print(f"Main image loaded: {self.main_image.shape}")
        return True
    
    def load_templates(self, template_paths=None):
        """Load template images for A, B, C, D"""
        if template_paths:
            # Load from provided paths
            for letter, path in template_paths.items():
                template = cv2.imread(path)
                if template is not None:
                    self.templates[letter] = template
                    print(f"Template {letter} loaded from {path}")
        elif self.template_folder_path and os.path.exists(self.template_folder_path):
            # Load from folder
            for filename in os.listdir(self.template_folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    letter = filename[0].upper()  # Assume filename starts with letter
                    if letter in ['A', 'B', 'C', 'D']:
                        template_path = os.path.join(self.template_folder_path, filename)
                        template = cv2.imread(template_path)
                        if template is not None:
                            self.templates[letter] = template
                            print(f"Template {letter} loaded from {filename}")
        else:
            print("No template paths provided. Will try to extract templates from main image.")
            return False
        
        print(f"Loaded {len(self.templates)} templates: {list(self.templates.keys())}")
        return len(self.templates) > 0
    
    def detect_purple_circles(self):
        """Detect purple answer circles using advanced color segmentation"""
        # Convert to different color spaces for robust detection
        hsv = cv2.cvtColor(self.main_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(self.main_image, cv2.COLOR_BGR2LAB)
        
        # Multiple purple ranges to handle different lighting conditions
        purple_ranges = [
            # Standard purple
            (np.array([120, 100, 100]), np.array([150, 255, 255])),
            # Darker purple
            (np.array([125, 80, 80]), np.array([145, 255, 200])),
            # Broader range
            (np.array([115, 70, 70]), np.array([160, 255, 255])),
            # Blue-purple
            (np.array([110, 90, 90]), np.array([135, 255, 255]))
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in purple_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust based on your image size)
            if 300 < area < 10000:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (should be roughly square for circles)
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.3:
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.4:  # Reasonably circular
                            center_x = x + w // 2
                            center_y = y + h // 2
                            radius = max(w, h) // 2
                            
                            circles.append({
                                'center': (center_x, center_y),
                                'bbox': (x, y, w, h),
                                'radius': radius,
                                'area': area,
                                'circularity': circularity
                            })
        
        # Sort circles by position (top to bottom, left to right)
        circles = sorted(circles, key=lambda c: (c['center'][1] // 50, c['center'][0]))
        
        print(f"Detected {len(circles)} purple circles")
        return circles
    
    def extract_circle_roi(self, circle, padding=5):
        """Extract region of interest around a circle"""
        x, y, w, h = circle['bbox']
        
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(self.main_image.shape[1], x + w + padding)
        y2 = min(self.main_image.shape[0], y + h + padding)
        
        roi = self.main_image[y1:y2, x1:x2]
        return roi
    
    def preprocess_for_matching(self, image):
        """Preprocess image for better template matching"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return blurred
    
    def match_templates_multiscale(self, roi):
        """Match templates at multiple scales and rotations"""
        if not self.templates:
            return None, 0.0
        
        roi_processed = self.preprocess_for_matching(roi)
        
        best_match = None
        best_confidence = 0.0
        best_letter = None
        
        # Try multiple scales
        scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        for letter, template in self.templates.items():
            template_processed = self.preprocess_for_matching(template)
            
            for scale in scales:
                # Resize template
                new_width = int(template_processed.shape[1] * scale)
                new_height = int(template_processed.shape[0] * scale)
                
                if new_width > roi_processed.shape[1] or new_height > roi_processed.shape[0]:
                    continue
                
                scaled_template = cv2.resize(template_processed, (new_width, new_height))
                
                # Template matching
                try:
                    result = cv2.matchTemplate(roi_processed, scaled_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_confidence:
                        best_confidence = max_val
                        best_letter = letter
                        best_match = {
                            'letter': letter,
                            'confidence': max_val,
                            'scale': scale,
                            'location': max_loc
                        }
                except:
                    continue
        
        return best_match, best_confidence
    
    def extract_using_color_segmentation_fallback(self, roi):
        """Fallback method using color segmentation for white text extraction"""
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Extract white regions (letters are typically white on purple background)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 50, 255])
        white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
        
        # Find contours in white mask
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0.0
        
        # Find the largest white region (likely the letter)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 50:  # Too small to be a letter
            return None, 0.0
        
        # Extract letter region
        x, y, w, h = cv2.boundingRect(largest_contour)
        letter_roi = white_mask[y:y+h, x:x+w]
        
        # Simple pattern matching based on shape characteristics
        # This is a basic implementation - you might want to enhance this
        aspect_ratio = w / h if h > 0 else 0
        
        # Basic heuristics (you can improve these)
        if aspect_ratio < 0.6:  # Tall and narrow
            return {'letter': 'B', 'confidence': 0.5}, 0.5
        elif aspect_ratio > 1.2:  # Wide
            return {'letter': 'C', 'confidence': 0.5}, 0.5
        elif area > 200:  # Large area
            return {'letter': 'A', 'confidence': 0.5}, 0.5
        else:
            return {'letter': 'D', 'confidence': 0.5}, 0.5
    
    def process_answer_sheet(self):
        """Main processing function"""
        print("=== Template-Based Olympiad Answer Extraction ===")
        
        # Load main image
        self.load_main_image()
        
        # Load templates
        templates_loaded = self.load_templates()
        
        # Detect circles
        circles = self.detect_purple_circles()
        
        if not circles:
            print("No answer circles detected!")
            return pd.DataFrame(), []
        
        results = []
        detailed_results = []
        
        print(f"\nProcessing {len(circles)} detected circles...")
        
        for i, circle in enumerate(circles, 1):
            # Extract ROI
            roi = self.extract_circle_roi(circle)
            
            if roi.size == 0:
                continue
            
            # Try template matching first
            match_result = None
            confidence = 0.0
            
            if templates_loaded:
                match_result, confidence = self.match_templates_multiscale(roi)
            
            # Fallback to color segmentation if template matching fails
            if confidence < self.min_confidence:
                fallback_result, fallback_confidence = self.extract_using_color_segmentation_fallback(roi)
                if fallback_confidence > confidence:
                    match_result = fallback_result
                    confidence = fallback_confidence
            
            if match_result and confidence >= self.min_confidence:
                letter = match_result['letter']
                results.append({
                    'Question Number': i,
                    'Answer': letter
                })
                
                detailed_results.append({
                    'question_number': i,
                    'answer': letter,
                    'confidence': confidence,
                    'method': 'template' if templates_loaded and confidence > 0.6 else 'fallback',
                    'center': circle['center'],
                    'bbox': circle['bbox']
                })
                
                print(f"Q{i}: {letter} (confidence: {confidence:.3f})")
            else:
                print(f"Q{i}: Could not determine answer (low confidence: {confidence:.3f})")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('Question Number').reset_index(drop=True)
        
        return df, detailed_results
    
    def create_visualization(self, detailed_results):
        """Create visualization showing detected circles and matched letters"""
        if not detailed_results:
            print("No results to visualize")
            return None
        
        vis_image = self.main_image.copy()
        
        # Color coding for confidence levels
        def get_color_by_confidence(conf):
            if conf > 0.8:
                return (0, 255, 0)    # Green - high confidence
            elif conf > 0.6:
                return (0, 255, 255)  # Yellow - medium confidence
            else:
                return (0, 0, 255)    # Red - low confidence
        
        for result in detailed_results:
            center = result['center']
            answer = result['answer']
            confidence = result['confidence']
            q_num = result['question_number']
            
            color = get_color_by_confidence(confidence)
            
            # Draw circle around detected answer
            cv2.circle(vis_image, center, 30, color, 3)
            
            # Put question number and answer with confidence
            text = f"Q{q_num}:{answer}"
            conf_text = f"{confidence:.2f}"
            
            cv2.putText(vis_image, text, 
                       (center[0] - 25, center[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(vis_image, conf_text, 
                       (center[0] - 15, center[1] + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "Green: High confidence (>0.8)", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_image, "Yellow: Medium confidence (>0.6)", 
                   (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_image, "Red: Low confidence (<0.6)", 
                   (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save visualization
        output_path = 'template_matching_results.png'
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved as '{output_path}'")
        
        return vis_image
    
    def save_results(self, df, filename='template_extracted_answers.xlsx'):
        """Save results to Excel file"""
        if not df.empty:
            df.to_excel(filename, index=False)
            print(f"Results saved to {filename}")
            return filename
        else:
            print("No results to save")
            return None

# Main function to use the extractor
def extract_answers_with_templates(main_image_path, template_paths=None, template_folder=None):
    """
    Extract answers using template matching
    
    Args:
        main_image_path: Path to the main answer sheet image
        template_paths: Dictionary like {'A': 'path_to_A.png', 'B': 'path_to_B.png', ...}
        template_folder: Path to folder containing template images (A.png, B.png, C.png, D.png)
    """
    try:
        # Initialize extractor
        extractor = TemplateBasedOlympiadExtractor(main_image_path, template_folder)
        
        # Load templates if provided
        if template_paths:
            extractor.load_templates(template_paths)
        
        # Process the answer sheet
        results_df, detailed_results = extractor.process_answer_sheet()
        
        if not results_df.empty:
            print("\n=== EXTRACTED ANSWERS ===")
            print(results_df.to_string(index=False))
            
            # Save results
            extractor.save_results(results_df)
            
            # Create visualization
            extractor.create_visualization(detailed_results)
            
            # Show statistics
            print(f"\nTotal answers extracted: {len(results_df)}")
            
            if len(detailed_results) > 0:
                avg_confidence = sum(r['confidence'] for r in detailed_results) / len(detailed_results)
                print(f"Average confidence: {avg_confidence:.3f}")
                
                # Method distribution
                method_dist = Counter([r['method'] for r in detailed_results])
                print(f"Methods used: {dict(method_dist)}")
                
                # Answer distribution
                answer_dist = results_df['Answer'].value_counts().sort_index()
                print("\nAnswer distribution:")
                for answer, count in answer_dist.items():
                    print(f"  {answer}: {count} times")
            
            return results_df, detailed_results
        else:
            print("No answers could be extracted")
            return pd.DataFrame(), []
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), []

# Example usage
if __name__ == "__main__":
    # Method 1: Using template folder
    main_image = "grid.png"
    template_folder = "templates/"  # Folder containing A.png, B.png, C.png, D.png
    
    results, details = extract_answers_with_templates(main_image, template_folder=template_folder)
    
    # Method 2: Using specific template paths
    # template_paths = {
    #     'A': 'template_A.png',
    #     'B': 'template_B.png', 
    #     'C': 'template_C.png',
    #     'D': 'template_D.png'
    # }
    # results, details = extract_answers_with_templates(main_image, template_paths=template_paths)
    
    # Method 3: Without templates (fallback to color segmentation)
    # results, details = extract_answers_with_templates(main_image)