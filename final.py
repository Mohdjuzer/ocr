import cv2
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def upscale_image_only(image_path, scale_factor=2, interpolation_method='linear', output_filename=None):
    """
    Loads an image and upscales its pixel size based on user input or defaults.
    Modified version that defaults to 2x scaling with bilinear interpolation.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}. Check file format/corruption.")
        return None

    print(f"Original image dimensions: {original_image.shape[1]}x{original_image.shape[0]}")

    # Map interpolation methods
    method_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    chosen_interpolation = method_map.get(interpolation_method.lower(), cv2.INTER_LINEAR)
    chosen_interpolation_name = interpolation_method.capitalize()

    # Calculate new dimensions
    new_width = int(original_image.shape[1] * scale_factor)
    new_height = int(original_image.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)

    # Perform upscaling
    upscaled_image = cv2.resize(original_image, new_dimensions, interpolation=chosen_interpolation)

    print(f"\nUpscaled image dimensions: {upscaled_image.shape[1]}x{upscaled_image.shape[0]}")
    print(f"Interpolation method used: {chosen_interpolation_name}")

    # Save if output filename provided
    if output_filename:
        cv2.imwrite(output_filename, upscaled_image)
        print(f"Upscaled image saved as '{output_filename}'")

    return upscaled_image


class OlympiadAnswerExtractor:
    def __init__(self, main_image_path, template_folder_path=None):
        self.main_image_path = main_image_path
        self.template_folder_path = template_folder_path
        self.main_image = None
        self.enhanced_image = None
        self.templates = {} # To store loaded templates
        self.min_matching_confidence = 0.63 # Minimum correlation score for a match (0.0 to 1.0)

    def load_main_image(self):
        """
        Loads the main image using OpenCV.
        Raises a ValueError if the image cannot be loaded.
        """
        self.main_image = cv2.imread(self.main_image_path)
        if self.main_image is None:
            raise ValueError(f"Could not load main image from {self.main_image_path}. "
                             "Please check the path and file integrity.")
        print(f"Main image loaded: {self.main_image.shape}")
        return True

    def load_templates(self):
        """
        Loads template images for options (A, B, C, D) from the specified folder.
        """
        if not self.template_folder_path or not os.path.exists(self.template_folder_path):
            print("Template folder not provided or does not exist. Template matching will be skipped.")
            return False

        print(f"Loading templates from: {self.template_folder_path}")
        loaded_any = False
        for filename in os.listdir(self.template_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Assumes template filenames start with the letter (e.g., A.png, B.jpg)
                letter = filename[0].upper()
                if letter in ['A', 'B', 'C', 'D','H']: # Only load expected templates
                    template_path = os.path.join(self.template_folder_path, filename)
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) # Load templates as grayscale
                    if template is not None:
                        # Preprocess templates for better matching: CLAHE and Gaussian blur
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        template_enhanced = clahe.apply(template)
                        template_processed = cv2.GaussianBlur(template_enhanced, (3, 3), 0)

                        self.templates[letter] = template_processed
                        print(f"Template {letter} loaded and processed from {filename}")
                        loaded_any = True
                    else:
                        print(f"Warning: Could not load template {filename} from folder.")
        
        print(f"Loaded {len(self.templates)} templates: {list(self.templates.keys())}")
        return loaded_any

    def enhance_image(self):
        """
        Enhances the loaded main image for better circle detection.
        Applies grayscale conversion, CLAHE for contrast enhancement, and Gaussian blur for denoising/smoothing.
        """
        if self.main_image is None:
            print("Main image not loaded. Cannot enhance.")
            return False

        gray_image = cv2.cvtColor(self.main_image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        enhanced_contrast_image = clahe.apply(gray_image)

        # Gaussian blur for smoothing and denoising.
        self.enhanced_image = cv2.GaussianBlur(enhanced_contrast_image, (7, 7), 0)

        print("Image enhanced: Grayscale, CLAHE, and Gaussian Blur applied.")
        return True

    def detect_solid_circles(self):
        """
        Detects solid circles in the PRE-ENHANCED image using adaptive thresholding
        and contour analysis. Filters by area, aspect ratio, and circularity.
        """
        if self.enhanced_image is None:
            print("Enhanced image not available. Please call enhance_image() first.")
            return []

        image_for_thresholding = self.enhanced_image.copy()

        # Adaptive Thresholding
        thresh_image = cv2.adaptiveThreshold(image_for_thresholding, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 31, 10)

        # Morphological Operations: Clean up and solidify shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_circles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Filter by area
            if 400 < area < 4000:
                x, y, w, h = cv2.boundingRect(contour)

                aspect_ratio = w / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.3:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.6:
                            center_x = x + w // 2
                            center_y = y + h // 2
                            radius = max(w, h) // 2

                            found_circles.append({
                                'id': i,
                                'center': (center_x, center_y),
                                'bbox': (x, y, w, h),
                                'radius': radius,
                                'area': area,
                                'circularity': circularity,
                                'contour': contour
                            })

        # Sort detected circles for consistent ordering
        found_circles = sorted(found_circles, key=lambda c: (c['center'][1] // 50, c['center'][0]))

        print(f"Detected {len(found_circles)} solid circles.")
        return found_circles

    def extract_circle_roi(self, circle_data, padding=5):
        """
        Extracts the Region of Interest (ROI) around a detected circle from the ORIGINAL image.
        """
        if self.main_image is None:
            print("Original main image not loaded. Cannot extract ROI.")
            return np.array([])

        x, y, w, h = circle_data['bbox']

        # Calculate coordinates for the padded ROI
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(self.main_image.shape[1], x + w + padding)
        y2 = min(self.main_image.shape[0], y + h + padding)

        roi = self.main_image[y1:y2, x1:x2]
        return roi

    def preprocess_roi_for_matching(self, roi):
        """
        Preprocesses an extracted ROI to prepare it for template matching.
        """
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_enhanced = clahe.apply(gray_roi)
        roi_processed = cv2.GaussianBlur(roi_enhanced, (3, 3), 0)

        return roi_processed

    def match_templates_multiscale(self, roi):
        """
        Performs template matching on the given ROI using loaded templates
        at multiple scales to improve robustness to size variations.
        """
        if not self.templates:
            return None, 0.0

        roi_processed = self.preprocess_roi_for_matching(roi)

        best_match_letter = None
        best_confidence = 0.0

        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

        for letter, template in self.templates.items():
            for scale in scales:
                new_width = int(template.shape[1] * scale)
                new_height = int(template.shape[0] * scale)

                if new_width > roi_processed.shape[1] or new_height > roi_processed.shape[0] or new_width == 0 or new_height == 0:
                    continue

                scaled_template = cv2.resize(template, (new_width, new_height))

                try:
                    result = cv2.matchTemplate(roi_processed, scaled_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)

                    if max_val > best_confidence:
                        best_confidence = max_val
                        best_match_letter = letter
                except Exception as e:
                    pass

        return best_match_letter, best_confidence

    def process_and_match_answers(self):
        """
        Orchestrates the entire process: image enhancement, circle detection,
        and template matching for answers.
        """
        print("=== Olympiad Answer Extraction and Matching ===")

        self.load_main_image()
        self.enhance_image()
        self.load_templates()

        circles = self.detect_solid_circles()

        if not circles:
            print("No solid circles detected. Cannot proceed with answer matching.")
            return pd.DataFrame(), []

        extracted_answers_list = []
        detailed_results_for_vis = []

        print(f"\nProcessing {len(circles)} detected circles for template matching...")
        for i, circle in enumerate(circles):
            roi = self.extract_circle_roi(circle)

            if roi.size == 0:
                print(f"Warning: Empty ROI for circle {i+1}. Skipping.")
                continue

            best_match_letter, confidence = self.match_templates_multiscale(roi)

            if best_match_letter and confidence >= self.min_matching_confidence:
                answer = best_match_letter
                print(f"Q{i+1}: Matched '{answer}' with confidence {confidence:.3f}")
            else:
                answer = "Unsure"
                print(f"Q{i+1}: Could not confidently match. Best match was '{best_match_letter}' (Conf: {confidence:.3f})")

            extracted_answers_list.append({
                'Question Number': i + 1,
                'Answer': answer,
                'Confidence': confidence
            })

            detailed_results_for_vis.append({
                'question_number': i + 1,
                'answer': answer,
                'confidence': confidence,
                'center': circle['center'],
                'bbox': circle['bbox'],
                'radius': circle['radius']
            })

        df = pd.DataFrame(extracted_answers_list)
        if not df.empty:
            df = df.sort_values('Question Number').reset_index(drop=True)
            print("\n--- Summary of Extracted Answers ---")
            print(df.to_string(index=False))

        return df, detailed_results_for_vis

    def create_visualization(self, detailed_results, output_path='template_matching_results.png'):
        """
        Creates a visual representation of the detection and matching results on the main image.
        """
        if self.main_image is None:
            print("Main image not loaded for visualization. Skipping visualization.")
            return None
        if not detailed_results:
            print("No detailed results to visualize. Skipping visualization.")
            return None

        vis_image = self.main_image.copy()
        
        def get_color_by_confidence(conf):
            if conf > 0.8:
                return (0, 255, 0)
            elif conf > self.min_matching_confidence:
                return (0, 255, 255)
            else:
                return (0, 0, 255)

        for result in detailed_results:
            center = result['center']
            answer = result['answer']
            confidence = result['confidence']
            q_num = result['question_number']
            radius = result['radius']

            color = get_color_by_confidence(confidence)

            cv2.circle(vis_image, center, radius + 10, color, 3)

            text = f"Q{q_num}:{answer}"
            conf_text = f"{confidence:.2f}"

            cv2.putText(vis_image, text,
                        (center[0] - 25, center[1] - (radius + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis_image, conf_text,
                        (center[0] - 20, center[1] + (radius + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        legend_y_start = 30
        cv2.putText(vis_image, "Green: High confidence (>0.8)",
                    (10, legend_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_image, f"Yellow: Matched (>={self.min_matching_confidence})",
                    (10, legend_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_image, f"Red: Unsure (<{self.min_matching_confidence})",
                    (10, legend_y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved as '{output_path}'")
        return vis_image

    def save_results_to_excel(self, df, filename='extracted_answers.xlsx'):
        """
        Saves the extracted answers DataFrame to an Excel file.
        """
        if not df.empty:
            df.to_excel(filename, index=False)
            print(f"Results saved to {filename}")
            return filename
        else:
            print("No results DataFrame to save.")
            return None


def run_olympiad_answer_extraction(main_image_path, template_folder):
    """
    Modified version that first upscales the image before processing.
    """
    # First upscale the image (default 2x with bilinear interpolation)
    print("\n=== Upscaling Input Image ===")
    upscaled_image = upscale_image_only(main_image_path, scale_factor=2, interpolation_method='linear')
    
    if upscaled_image is None:
        print("Image upscaling failed. Aborting answer extraction.")
        return pd.DataFrame(), []
    
    # Save the upscaled image to a temporary file
    temp_upscaled_path = "temp_upscaled_image.png"
    cv2.imwrite(temp_upscaled_path, upscaled_image)
    print(f"Upscaled image saved temporarily to {temp_upscaled_path}")
    
    # Now process the upscaled image
    print("\n=== Starting Answer Extraction ===")
    extractor = OlympiadAnswerExtractor(temp_upscaled_path, template_folder)

    try:
        extracted_answers_df, detailed_results = extractor.process_and_match_answers()

        if not extracted_answers_df.empty:
            extractor.save_results_to_excel(extracted_answers_df)
            extractor.create_visualization(detailed_results)
            print("\nExtraction and matching process completed successfully.")
            
            # Clean up temporary file
            try:
                os.remove(temp_upscaled_path)
                print(f"Removed temporary upscaled image: {temp_upscaled_path}")
            except:
                pass
                
            return extracted_answers_df, detailed_results
        else:
            print("\nNo answers could be extracted or matched.")
            return pd.DataFrame(), []

    except Exception as e:
        print(f"An error occurred during the process: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), []


if __name__ == "__main__":
    # --- Configuration ---
    main_image_file = "clean_light.png"  # Original input image
    templates_directory = "templates/"  # Folder with A.png, B.png, etc.

    # Run the full process (will first upscale, then process)
    extracted_df, detailed_info = run_olympiad_answer_extraction(
        main_image_file,
        templates_directory
    )

    if not extracted_df.empty:
        print("\nFinal Extracted Answers DataFrame:")
        print(extracted_df)

    # Just for exam purpose    