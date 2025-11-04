import cv2
import numpy as np
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt


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
                if letter in ['A', 'B', 'C', 'D']: # Only load expected templates
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

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_contrast_image = clahe.apply(gray_image)

        # Gaussian blur for smoothing and denoising.
        # This prepares the image for robust thresholding.
        self.enhanced_image = cv2.GaussianBlur(enhanced_contrast_image, (7, 7), 0) # Adjust kernel size

        print("Image enhanced: Grayscale, CLAHE, and Gaussian Blur applied.")

        # --- Debugging Visualization: Enhanced Image ---
        # Uncomment to see the enhanced image before thresholding
        # cv2.imshow("Enhanced Image (Grayscale + CLAHE + Gaussian Blur)", self.enhanced_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # -----------------------------------------------
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

        # Adaptive Thresholding: Experiment with blockSize and C.
        # Larger blockSize and smaller C might be needed for blurred/faint circles.
        thresh_image = cv2.adaptiveThreshold(image_for_thresholding, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 31, 10)

        # --- Debugging Visualization: Thresholded Image ---
        # Uncomment to visualize the thresholded image. Crucial for tuning `adaptiveThreshold` parameters.
        # You want circles as solid white blobs against a black background.
        # cv2.imshow("Thresholded Image for Circle Detection (Adjust me!)", thresh_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # --------------------------------------------------

        # Morphological Operations: Clean up and solidify shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found_circles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Filter by area: Adjust based on your image's circle sizes.
            if 400 < area < 4000:
                x, y, w, h = cv2.boundingRect(contour)

                aspect_ratio = w / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.3: # Relaxed for slight distortions
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        # Slightly relaxed circularity threshold for blurred/imperfect circles
                        if circularity > 0.6: # Experiment with this value (e.g., 0.5 to 0.8)
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

        # Sort detected circles for consistent ordering (e.g., for question numbering)
        found_circles = sorted(found_circles, key=lambda c: (c['center'][1] // 50, c['center'][0]))

        print(f"Detected {len(found_circles)} solid circles.")
        return found_circles

    def extract_circle_roi(self, circle_data, padding=5):
        """
        Extracts the Region of Interest (ROI) around a detected circle from the ORIGINAL image.
        Adds padding to ensure the entire circle and its content are captured.
        """
        if self.main_image is None:
            print("Original main image not loaded. Cannot extract ROI.")
            return np.array([])

        x, y, w, h = circle_data['bbox']

        # Calculate coordinates for the padded ROI, ensuring they stay within image bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(self.main_image.shape[1], x + w + padding)
        y2 = min(self.main_image.shape[0], y + h + padding)

        # Extract ROI from the original color image (or enhanced if preferred, but original is better for template matching)
        roi = self.main_image[y1:y2, x1:x2]
        return roi

    def preprocess_roi_for_matching(self, roi):
        """
        Preprocesses an extracted ROI to prepare it for template matching.
        Converts to grayscale, applies CLAHE, and Gaussian blur, similar to templates.
        """
        if len(roi.shape) == 3: # If ROI is still color
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_enhanced = clahe.apply(gray_roi)
        roi_processed = cv2.GaussianBlur(roi_enhanced, (3, 3), 0) # Smaller kernel for ROI to preserve detail

        return roi_processed

    def match_templates_multiscale(self, roi):
        """
        Performs template matching on the given ROI using loaded templates
        at multiple scales to improve robustness to size variations.
        Returns the best matching letter and its confidence.
        """
        if not self.templates:
            return None, 0.0 # No templates to match against

        roi_processed = self.preprocess_roi_for_matching(roi)

        best_match_letter = None
        best_confidence = 0.0

        # Define a range of scales to try for template matching.
        # This is important if the size of letters in circles varies slightly.
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] # Experiment with these scales

        for letter, template in self.templates.items():
            for scale in scales:
                new_width = int(template.shape[1] * scale)
                new_height = int(template.shape[0] * scale)

                # Ensure scaled template is smaller than or equal to ROI
                if new_width > roi_processed.shape[1] or new_height > roi_processed.shape[0] or new_width == 0 or new_height == 0:
                    continue

                scaled_template = cv2.resize(template, (new_width, new_height))

                try:
                    # Perform template matching using normalized cross-correlation
                    # TM_CCOEFF_NORMED gives a score from -1 (perfect mismatch) to 1 (perfect match)
                    result = cv2.matchTemplate(roi_processed, scaled_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result) # Get the maximum correlation value

                    if max_val > best_confidence:
                        best_confidence = max_val
                        best_match_letter = letter
                except Exception as e:
                    # print(f"Error matching {letter} at scale {scale}: {e}")
                    pass # Ignore errors, typically dimension mismatches

        return best_match_letter, best_confidence

    def process_and_match_answers(self):
        """
        Orchestrates the entire process: image enhancement, circle detection,
        and template matching for answers.
        """
        print("=== Olympiad Answer Extraction and Matching ===")

        self.load_main_image()
        self.enhance_image()
        self.load_templates() # Load templates after enhancing main image (can be done before or after)

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

            # Assign answer only if confidence is above threshold
            if best_match_letter and confidence >= self.min_matching_confidence:
                answer = best_match_letter
                print(f"Q{i+1}: Matched '{answer}' with confidence {confidence:.3f}")
            else:
                answer = "Unsure" # Or a placeholder for unmatched circles
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
        Draws circles around detected answers, labels them, and color-codes by confidence.
        """
        if self.main_image is None:
            print("Main image not loaded for visualization. Skipping visualization.")
            return None
        if not detailed_results:
            print("No detailed results to visualize. Skipping visualization.")
            return None

        vis_image = self.main_image.copy()
        
        # Helper function to determine circle color based on confidence level
        def get_color_by_confidence(conf):
            if conf > 0.8:
                return (0, 255, 0)     # Green for high confidence (BGR format)
            elif conf > self.min_matching_confidence: # Above threshold but not very high
                return (0, 255, 255)   # Yellow for medium confidence
            else:
                return (0, 0, 255)     # Red for low confidence (below threshold or no match)

        for result in detailed_results:
            center = result['center']
            answer = result['answer']
            confidence = result['confidence']
            q_num = result['question_number']
            radius = result['radius']

            color = get_color_by_confidence(confidence)

            # Draw a circle around the detected answer region
            cv2.circle(vis_image, center, radius + 10, color, 3)

            # Add question number and detected answer text
            text = f"Q{q_num}:{answer}"
            conf_text = f"{confidence:.2f}"

            cv2.putText(vis_image, text,
                        (center[0] - 25, center[1] - (radius + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis_image, conf_text,
                        (center[0] - 20, center[1] + (radius + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Add a legend
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

# Main function to run the process
def run_olympiad_answer_extraction(main_image_path, template_folder):
    """
    Orchestrates the entire answer extraction and matching process.
    """
    extractor = OlympiadAnswerExtractor(main_image_path, template_folder)

    try:
        extracted_answers_df, detailed_results = extractor.process_and_match_answers()

        if not extracted_answers_df.empty:
            extractor.save_results_to_excel(extracted_answers_df)
            extractor.create_visualization(detailed_results)
            print("\nExtraction and matching process completed successfully.")
            return extracted_answers_df, detailed_results
        else:
            print("\nNo answers could be extracted or matched.")
            return pd.DataFrame(), []

    except Exception as e:
        print(f"An error occurred during the process: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), []

# Example Usage
if __name__ == "__main__":
    # --- Configuration ---
    # main_image_file = "_1746710611.png" # Ensure this file is in the same directory
    main_image_file = "dark.png" # Ensure this file is in the same directory
    templates_directory = "templates/" # Create this folder and put A.png, B.png, etc. inside

    # Run the full process
    extracted_df, detailed_info = run_olympiad_answer_extraction(
        main_image_file,
        templates_directory
    )

    if not extracted_df.empty:
        print("\nFinal Extracted Answers DataFrame:")
        print(extracted_df)