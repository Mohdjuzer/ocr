import cv2
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def detect_and_crop_rectangle(image_path):
    """
    Detects and crops a region of interest from the input image based on fixed percentages.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Calculate crop dimensions based on percentages
    image_height, image_width = image.shape[:2]
    top_percent = 0.15
    bottom_percent = 0.30
    
    # Calculate crop coordinates
    cropy = int(image_height * top_percent)
    croph = int(image_height * (1-top_percent-bottom_percent))
    cropx = 0
    cropw = image_width

    # Extract the ROI
    cropped_image = image[cropy:cropy+croph, cropx:cropx+cropw]

    # Optional: Draw rectangle on original image for visualization
    # cv2.rectangle(image, (cropx, cropy), (cropx + cropw, cropy + croph), (255, 0, 0), 2)

    if cropped_image is None or cropped_image.size == 0:
        print("Error: Cropping resulted in an empty image")
        return None

    return cropped_image


def upscale_image_only(image_array, scale_factor=2, interpolation_method='linear'):
    """
    Upscales an image array based on user input or defaults.
    Modified to work with image arrays instead of file paths.
    """
    if image_array is None:
        
        return None

   

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
    new_width = int(image_array.shape[1] * scale_factor)
    new_height = int(image_array.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)

    # Perform upscaling
    upscaled_image = cv2.resize(image_array, new_dimensions, interpolation=chosen_interpolation)

    

    return upscaled_image


class OlympiadAnswerExtractor:
    def __init__(self, image_array, template_folder_path=None):
        self.image_array = image_array
        self.template_folder_path = template_folder_path
        self.main_image = None
        self.enhanced_image = None
        self.templates = {} # To store loaded templates
        self.min_matching_confidence = 0.35  # Lowered to better detect "-" marks

    def load_main_image(self):
        """
        Uses the provided image array as the main image.
        """
        if self.image_array is None:
            raise ValueError("No image array provided to the extractor.")
        
        self.main_image = self.image_array.copy()
        
        return True

    def load_templates(self):
        """
        Loads template images for options (A, B, C, D) from the specified folder.
        """
        if not self.template_folder_path or not os.path.exists(self.template_folder_path):
           
            return False

        
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
                        
                        loaded_any = True
                    else:
                        print(f"Warning: Could not load template {filename} from folder.")
        
        
        return loaded_any

    def enhance_image(self):
        """
        Enhances the loaded main image using CLAHE and median blur.
        """
        if self.main_image is None:
           
            return False

        gray_image = cv2.cvtColor(self.main_image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_contrast_image = clahe.apply(gray_image)
        
        # Apply median blur instead of Gaussian
        self.enhanced_image = cv2.medianBlur(enhanced_contrast_image, 5)

        
        return True

    def detect_solid_circles(self):
        """
        Detects circles using an improved HoughCircles method with CLAHE and median blur,
        tuned for faded/weak circles. Now includes param grid search for optimal detection.
        """
        if self.enhanced_image is None:
            return []

        # Convert to grayscale if not already
        if len(self.enhanced_image.shape) == 3:
            gray = cv2.cvtColor(self.enhanced_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.enhanced_image.copy()

        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)

        # Median blur to reduce noise but preserve edges
        gray_blurred = cv2.medianBlur(gray_clahe, 5)

        # --- Parameter grid search for HoughCircles ---
        param2_scale = [35,34,33,32,31,30,20,15,10]
        param1_scale = [150,160,170,180,200,250]
        circles = None
        perf = False
        for i in param2_scale:
            for j in param1_scale:
                circles = cv2.HoughCircles(
                    gray_blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=50,
                    param1=j,
                    param2=i,
                    minRadius=10,
                    maxRadius=20
                )
                if circles is not None and (len(circles[0]) == 25 or len(circles[0]) == 30 or len(circles[0]) == 35):
                    perf = True
                    break
            if perf:
                break

        if circles is not None:
            print(f"Circles Found:- {len(circles[0])}")
        else:
            print("Circles Found:- 0")

        found_circles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, circle in enumerate(circles[0, :]):
                center_x, center_y, radius = circle[0], circle[1], circle[2]
                found_circles.append({
                    'id': i,
                    'center': (center_x, center_y),
                    'bbox': (center_x - radius, center_y - radius, radius * 2, radius * 2),
                    'radius': radius,
                    'area': np.pi * radius * radius,
                    'circularity': 1.0,
                    'contour': None
                })

        # --- Robust grid sorting (unchanged) ---
        y_coords = np.array([c['center'][1] for c in found_circles])
        y_sorted = np.sort(y_coords)
        row_centers = []
        tolerance = 25  # pixels, adjust if needed
        for y in y_sorted:
            if not row_centers or abs(y - row_centers[-1]) > tolerance:
                row_centers.append(y)
        x_coords = np.array([c['center'][0] for c in found_circles])
        x_sorted = np.sort(x_coords)
        col_centers = []
        for x in x_sorted:
            if not col_centers or abs(x - col_centers[-1]) > tolerance:
                col_centers.append(x)
        def find_nearest(val, centers):
            return np.argmin([abs(val - c) for c in centers])
        for c in found_circles:
            c['row'] = find_nearest(c['center'][1], row_centers)
            c['col'] = find_nearest(c['center'][0], col_centers)
        found_circles = sorted(found_circles, key=lambda c: (c['row'], c['col']))
        for i, circle in enumerate(found_circles):
            circle['question_number'] = i + 1

        return found_circles

    def extract_circle_roi(self, circle_data, padding=5):
        """
        Extracts the Region of Interest (ROI) around a detected circle from the ORIGINAL image.
        """
        if self.main_image is None:
            
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
        

        self.load_main_image()
        self.enhance_image()
        self.load_templates()

        circles = self.detect_solid_circles()

        if not circles:
            
            return pd.DataFrame(), []

        extracted_answers_list = []
        detailed_results_for_vis = []

        for i, circle in enumerate(circles):
            roi = self.extract_circle_roi(circle)

            if roi.size == 0:
                
                continue

            best_match_letter, confidence = self.match_templates_multiscale(roi)

            if best_match_letter and confidence >= self.min_matching_confidence:
                answer = best_match_letter
               
            else:
                answer = "Unsure"
                

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
            

        return df, detailed_results_for_vis

    def create_visualization(self, detailed_results, output_path='template_matching_results.png'):
        """
        Creates a visual representation of the detection and matching results on the main image.
        """
        if self.main_image is None:
            
            return None
        if not detailed_results:
            
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
        
        return vis_image

    def save_results_to_excel(self, df, filename='extracted_answers.xlsx'):
        """
        Saves the extracted answers DataFrame to an Excel file.
        """
        if not df.empty:
            df.to_excel(filename, index=False)
           
            return filename
        else:
            
            return None


def run_integrated_olympiad_extraction(main_image_path, template_folder):
    """
    Integrated workflow with automatic scale factor optimization
    """
   
    
    # Step 1: Detect and crop rectangle
   
    cropped_image = detect_and_crop_rectangle(main_image_path)
    
    if cropped_image is None:
       
        return pd.DataFrame(), []
    
    # Try different scale factors
    scale_factors = [2.0]  
    best_results = None
    max_circles = 0
    best_scale = None
    
    for scale_factor in scale_factors:
        
        # Step 2: Upscale the cropped image
        upscaled_image = upscale_image_only(cropped_image, scale_factor=scale_factor, 
                                          interpolation_method='linear')
        
        if upscaled_image is None:
            continue
            
        # Step 3: Process the upscaled cropped image
        extractor = OlympiadAnswerExtractor(upscaled_image, template_folder)
        
        try:
            extracted_answers_df, detailed_results = extractor.process_and_match_answers()
            
            if not extracted_answers_df.empty:
                num_circles = len(detailed_results)
                
                
                # Update best result if we found more circles
                if num_circles > max_circles:
                    max_circles = num_circles
                    best_results = (extracted_answers_df, detailed_results, extractor)
                    best_scale = scale_factor
                
                # If we found all 35 circles, we can stop
                if num_circles >= 35:
                    
                    break
                    
        except Exception as e:
            print(f"Error with scale factor {scale_factor}: {str(e)}")
            continue
    
    # Process the best result
    if best_results:
        extracted_answers_df, detailed_results, best_extractor = best_results
       
        
        best_extractor.save_results_to_excel(extracted_answers_df)
        best_extractor.create_visualization(detailed_results)
        
        return extracted_answers_df, detailed_results
    else:
       
        return pd.DataFrame(), []


if __name__ == "__main__":
    # --- Configuration ---
    main_image_file = "issue3.png"  # Original input image
    templates_directory = "templates/"  # Folder with A.png, B.png, etc.

    # Run the integrated process with automatic scale factor optimization
    extracted_df, detailed_info = run_integrated_olympiad_extraction(
        main_image_file,
        templates_directory
    )

    if not extracted_df.empty:
       
        print("Extracted Answers:")
        
    else:
        print("No answers extracted.")