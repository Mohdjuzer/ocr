import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image


class CleanUIConverter:
    def __init__(self):
        pass

    def convert_clean(self, image_path, output_path=None):
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        inverted_array = 255 - img_array
        result = Image.fromarray(inverted_array)
        if output_path:
            result.save(output_path, quality=100)
        return result

    def convert_smart_invert(self, image_path, output_path=None):
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        brightness = np.mean(img_array, axis=2)
        result = img_array.copy()
        dark_mask = brightness < 128
        for i in range(3):
            result[:, :, i] = np.where(dark_mask, 255 - img_array[:, :, i], img_array[:, :, i])
        result_img = Image.fromarray(result)
        if output_path:
            result_img.save(output_path, quality=95)
        return result_img

    def convert_levels_adjustment(self, image_path, output_path=None):
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img, dtype=np.float32)
        normalized = img_array / 255.0
        adjusted = np.where(normalized < 0.5, 1.0 - (normalized * 1.5), normalized * 0.3)
        adjusted = np.clip(adjusted, 0, 1)
        result_array = (adjusted * 255).astype(np.uint8)
        result_img = Image.fromarray(result_array)
        if output_path:
            result_img.save(output_path, quality=95)
        return result_img

    def convert_threshold_based(self, image_path, output_path=None, threshold=100):
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        gray = np.mean(img_array, axis=2)
        invert_mask = gray < threshold
        result = img_array.copy()
        result[invert_mask] = 255 - img_array[invert_mask]
        result_img = Image.fromarray(result)
        if output_path:
            result_img.save(output_path, quality=95)
        return result_img

def simple_invert(input_path, output_path):
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    arr = np.array(img)
    inverted = 255 - arr
    result = Image.fromarray(inverted)
    result.save(output_path, quality=100)
    return result

def smart_invert(input_path, output_path):
    converter = CleanUIConverter()
    return converter.convert_smart_invert(input_path, output_path)

def threshold_invert(input_path, output_path, darkness_threshold=80):
    converter = CleanUIConverter()
    return converter.convert_threshold_based(input_path, output_path, darkness_threshold)

# --- Main Olympiad extraction pipeline ---
def detect_and_crop_rectangle(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 4)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    biggest_rect = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                biggest_rect = (x, y, w, h)
    if biggest_rect is not None:
        x, y, w, h = biggest_rect
        cropped = image[y:y+h, x:x+w]
        print(f"Biggest rectangle cropped: {w}x{h} pixels")
        return cropped
    else:
        print("No rectangle found. Using original image.")
        return image

def upscale_image_only(image_array, scale_factor=2, interpolation_method='linear'):
    if image_array is None:
        print("Error: No image array provided")
        return None
    method_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    chosen_interpolation = method_map.get(interpolation_method.lower(), cv2.INTER_LINEAR)
    new_width = int(image_array.shape[1] * scale_factor)
    new_height = int(image_array.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)
    upscaled_image = cv2.resize(image_array, new_dimensions, interpolation=chosen_interpolation)
    return upscaled_image

class OlympiadAnswerExtractor:
    def __init__(self, image_array, template_folder_path=None):
        self.image_array = image_array
        self.template_folder_path = template_folder_path
        self.main_image = None
        self.enhanced_image = None
        self.templates = {}
        self.min_matching_confidence = 0.42

    def load_main_image(self):
        if self.image_array is None:
            raise ValueError("No image array provided to the extractor.")
        self.main_image = self.image_array.copy()
        return True

    def load_templates(self):
        if not self.template_folder_path or not os.path.exists(self.template_folder_path):
            print("Template folder not provided or does not exist. Template matching will be skipped.")
            return False
        
        valid_prefixes = ['A', 'B', 'C', 'D', 'H','AA','BB','CC','DD']  # All valid template prefixes
        
        for filename in os.listdir(self.template_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Get the base name without extension
                base_name = os.path.splitext(filename)[0].upper()
                
                if base_name in valid_prefixes:
                    template_path = os.path.join(self.template_folder_path, filename)
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        template_enhanced = clahe.apply(template)
                        template_processed = cv2.GaussianBlur(template_enhanced, (3, 3), 0)
                        self.templates[base_name] = template_processed
        print(f"Loaded {len(self.templates)} templates: {list(self.templates.keys())}")
        return bool(self.templates)

    def enhance_image(self):
        if self.main_image is None:
            print("Main image not loaded. Cannot enhance.")
            return False
        gray_image = cv2.cvtColor(self.main_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        enhanced_contrast_image = clahe.apply(gray_image)
        self.enhanced_image = cv2.GaussianBlur(enhanced_contrast_image, (7, 7), 0)
        return True

    def detect_solid_circles(self):
        if self.main_image is None:
            print("Main image not loaded. Cannot detect circles.")
            return []
        hsv = cv2.cvtColor(self.main_image, cv2.COLOR_BGR2HSV)
        # Light theme: yellow-green mask (for attempted circles)
        lower_yellow_green = np.array([35, 30, 120])
        upper_yellow_green = np.array([90, 255, 255])
        yellow_green_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
        # Dark theme: purple mask
        lower_purple = np.array([120, 40, 40])
        upper_purple = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        # Combine masks
        combined_mask = cv2.bitwise_or(yellow_green_mask, purple_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_circles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 100 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.3:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.5:
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
        found_circles = sorted(found_circles, key=lambda c: (c['center'][1] // 50, c['center'][0]))
        print(f"Detected {len(found_circles)} attempted circles.")
        return found_circles

    def extract_circle_roi(self, circle_data, padding=5):
        if self.main_image is None:
            print("Original main image not loaded. Cannot extract ROI.")
            return np.array([])
        x, y, w, h = circle_data['bbox']
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(self.main_image.shape[1], x + w + padding)
        y2 = min(self.main_image.shape[0], y + h + padding)
        roi = self.main_image[y1:y2, x1:x2]
        return roi

    def preprocess_roi_for_matching(self, roi):
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_enhanced = clahe.apply(gray_roi)
        roi_processed = cv2.GaussianBlur(roi_enhanced, (3, 3), 0)
        return roi_processed

    def match_templates_multiscale(self, roi):
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
                except Exception:
                    pass
        return best_match_letter, best_confidence

    def process_and_match_answers(self):
        self.load_main_image()
        self.enhance_image()
        self.load_templates()
        circles = self.detect_solid_circles()
        if not circles:
            print("No attempted circles detected. Cannot proceed with answer matching.")
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
        if self.main_image is None or not detailed_results:
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
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved as '{output_path}'")
        return vis_image

    def save_results_to_excel(self, df, filename='extracted_answers.xlsx'):
        if not df.empty:
            df.to_excel(filename, index=False)
            print(f"Results saved to {filename}")
            return filename
        else:
            print("No results DataFrame to save.")
            return None

def run_integrated_olympiad_extraction(main_image_path, template_folder):
    print("=== Starting Integrated Rectangle Detection and Answer Extraction ===")
    cropped_image = detect_and_crop_rectangle(main_image_path)
    if cropped_image is None:
        print("Rectangle detection failed. Aborting process.")
        return pd.DataFrame(), []
    upscaled_image = upscale_image_only(cropped_image, scale_factor=2, interpolation_method='linear')
    if upscaled_image is None:
        print("Image upscaling failed. Aborting answer extraction.")
        return pd.DataFrame(), []
    extractor = OlympiadAnswerExtractor(upscaled_image, template_folder)
    try:
        extracted_answers_df, detailed_results = extractor.process_and_match_answers()
        if not extracted_answers_df.empty:
            extractor.save_results_to_excel(extracted_answers_df)
            extractor.create_visualization(detailed_results)
            print("\nIntegrated extraction and matching process completed successfully.")
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
    # --- Step 1: Always convert dark image to light version ---
    dark_image_file = "issue3.png"  # Input dark image
    light_image_file = "clean_light.png"  # Output light image

    print("Converting dark image to light version...")
    simple_invert(dark_image_file, light_image_file)
    print(f"Saved light image as {light_image_file}")

    # --- Step 2: Run answer extraction on the light image ---
    templates_directory = "Template/"    # Folder with A.png, B.png, etc.
    extracted_df, detailed_info = run_integrated_olympiad_extraction(
        light_image_file,
        templates_directory
    )
    if not extracted_df.empty:
        print("\nFinal Extracted Answers DataFrame:")
        print(extracted_df)