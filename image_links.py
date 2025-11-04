import numpy as np
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import requests
from urllib.parse import urlparse
import testing
import mohd
import example
import sys
import io
from io import BytesIO

def download_image(url):
    """
    Downloads an image from URL and returns it as a BytesIO object
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        print(f"Successfully downloaded image from: {url}")
        return image_data
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None

def detect_mode(image_path, threshold=128):
    """
    Detects if an image is dark or light based on average brightness.
    """
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    avg_brightness = np.mean(arr)
    
    return 'dark' if avg_brightness < threshold else 'light'

def run_dark_mode_code(image_path):
    """
    Runs the dark mode processing pipeline using mohd.py
    """
    print("Running DARK MODE code on:", image_path)
    light_image_file = "temp_clean_light.png"
    try:
        mohd.simple_invert(image_path, light_image_file)
        
        template_folder = "Template/"
        extracted_df, detailed_results = mohd.run_integrated_olympiad_extraction(
            light_image_file, 
            template_folder
        )
        if os.path.exists(light_image_file):
            os.remove(light_image_file)
        if not extracted_df.empty:
            print("\n=== DARK MODE RESULTS ===")
            # print("Extracted Answers:")
            # print(extracted_df)
            # print(f"\nProcessed {len(detailed_results)} circles")
        else:
            print("No answers extracted in dark mode.")
        return extracted_df, detailed_results
    except Exception as e:
        print(f"Error during processing: {e}")
        if os.path.exists(light_image_file):
            os.remove(light_image_file)
        return None, None

def run_light_mode_code(image_path):
    """
    Runs the light mode processing pipeline using testing.py
    """
    print("Running LIGHT MODE code on:", image_path)
    template_folder = "templates/"
    try:
        extracted_df, detailed_results = testing.run_integrated_olympiad_extraction(
            image_path, 
            template_folder
        )
        if not extracted_df.empty:
            print("\n=== LIGHT MODE RESULTS ===")
            
        else:
            print("No answers extracted in light mode.")
        return extracted_df, detailed_results
    except Exception as e:
        print(f"Error during light mode processing: {e}")
        return None, None

def run_with_fallback(image_path, primary_mode):
    """
    Runs the primary mode first, then falls back to the other mode if needed
    """
    print(f"\n=== TRYING PRIMARY MODE: {primary_mode.upper()} ===")
    if primary_mode == 'dark':
        extracted_df, detailed_results = run_dark_mode_code(image_path)
        fallback_mode = 'light'
    else:
        extracted_df, detailed_results = run_light_mode_code(image_path)
        fallback_mode = 'dark'
    if extracted_df is not None and not extracted_df.empty:
        print(f"✓ SUCCESS with {primary_mode.upper()} mode!")
        return extracted_df, detailed_results, primary_mode
    print(f"\n=== PRIMARY MODE FAILED, TRYING FALLBACK: {fallback_mode.upper()} ===")
    if fallback_mode == 'dark':
        # FORCE INVERTION IN FALLBACK DARK MODE
        inverted_path = "temp_fallback_inverted.png"
        mohd.simple_invert(image_path, inverted_path)
        extracted_df, detailed_results = run_dark_mode_code(inverted_path)
        if os.path.exists(inverted_path):
            os.remove(inverted_path)
    else:
        extracted_df, detailed_results = run_light_mode_code(image_path)
    if extracted_df is not None and not extracted_df.empty:
        print(f"SUCCESS with {fallback_mode.upper()} mode!")
        return extracted_df, detailed_results, fallback_mode
    else:
        print(f"BOTH MODES FAILED")
        return None, None, None

def process_single_image(input_image, results_folder):
    """
    Process a single image and save its template matching results as image
    """
    try:
        # Run example.py processing first
        example.extract_exam_stats_visual(input_image)
        # Detect if the image is dark or light
        mode = detect_mode(input_image)
        
        # Run processing with fallback
        extracted_df, detailed_results, successful_mode = run_with_fallback(input_image, mode)
        if extracted_df is not None and not extracted_df.empty:
            # Copy or move the template_matching_results.png to the results folder
            template_results = "template_matching_results.png"
            if os.path.exists(template_results):
                # Use the input image name as prefix for the result
                image_name = os.path.splitext(os.path.basename(input_image))[0]
                new_filename = f"{image_name}_template_matching_results.png"
                new_path = os.path.join(results_folder, new_filename)
                os.replace(template_results, new_path)
                
            # Print summary
            print(f"\n=== PROCESSING COMPLETE ===")
            
            return True
        else:
            
            
            return False
    except Exception as e:
        print(f"Error processing image {input_image}: {str(e)}")
        return False

def extract_answers_from_df(extracted_df):
    """
    Converts the extracted_df (with columns 'Question Number', 'Answer', ...) 
    to a list of answers for Q1-Q35.
    """
    answers = [None] * 35
    if extracted_df is not None and not extracted_df.empty:
        for _, row in extracted_df.iterrows():
            q_num = int(row['Question Number'])
            if 1 <= q_num <= 35:
                answers[q_num - 1] = row['Answer']
    return answers

def get_attempt_stats(image_path):
    """
    Runs example.extract_exam_stats_visual and captures Attempted, Not Attempted, Not Saved.
    Returns (attempted, not_attempted, not_saved)
    """
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        example.extract_exam_stats_visual(image_path)
    finally:
        sys.stdout = old_stdout
    output = mystdout.getvalue()
    attempted = not_attempted = not_saved = None
    for line in output.splitlines():
        line = line.strip()
        # Only match lines that start with the exact stat
        if line.startswith("Attempted:") and "Not" not in line:
            try:
                attempted = int(line.split(":")[1].strip())
            except Exception:
                attempted = None
        elif line.startswith("Not Attempted:"):
            try:
                not_attempted = int(line.split(":")[1].strip())
            except Exception:
                not_attempted = None
        elif line.startswith("Not Saved:"):
            try:
                not_saved = int(line.split(":")[1].strip())
            except Exception:
                not_saved = None
    return attempted, not_attempted, not_saved

def main():
    """
    Main function that processes multiple images from an Excel file
    """
    excel_file = "image_links.xlsx"
    try:
        # Read and display Excel file contents for debugging
        df = pd.read_excel(excel_file)
        df.columns = df.columns.str.strip()  # Clean column names
        image_column = "ImagePath"
        if image_column not in df.columns:
            print(f"Error: Column '{image_column}' not found!")
            print("Available columns:", ", ".join(df.columns.tolist()))
            return
        # Create main results folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"template_matching_results_{timestamp}"
        os.makedirs(results_folder, exist_ok=True)
        successful_count = 0
        total_images = len(df)
        all_results = []
        print(f"\n=== PROCESSING {total_images} IMAGES ===")
        for index, row in df.iterrows():
            image_path = row[image_column]
                
            
            # Handle URL or local path
            if image_path.startswith(('http://', 'https://')):
                image_data = download_image(image_path)
                if image_data is None:
                    row_result = {
                        'ImagePath': os.path.basename(urlparse(image_path).path),
                        'Attempted': None,
                        'Not_Attempted': None,
                        'Not_Saved': None
                    }
                    for q in range(1, 36):
                        row_result[f"Q{q}"] = None
                    all_results.append(row_result)
                    continue
                # Save temporarily for processing
                temp_path = f"temp_processing_{index}.png"
                with Image.open(image_data) as img:
                    img.save(temp_path)
                local_path = temp_path
            else:
                local_path = image_path
                if not os.path.exists(local_path):
                    print(f"Error: Image file '{local_path}' not found.")
                    row_result = {
                        'ImagePath': os.path.basename(urlparse(image_path).path),
                        'Attempted': None,
                        'Not_Attempted': None,
                        'Not_Saved': None
                    }
                    for q in range(1, 36):
                        row_result[f"Q{q}"] = None
                    all_results.append(row_result)
                    continue

            try:
                if process_single_image(local_path, results_folder):
                    # Try to extract answers again for Excel output
                    mode = detect_mode(local_path)
                    extracted_df, _, _ = run_with_fallback(local_path, mode)
                    if extracted_df is not None and not extracted_df.empty:
                        answers = extract_answers_from_df(extracted_df)
                    else:
                        answers = [None] * 35
                    # Get attempted stats
                    attempted, not_attempted, not_saved = get_attempt_stats(local_path)
                    row_result = {
                        'ImagePath': os.path.basename(urlparse(image_path).path),
                        'Attempted': attempted,
                        'Not_Attempted': not_attempted,
                        'Not_Saved': not_saved
                    }
                    for i, ans in enumerate(answers, 1):
                        row_result[f"Q{i}"] = ans
                    all_results.append(row_result)
                    successful_count += 1
                else:
                    row_result = {
                        'ImagePath': os.path.basename(urlparse(image_path).path),
                        'Attempted': None,
                        'Not_Attempted': None,
                        'Not_Saved': None
                    }
                    for q in range(1, 36):
                        row_result[f"Q{q}"] = None
                    all_results.append(row_result)
                
                # Clean up temporary file if it was created
                if image_path.startswith(('http://', 'https://')):
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                if image_path.startswith(('http://', 'https://')) and os.path.exists(temp_path):
                    os.remove(temp_path)
                
        # Save all results to Excel
        results_df = pd.DataFrame(all_results)
        excel_output_path = os.path.join(results_folder, "all_results.xlsx")
        results_df.to_excel(excel_output_path, index=False)
        print(f"\nAll results saved to: {excel_output_path}")
        # Print final summary
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total images processed: {total_images}")
        print(f"Successful extractions: {successful_count}")
        print(f"Failed extractions: {total_images - successful_count}")
        print(f"Results saved in: {results_folder}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()