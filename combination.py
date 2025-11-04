import numpy as np
from PIL import Image
import os

# Import the functions from your modules
import testing
import mohd

def detect_mode(image_path, threshold=128):
    """
    Detects if an image is dark or light based on average brightness.
    
    Args:
        image_path (str): Path to the input image
        threshold (int): Brightness threshold (default: 128)
    
    Returns:
        str: 'dark' if average brightness < threshold, 'light' otherwise
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    arr = np.array(img)
    avg_brightness = np.mean(arr)
    print(f"Average brightness: {avg_brightness:.2f}")
    return 'dark' if avg_brightness < threshold else 'light'

def run_dark_mode_code(image_path):
    """
    Runs the dark mode processing pipeline using mohd.py
    This includes converting dark image to light mode first, then processing
    
    Args:
        image_path (str): Path to the input image
    """
    print("Running DARK MODE code on:", image_path)
    
    # Step 1: Convert dark image to light using mohd.py functions
    light_image_file = "temp_clean_light.png"  # Temporary light image
    
    print("Converting dark image to light version...")
    try:
        # Use the simple_invert function from mohd.py
        mohd.simple_invert(image_path, light_image_file)
        print(f"Saved light image as {light_image_file}")
    except Exception as e:
        print(f"Error converting image: {e}")
        return None, None
    
    # Step 2: Run answer extraction on the light image
    template_folder = "Template/"  # Template folder for dark mode
    
    print("Processing converted light image for answer extraction...")
    try:
        # Call the main function from mohd.py with the converted light image
        extracted_df, detailed_results = mohd.run_integrated_olympiad_extraction(
            light_image_file, 
            template_folder
        )
        
        # Clean up temporary file
        if os.path.exists(light_image_file):
            os.remove(light_image_file)
            print(f"Cleaned up temporary file: {light_image_file}")
        
        if not extracted_df.empty:
            print("\n=== DARK MODE RESULTS ===")
            print("Extracted Answers:")
            print(extracted_df)
            print(f"\nProcessed {len(detailed_results)} circles")
        else:
            print("No answers extracted in dark mode.")
        
        return extracted_df, detailed_results
        
    except Exception as e:
        print(f"Error during processing: {e}")
        # Clean up temporary file even if there's an error
        if os.path.exists(light_image_file):
            os.remove(light_image_file)
        return None, None

def run_light_mode_code(image_path):
    """
    Runs the light mode processing pipeline using testing.py
    
    Args:
        image_path (str): Path to the input image
    """
    print("Running LIGHT MODE code on:", image_path)
    template_folder = "templates/"  # Template folder for light mode
    
    try:
        # Call the main function from testing.py
        extracted_df, detailed_results = testing.run_integrated_olympiad_extraction(
            image_path, 
            template_folder
        )
        
        if not extracted_df.empty:
            print("\n=== LIGHT MODE RESULTS ===")
            print("Extracted Answers:")
            print(extracted_df)
            print(f"\nProcessed {len(detailed_results)} circles")
        else:
            print("No answers extracted in light mode.")
        
        return extracted_df, detailed_results
        
    except Exception as e:
        print(f"Error during light mode processing: {e}")
        return None, None

def run_with_fallback(image_path, primary_mode):
    """
    Runs the primary mode first, then falls back to the other mode if needed
    
    Args:
        image_path (str): Path to the input image
        primary_mode (str): 'dark' or 'light'
    
    Returns:
        tuple: (extracted_df, detailed_results, successful_mode)
    """
    print(f"\n=== TRYING PRIMARY MODE: {primary_mode.upper()} ===")
    
    # Try primary mode first
    if primary_mode == 'dark':
        extracted_df, detailed_results = run_dark_mode_code(image_path)
        fallback_mode = 'light'
    else:
        extracted_df, detailed_results = run_light_mode_code(image_path)
        fallback_mode = 'dark'
    
    # Check if primary mode was successful
    if extracted_df is not None and not extracted_df.empty:
        print(f"✓ SUCCESS with {primary_mode.upper()} mode!")
        return extracted_df, detailed_results, primary_mode
    
    # Try fallback mode
    print(f"\n=== PRIMARY MODE FAILED, TRYING FALLBACK: {fallback_mode.upper()} ===")
    
    if fallback_mode == 'dark':
        extracted_df, detailed_results = run_dark_mode_code(image_path)
    else:
        extracted_df, detailed_results = run_light_mode_code(image_path)
    
    if extracted_df is not None and not extracted_df.empty:
        print(f"✓ SUCCESS with {fallback_mode.upper()} mode!")
        return extracted_df, detailed_results, fallback_mode
    else:
        print(f"✗ BOTH MODES FAILED")
        return None, None, None

def main():
    """
    Main function that detects image mode and runs appropriate processing pipeline.
    """
    # Configuration
    input_image = "dark.png"  # Change this to your image filename
    
    print("=== IMAGE MODE DETECTION AND PROCESSING ===")
    print(f"Processing image: {input_image}")
    
    try:
        # Check if image file exists
        if not os.path.exists(input_image):
            print(f"Error: Image file '{input_image}' not found.")
            return
        
        # Detect if the image is dark or light
        mode = detect_mode(input_image)
        print(f"Detected mode: {mode.upper()}")
        
        # Run processing with fallback
        extracted_df, detailed_results, successful_mode = run_with_fallback(input_image, mode)
        
        # Summary
        if extracted_df is not None and not extracted_df.empty:
            print(f"\n=== PROCESSING COMPLETE ===")
            print(f"Original detected mode: {mode.upper()}")
            print(f"Successful processing mode: {successful_mode.upper()}")
            print(f"Total questions processed: {len(extracted_df)}")
            
            # Count successful answers
            successful_answers = len(extracted_df[extracted_df['Answer'] != 'Unsure'])
            uncertain_answers = len(extracted_df[extracted_df['Answer'] == 'Unsure'])
            
            print(f"Answers extracted: {successful_answers}")
            print(f"Uncertain answers: {uncertain_answers}")
            
            # Show answer distribution
            if successful_answers > 0:
                answer_counts = extracted_df[extracted_df['Answer'] != 'Unsure']['Answer'].value_counts()
                print(f"Answer distribution: {dict(answer_counts)}")
                
        else:
            print(f"\n=== PROCESSING FAILED ===")
            print("No answers could be extracted from the image using either mode.")
            print("Possible issues:")
            print("- Image quality too low")
            print("- No circles detected")
            print("- Template matching failed")
            print("- Wrong template folder or missing templates")
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()