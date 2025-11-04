import cv2
import os

def upscale_image_only(image_path, scale_factor=None, interpolation_method=None, output_filename=None):
    """
    Loads an image and upscales its pixel size based on user input or defaults.

    Args:
        image_path (str): Path to the input image file.
        scale_factor (int, optional): The factor by which to increase the image resolution.
                                      E.g., 2 for 2x, 4 for 4x. If None, prompts user.
        interpolation_method (str, optional): The interpolation method to use.
                                              Options: 'nearest', 'linear', 'cubic', 'lanczos'.
                                              If None, prompts user.
        output_filename (str, optional): Name for the output upscaled image file.
                                         If None, a default name will be generated.

    Returns:
        numpy.ndarray: The upscaled image.
        None: If the image cannot be loaded or an error occurs.
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

    # --- Get Scale Factor ---
    if scale_factor is None:
        while True:
            try:
                scale_input = input("Enter scale factor (e.g., 2 for 2x, 4 for 4x): ")
                scale_factor = int(scale_input)
                if scale_factor >= 1:
                    break
                else:
                    print("Scale factor must be an integer >= 1.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
    else:
        if not isinstance(scale_factor, int) or scale_factor < 1:
            print("Error: Provided scale_factor must be an integer >= 1. Using default prompt.")
            return upscale_image_only(image_path, output_filename=output_filename)


    # --- Get Interpolation Method ---
    interpolation_options = {
        '1': cv2.INTER_NEAREST,  # Nearest Neighbor (fastest, blocky)
        '2': cv2.INTER_LINEAR,   # Bilinear (default, good for downscaling, decent for up)
        '3': cv2.INTER_CUBIC,    # Bicubic (slower, smoother, generally good for upscaling)
        '4': cv2.INTER_LANCZOS4  # Lanczos (slowest, best quality for upscaling, less aliasing)
    }
    interpolation_names = {
        '1': 'Nearest Neighbor',
        '2': 'Bilinear',
        '3': 'Bicubic',
        '4': 'Lanczos'
    }

    if interpolation_method is None:
        print("\nChoose an interpolation method:")
        print("1. Nearest Neighbor (Fastest, pixelated results)")
        print("2. Bilinear (Good balance of speed and quality)")
        print("3. Bicubic (Smoother, generally better for upscaling)")
        print("4. Lanczos (Highest quality for upscaling, can be slowest)")

        while True:
            choice = input("Enter your choice (1-4): ")
            if choice in interpolation_options:
                chosen_interpolation = interpolation_options[choice]
                chosen_interpolation_name = interpolation_names[choice]
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 4.")
    else:
        method_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        if interpolation_method.lower() in method_map:
            chosen_interpolation = method_map[interpolation_method.lower()]
            chosen_interpolation_name = interpolation_method.capitalize()
        else:
            print(f"Error: Invalid interpolation_method '{interpolation_method}'. Using default prompt.")
            return upscale_image_only(image_path, scale_factor=scale_factor, output_filename=output_filename)

    # Calculate new dimensions
    new_width = int(original_image.shape[1] * scale_factor)
    new_height = int(original_image.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)

    # Perform upscaling
    upscaled_image = cv2.resize(original_image, new_dimensions, interpolation=chosen_interpolation)

    print(f"\nUpscaled image dimensions: {upscaled_image.shape[1]}x{upscaled_image.shape[0]}")
    print(f"Interpolation method used: {chosen_interpolation_name}")

    # --- Save the result ---
    if output_filename is None:
        base_name, ext = os.path.splitext(os.path.basename(image_path))
        output_filename = f"{base_name}_upscaled_x{scale_factor}_{chosen_interpolation_name.replace(' ', '')}{ext}"

    cv2.imwrite(output_filename, upscaled_image)
    print(f"Upscaled image saved as '{output_filename}'")

    # --- Optional: Display images ---
    # cv2.imshow("Original Image", original_image)
    # cv2.imshow("Upscaled Image", upscaled_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return upscaled_image

# --- Example Usage ---
if __name__ == "__main__":
    input_image_path = "test4.png" 
    upscaled_img = upscale_image_only(input_image_path, scale_factor=2, interpolation_method='linear')

    # Option 2: Run with specific scale_factor and interpolation_method (uncomment to use)
    # upscaled_img_2x_lanczos = upscale_image_only(input_image_path, scale_factor=2, interpolation_method='lanczos')

    # Option 3: Run with a different scale factor and method
    # upscaled_img_4x_cubic = upscale_image_only(input_image_path, scale_factor=4, interpolation_method='cubic')

    if upscaled_img is not None:
        print("\nImage upscaling complete.")
    else:
        print("\nImage upscaling failed.")