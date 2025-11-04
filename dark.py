import cv2
import numpy as np
from PIL import Image

class CleanUIConverter:
    def __init__(self):
        pass
    
    def convert_clean(self, image_path, output_path=None):
        """
        Clean conversion - just invert colors without blur or quality loss
        """
        # Load image with PIL to maintain quality
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Simple inversion: 255 - pixel_value
        inverted_array = 255 - img_array
        
        # Convert back to PIL Image
        result = Image.fromarray(inverted_array)
        
        if output_path:
            result.save(output_path, quality=100)  # High quality save
        
        return result
    
    def convert_smart_invert(self, image_path, output_path=None):
        """
        Smart inversion - only invert dark areas, keep bright areas
        """
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Calculate brightness of each pixel
        brightness = np.mean(img_array, axis=2)
        
        # Only invert pixels that are dark (brightness < 128)
        result = img_array.copy()
        dark_mask = brightness < 128
        
        # Invert only dark pixels
        for i in range(3):  # RGB channels
            result[:, :, i] = np.where(dark_mask, 
                                     255 - img_array[:, :, i], 
                                     img_array[:, :, i])
        
        result_img = Image.fromarray(result)
        
        if output_path:
            result_img.save(output_path, quality=95)
        
        return result_img
    
    def convert_levels_adjustment(self, image_path, output_path=None):
        """
        Convert using levels adjustment - brighten darks, darken lights
        """
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to 0-1
        normalized = img_array / 255.0
        
        # Apply levels adjustment - this brightens dark areas significantly
        # and darkens light areas
        adjusted = np.where(normalized < 0.5, 
                           1.0 - (normalized * 1.5),  # Brighten darks more
                           normalized * 0.3)          # Darken lights
        
        # Ensure values stay in valid range
        adjusted = np.clip(adjusted, 0, 1)
        
        # Convert back to 0-255
        result_array = (adjusted * 255).astype(np.uint8)
        
        result_img = Image.fromarray(result_array)
        
        if output_path:
            result_img.save(output_path, quality=95)
        
        return result_img
    
    def convert_threshold_based(self, image_path, output_path=None, threshold=100):
        """
        Threshold-based conversion - more control over what gets inverted
        """
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Calculate grayscale version for thresholding
        gray = np.mean(img_array, axis=2)
        
        # Create mask for areas to invert (dark areas)
        invert_mask = gray < threshold
        
        # Apply inversion only to masked areas
        result = img_array.copy()
        result[invert_mask] = 255 - img_array[invert_mask]
        
        result_img = Image.fromarray(result)
        
        if output_path:
            result_img.save(output_path, quality=95)
        
        return result_img

# Ultra-simple functions for quick use
def simple_invert(input_path, output_path):
    """
    Ultra-simple inversion - no quality loss
    """
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Simple numpy inversion
    arr = np.array(img)
    inverted = 255 - arr
    result = Image.fromarray(inverted)
    
    result.save(output_path, quality=100)
    return result

def smart_invert(input_path, output_path):
    """
    Only invert dark parts, preserve light parts
    """
    converter = CleanUIConverter()
    return converter.convert_smart_invert(input_path, output_path)

def threshold_invert(input_path, output_path, darkness_threshold=80):
    """
    Invert only pixels darker than threshold
    """
    converter = CleanUIConverter()
    return converter.convert_threshold_based(input_path, output_path, darkness_threshold)

# Example usage
if __name__ == "__main__":
    print("Clean UI Converter - No blur, no quality loss!")
    print("\nUsage examples:")
    print("1. simple_invert('dark.png', 'light.png')")
    print("2. smart_invert('dark.png', 'light.png')")
    print("3. threshold_invert('dark.png', 'light.png', 100)")
    
    # Uncomment to test:
    simple_invert('dark.png', 'clean_light.png')
