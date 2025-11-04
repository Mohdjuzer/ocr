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
from multiprocessing import Pool, cpu_count
import tempfile
import gc
import glob
from tqdm import tqdm

# Constants
BATCH_SIZE = 100  # Images per batch
MAX_WORKERS = max(1, int(cpu_count() * 0.8))  # Use 80% of CPU cores
TEMPLATE_DARK = "Template/"
TEMPLATE_LIGHT = "templates/"

def download_image(url, retries=3):
    """Robust image downloader with retries"""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=(3.05, 10))
            response.raise_for_status()
            return BytesIO(response.content)
        except Exception as e:
            if attempt == retries - 1:
                print(f"Download failed after {retries} attempts: {url}")
                return None
            continue

def detect_mode(image_path, threshold=128):
    """Fast mode detection with downsampling"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((100, 100))  # Downsample for speed
            return 'dark' if np.mean(np.array(img)) < threshold else 'light'
    except Exception as e:
        print(f"Mode detection failed: {str(e)}")
        return 'light'  # Default fallback

def process_dark_image(image_path):
    """Dark mode processing pipeline"""
    light_image = os.path.join(tempfile.gettempdir(), f"temp_light_{os.getpid()}.png")
    try:
        mohd.simple_invert(image_path, light_image)
        df, _ = mohd.run_integrated_olympiad_extraction(light_image, TEMPLATE_DARK)
        return df
    except Exception as e:
        print(f"Dark processing error: {str(e)}")
        return None
    finally:
        if os.path.exists(light_image):
            os.remove(light_image)

def process_light_image(image_path):
    """Light mode processing pipeline"""
    try:
        df, _ = testing.run_integrated_olympiad_extraction(image_path, TEMPLATE_LIGHT)
        return df
    except Exception as e:
        print(f"Light processing error: {str(e)}")
        return None

def get_attempt_stats(image_path):
    """Captures attempt statistics from console output"""
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        example.extract_exam_stats_visual(image_path)
        output = mystdout.getvalue()
        stats = {'Attempted': None, 'Not_Attempted': None, 'Not_Saved': None}
        for line in output.splitlines():
            if line.startswith("Attempted:") and "Not" not in line:
                stats['Attempted'] = int(line.split(":")[1].strip())
            elif line.startswith("Not Attempted:"):
                stats['Not_Attempted'] = int(line.split(":")[1].strip())
            elif line.startswith("Not Saved:"):
                stats['Not_Saved'] = int(line.split(":")[1].strip())
        return stats
    except Exception:
        return {'Attempted': None, 'Not_Attempted': None, 'Not_Saved': None}
    finally:
        sys.stdout = old_stdout

def process_single_image(args):
    """Core processing function for parallel execution"""
    idx, row, temp_dir = args
    image_path = row['ImagePath']
    result = {
        'ImageID': idx,
        'ImagePath': os.path.basename(urlparse(image_path).path),
        **{f'Q{q}': None for q in range(1, 36)}
    }

    try:
        # Handle URL/local path
        if image_path.startswith(('http://', 'https://')):
            img_data = download_image(image_path)
            if not img_data:
                return result
            local_path = os.path.join(temp_dir, f"temp_{idx}.png")
            with Image.open(img_data) as img:
                img.save(local_path)
        else:
            local_path = image_path
            if not os.path.exists(local_path):
                return result

        # Process image
        example.extract_exam_stats_visual(local_path)
        mode = detect_mode(local_path)
        
        # Try primary mode then fallback
        primary_df = process_dark_image(local_path) if mode == 'dark' else process_light_image(local_path)
        fallback_df = process_light_image(local_path) if primary_df is None and mode == 'dark' else (
                     process_dark_image(local_path) if primary_df is None else None)
        
        final_df = primary_df if primary_df is not None else fallback_df
        
        # Update results
        if final_df is not None:
            for _, row in final_df.iterrows():
                q_num = int(row['Question Number'])
                if 1 <= q_num <= 35:
                    result[f'Q{q_num}'] = row['Answer']
        
        # Add attempt stats
        stats = get_attempt_stats(local_path)
        result.update(stats)

    except Exception as e:
        print(f"Error processing image {idx}: {str(e)}")
    finally:
        # Cleanup
        if image_path.startswith(('http://', 'https://')) and 'local_path' in locals():
            if os.path.exists(local_path):
                os.remove(local_path)
        gc.collect()
        return result

def main():
    """Optimized main function with parallel processing"""
    start_time = datetime.now()
    
    # Setup directories
    output_dir = f"results_{start_time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    try:
        df = pd.read_excel("image_links.xlsx")
        df.columns = df.columns.str.strip()
        if 'ImagePath' not in df.columns:
            raise ValueError("Input Excel must contain 'ImagePath' column")
    except Exception as e:
        print(f"Failed to load input: {str(e)}")
        return

    total_images = len(df)
    print(f"Processing {total_images} images using {MAX_WORKERS} CPU cores")

    # Process in parallel batches
    with tempfile.TemporaryDirectory() as temp_dir:
        results = []
        for batch_start in tqdm(range(0, total_images, BATCH_SIZE), desc="Processing Batches"):
            batch_end = min(batch_start + BATCH_SIZE, total_images)
            batch_df = df.iloc[batch_start:batch_end]
            
            with Pool(MAX_WORKERS) as pool:
                batch_results = list(pool.imap(
                    process_single_image,
                    [(idx, row, temp_dir) for idx, row in batch_df.iterrows()],
                    chunksize=10
                ))
            
            # Save batch results
            batch_file = os.path.join(output_dir, f"batch_{batch_start}_{batch_end}.parquet")
            pd.DataFrame(batch_results).to_parquet(batch_file)
            results.extend(batch_results)

    # Merge all results
    final_df = pd.DataFrame(results).sort_values('ImageID')
    final_output = os.path.join(output_dir, "final_results.xlsx")
    final_df.to_excel(final_output, index=False)

    # Print summary
    duration = (datetime.now() - start_time).total_seconds() / 60
    success_count = sum(1 for r in results if any(r[f'Q{q}'] is not None for q in range(1, 36)))
    print(f"\nProcessing completed in {duration:.2f} minutes")
    print(f"Success rate: {success_count}/{total_images} ({success_count/total_images:.1%})")
    print(f"Results saved to: {os.path.abspath(final_output)}")

if __name__ == "__main__":
    main()