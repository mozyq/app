import json
import math
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def filter_resize_crop(
    *,
    in_folder: Path,
    out_folder: Path,
    min_width: int,
    min_height: int,
    target_width: int,
        target_height: int):
    """
    Filter, resize, and crop images from input folder to output folder.

    Args:
        in_folder: Source directory containing images
        out_folder: Destination directory for processed images
        min_width: Minimum width threshold for filtering
        min_height: Minimum height threshold for filtering
        target_width: Target width after resizing
        target_height: Target height after cropping
    """
    # Create output folder if it doesn't exist
    out_folder.mkdir(parents=True, exist_ok=True)

    # Get all image files (common image extensions)
    image_extensions = {'.jpg', '.jpeg', '.png',
                        '.bmp', '.tiff', '.tif', '.webp'}
    image_files = [
        f for f in in_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    processed_count = 0
    skipped_count = 0

    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Open image and get dimensions
            with Image.open(image_path) as img:
                width, height = img.size

                # Skip images that don't meet minimum size requirements
                if width < min_width or height < min_height:
                    skipped_count += 1
                    continue

                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize to target width while maintaining aspect ratio
                aspect_ratio = height / width
                new_height = int(target_width * aspect_ratio)
                img_resized = img.resize(
                    (target_width, new_height), Image.Resampling.LANCZOS)

                # Crop to target height (center crop)
                if new_height >= target_height:
                    # Crop from center
                    top = (new_height - target_height) // 2
                    bottom = top + target_height
                    img_cropped = img_resized.crop(
                        (0, top, target_width, bottom))
                else:
                    # If resized height is less than target, pad or skip
                    # For now, we'll skip such images
                    skipped_count += 1
                    continue

                # Save to output folder with same filename
                output_path = out_folder / f'{i:04d}.jpg'
                img_cropped.save(output_path, 'JPEG', quality=95)
                processed_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            skipped_count += 1
            continue

    print(
        f"Processing complete: {processed_count} images processed, {skipped_count} images skipped")


def index_html(input_json: Path) -> str:
    """
    Generate an HTML page displaying a grid of tiles with master image blended overlay.

    Args:
        input_json: Path to JSON file containing master and tiles information

    Returns:
        HTML string containing the grid visualization
    """
    # Load the JSON data
    with open(input_json, 'r') as f:
        data = json.load(f)

    master_image = data['master']
    tiles = data['tiles']

    # Calculate grid dimensions (assuming square grid)
    num_tiles = len(tiles)
    grid_size = int(math.sqrt(num_tiles))

    # If not a perfect square, calculate rectangular grid
    if grid_size * grid_size != num_tiles:
        # Find the best rectangular arrangement
        for rows in range(1, int(math.sqrt(num_tiles)) + 1):
            if num_tiles % rows == 0:
                cols = num_tiles // rows
                grid_size = max(rows, cols)
                grid_rows = rows
                grid_cols = cols
                break
    else:
        grid_rows = grid_cols = grid_size

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mozyq Grid Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}

        .grid-container {{
            position: relative;
            display: inline-block;
            width: 600px;
            height: 750px;
        }}

        .tiles-grid {{
            display: grid;
            grid-template-columns: repeat({grid_cols}, 1fr);
            grid-template-rows: repeat({grid_rows}, 1fr);
            gap: 0;
            width: 600px;
            height: 750px;
        }}

        .tile {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}

        .master-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.5;
        }}
    </style>
</head>
<body>
    <div class="grid-container">
        <div class="tiles-grid">"""

    # Add each tile to the grid
    for i, tile_path in enumerate(tiles):
        html += f"""
            <img src="{tile_path}" alt="Tile {i+1}" class="tile">"""

    html += f"""
        </div>
        <img src="{master_image}" alt="Master Image" class="master-overlay">
    </div>
</body>
</html>"""

    return html


if __name__ == '__main__':
    with open('index.html', 'w') as f:
        f.write(index_html(Path('output.json')))
