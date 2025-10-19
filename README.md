# 🎨 mozyq

<video src="https://github.com/user-attachments/assets/c1ac633a-3eaa-405d-8a90-749d68fcd234" autoplay muted></video>

**mozyq** is a powerful Python command-line tool that creates stunning mosaic videos from collections of photographs. Transform your photo library into mesmerizing mosaic animations where each frame is composed of tiny tiles from your image collection, with smooth zoom transitions that reveal the underlying master image.

## ✨ Features

- 🖼️ **Photo Mosaics**: Create beautiful mosaics from any collection of images
- 🎬 **Video Generation**: Generate smooth transition videos with zoom effects
- 🔄 **Smart Tile Matching**: Uses advanced algorithms to find the best tile matches
- 📐 **Flexible Grid Sizes**: Support for various grid dimensions (15x15, 21x21, etc.)
- 🎯 **Arbitrary Aspect Ratios**: Works with both square and rectangular image formats
- 🖥️ **HTML Visualization**: Generate interactive HTML previews of your mosaics
- ⚡ **Fast Processing**: Optimized with NumPy, OpenCV, and SciPy
- 🛠️ **Image Normalization**: Built-in tools to prepare your image collection

## 🚀 Quick Start

### Prerequisites

You need **ffmpeg** installed on your system:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

### Installation

```bash
pip install mozyq
```

### Basic Usage

1. **Prepare your photos** (you need at least 225 images for a 15x15 grid):
   ```bash
   mkdir photos
   # Add your photos to the photos/ directory
   # Or download sample images:
   seq 999 | xargs -I {} -n 1 -P 8 wget https://picsum.photos/630 -O photos/{}.jpg
   ```

2. **Normalize your images** (optional but recommended):
   ```bash
   mzq normalize photos/ normalized/ --target-width 630 --target-height 630
   ```

3. **Generate mosaic data**:
   ```bash
   mzq json normalized/0001.jpg --width 630 --height 630 --num-tiles 21
   ```

4. **Create video frames**:
   ```bash
   mzq frames mzq.json output_frames/
   ```

5. **Generate final video** (using ffmpeg):
   ```bash
   ffmpeg -r 30 -i output_frames/%04d.jpg -c:v libx264 -pix_fmt yuv420p mosaic_video.mp4
   ```

## 📚 Detailed Documentation

### Commands Overview

mozyq provides three main commands that work together to create mosaic videos:

#### 1. `mzq normalize` - Image Preprocessing

Normalizes a collection of images to consistent dimensions and format.

```bash
mzq normalize INPUT_FOLDER OUTPUT_FOLDER [OPTIONS]
```

**Arguments:**
- `INPUT_FOLDER`: Directory containing source images
- `OUTPUT_FOLDER`: Directory for normalized images

**Options:**
- `--min-width INTEGER`: Minimum width for input filtering (default: 630)
- `--min-height INTEGER`: Minimum height for input filtering (default: 630)
- `--target-width INTEGER`: Target width for output images (default: 630)
- `--target-height INTEGER`: Target height for output images (default: 630)

**Example:**
```bash
mzq normalize photos/ normalized/ --target-width 630 --target-height 630
```

This command:
- Filters out images smaller than minimum dimensions
- Converts all images to RGB format
- Resizes and center-crops to target dimensions
- Saves as high-quality JPEG files

#### 2. `mzq json` - Mosaic Generation

Creates the mosaic composition data and saves it as JSON.

```bash
mzq json MASTER_IMAGE [OUTPUT_JSON] [OPTIONS]
```

**Arguments:**
- `MASTER_IMAGE`: The seed image that will be recreated as a mosaic
- `OUTPUT_JSON`: Path to output JSON file (default: mzq.json)

**Options:**
- `--width INTEGER`: Output width in pixels (default: 630)
- `--height INTEGER`: Output height in pixels (default: 630)
- `--num-tiles INTEGER`: Number of tiles per row/column (must be odd, default: 21)
- `--max-transitions INTEGER`: Number of mosaic transitions (default: 10)

**Example:**
```bash
mzq json normalized/0042.jpg result.json --width 630 --height 630 --num-tiles 21
```

**Important Notes:**
- `num_tiles` must be odd (e.g., 15, 21, 25)
- `width` and `height` must be divisible by `num_tiles`
- The tool will use images from the same folder as the master image
- Requires at least `num_tiles²` images (e.g., 225 images for 15x15 grid)

#### 3. `mzq frames` - Video Frame Generation

Generates individual frames for the mosaic video with smooth zoom transitions.

```bash
mzq frames MZQ_JSON OUTPUT_FOLDER
```

**Arguments:**
- `MZQ_JSON`: JSON file created by the `mzq json` command
- `OUTPUT_FOLDER`: Directory where video frames will be saved

**Example:**
```bash
mzq frames mzq.json video_frames/
```

This command creates numbered frame files (0000.jpg, 0001.jpg, etc.) with:
- Smooth zoom transitions from close-up to full mosaic view
- Proper tile arrangement and scaling
- High-quality output suitable for video encoding

### Advanced Usage

#### Custom Grid Sizes

Different grid sizes create different effects:

```bash
# Fine detail (25x25 = 625 tiles needed)
mzq json master.jpg --num-tiles 25 --width 750 --height 750

# Medium detail (15x15 = 225 tiles needed)
mzq json master.jpg --num-tiles 15 --width 600 --height 600

# Coarse detail (9x9 = 81 tiles needed)
mzq json master.jpg --num-tiles 9 --width 450 --height 450
```

#### Rectangular Formats

mozyq supports non-square aspect ratios:

```bash
# Portrait format (4:5 aspect ratio)
mzq normalize photos/ normalized/ --target-width 600 --target-height 750
mzq json normalized/image.jpg --width 600 --height 750 --num-tiles 15

# Landscape format (16:9 aspect ratio)
mzq normalize photos/ normalized/ --target-width 640 --target-height 360
mzq json normalized/image.jpg --width 640 --height 360 --num-tiles 16  # Not recommended (even)
```

#### Video Creation Pipeline

Complete workflow for creating a mosaic video:

```bash
# 1. Prepare images
mkdir original_photos normalized_photos video_frames
# (Add your photos to original_photos/)

# 2. Normalize images
mzq normalize original_photos/ normalized_photos/ \
    --target-width 600 --target-height 750 \
    --min-width 400 --min-height 400

# 3. Generate multiple mosaics (creates transitions)
mzq json normalized_photos/0001.jpg mosaics.json \
    --width 600 --height 750 --num-tiles 15 --max-transitions 10

# 4. Generate video frames
mzq frames mosaics.json video_frames/

# 5. Create final video with ffmpeg
ffmpeg -r 30 -i video_frames/%04d.jpg \
    -c:v libx264 -preset slow -crf 18 \
    -pix_fmt yuv420p final_video.mp4
```

## 🔧 Technical Details

### Algorithm Overview

1. **Image Preprocessing**: Images are converted to LAB color space for better perceptual color matching
2. **Tile Vectorization**: Each tile image is converted to a feature vector
3. **Patch Extraction**: The master image is divided into patches matching the grid size
4. **Optimal Assignment**: Uses the Hungarian algorithm (linear_sum_assignment) to find the best tile for each patch
5. **Frame Generation**: Creates smooth zoom transitions using crop-then-scale operations

### Performance Optimization

- **Caching**: Processed images are cached to speed up repeated operations
- **Vectorization**: NumPy operations for fast mathematical computations
- **Parallel Processing**: Multi-threaded image processing where possible
- **Memory Management**: Efficient handling of large image collections

### File Formats

- **Input**: Supports JPEG, PNG, BMP, TIFF, WebP
- **Output**: High-quality JPEG frames optimized for video encoding
- **Intermediate**: JSON files for mosaic composition data

## 🎛️ Configuration Tips

### Image Collection Guidelines

- **Quantity**: More images = better mosaic quality
  - Minimum: `num_tiles²` (e.g., 225 for 15x15)
  - Recommended: 500-2000 images
  - Optimal: 1000+ diverse images

- **Quality**: Higher resolution source images produce better results
  - Minimum: 400x400 pixels
  - Recommended: 600x600+ pixels
  - Processing: All images normalized to consistent size

- **Diversity**: Varied colors, subjects, and lighting improve matching
  - Include images with different dominant colors
  - Mix of bright and dark images
  - Various subjects and compositions

### Performance Tuning

```bash
# For faster processing (lower quality)
mzq normalize photos/ normalized/ --target-width 400 --target-height 400
mzq json normalized/image.jpg --num-tiles 9 --width 360 --height 360

# For highest quality (slower processing)
mzq normalize photos/ normalized/ --target-width 800 --target-height 800
mzq json normalized/image.jpg --num-tiles 25 --width 800 --height 800
```

## 🐛 Troubleshooting

### Common Issues

**"Need at least X images for YxY grid"**
- Solution: Add more images to your collection or reduce `num_tiles`

**"num_tiles must be odd"**
- Solution: Use odd numbers like 9, 15, 21, 25 instead of even numbers

**"width must be divisible by num_tiles"**
- Solution: Choose dimensions that divide evenly (e.g., 600÷15=40)

**"No supported image files found"**
- Solution: Ensure your folder contains JPEG, PNG, or other supported formats

**Memory errors with large collections**
- Solution: Process in smaller batches or reduce target image dimensions

### Getting Help

```bash
# General help
mzq --help

# Command-specific help
mzq normalize --help
mzq json --help
mzq frames --help
```

## 🔗 Links

- **Homepage**: [mozyq.org](https://mozyq.org)
- **Repository**: [github.com/mozyq/app](https://github.com/mozyq/app)
- **Issues**: [Report bugs and feature requests](https://github.com/mozyq/app/issues)

## 📄 License

This project is distributed under the terms specified in the project license.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Created with ❤️ by [Gilad Kutiel](mailto:gilad.kutiel@gmail.com)**
