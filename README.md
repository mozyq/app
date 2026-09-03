# 🎨 mozyq

<video src="https://github.com/user-attachments/assets/c1ac633a-3eaa-405d-8a90-749d68fcd234" autoplay muted></video>

**mozyq** is a Python command-line tool (`mzq`) that turns a collection of photographs into a mosaic video. Each frame of the video is a mosaic: a grid of small tile images, drawn from your photo collection, arranged so that from a distance they reproduce a "master" image. The video zooms out from a single tile to the full mosaic, then the tile it started on becomes the next master — chaining several mosaics into one continuous animation.

## ✨ Features

- 🖼️ **Photo mosaics** from any folder of images
- 🎬 **Zoom-transition videos** that chain multiple mosaics together
- 🎯 **Optimal tile matching** via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
- 🌈 **Perceptual color matching** in LAB color space (OpenCV)
- 📐 **Square or rectangular** tiles and output
- 🛠️ **Built-in image normalization** to prepare a collection

## 🚀 Quick start

### Prerequisites

- **Python 3.10–3.12**
- **ffmpeg** on your `PATH` — used to encode frames into a video:

  ```bash
  sudo apt install ffmpeg     # Debian/Ubuntu
  brew install ffmpeg         # macOS
  choco install ffmpeg        # Windows
  ```

### Install

```bash
pip install mozyq
```

### Minimal pipeline

```bash
# 1. Collect at least grid_size² images in a folder (81 for the default 9×9).
mkdir photos
seq 200 | xargs -I{} -P8 wget -q https://picsum.photos/630 -O photos/{}.jpg

# 2. Normalize them to a uniform size and format.
mzq normalize photos/ normalized/

# 3. Build the mosaic chain from a seed image (tiles come from its folder).
mzq json normalized/0000.jpg mzq.json --grid-size 9

# 4. Render the video frames.
mzq frames mzq.json frames/

# 5. Encode with ffmpeg.
ffmpeg -r 30 -i frames/%04d.jpg -c:v libx264 -pix_fmt yuv420p mosaic.mp4
```

## 📚 Commands

Run `mzq --help` or `mzq <command> --help` for the authoritative, up-to-date usage.

Supported image extensions everywhere: `.jpg .jpeg .png .bmp .tiff .tif .webp`.

### `mzq normalize`

```bash
mzq normalize IN_FOLDER OUT_FOLDER [OPTIONS]
```

Reads every image in `IN_FOLDER`, resizes it to `--target-width` (preserving aspect ratio), center-crops it to `--target-height`, converts to RGB, and writes sequentially numbered JPEGs (`0000.jpg`, `0001.jpg`, …) into `OUT_FOLDER`. Images smaller than `--min-width` × `--min-height`, or whose aspect ratio makes them too short to crop to the target height, are skipped.

| Option | Default | Meaning |
| --- | --- | --- |
| `--min-width` | `630` | Skip inputs narrower than this |
| `--min-height` | `630` | Skip inputs shorter than this |
| `--target-width` | `630` | Output width |
| `--target-height` | `630` | Output height |

### `mzq json`

```bash
mzq json MASTER [OUTPUT_JSON] [OPTIONS]
```

Builds a chain of mosaics and writes it to `OUTPUT_JSON` (default `mzq.json`). The tiles are taken from **the folder containing `MASTER`**. For each transition it solves an optimal tile-to-cell assignment for the current master, then picks a random tile in the result to become the next master — repeating up to `--max-transitions` times (stopping early if a master repeats).

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `MASTER` | — | Seed image; its folder supplies the tiles |
| `OUTPUT_JSON` | `mzq.json` | Output path |
| `--grid-size` | `9` | Tiles per row/column; **must be odd** |
| `--max-transitions` | `10` | Maximum number of mosaics in the chain |

Requires at least `grid_size²` images in the tile folder (81 for `--grid-size 9`, 441 for `--grid-size 21`).

### `mzq frames`

```bash
mzq frames MZQ_JSON OUT_FOLDER [OPTIONS]
```

Reads a JSON file produced by `mzq json` and renders the zoom animation as sequentially numbered JPEG frames (`0000.jpg`, `0001.jpg`, …) in `OUT_FOLDER`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--fpt` | `60` | Frames rendered per transition |

Feed the frames to ffmpeg to produce the final video:

```bash
ffmpeg -r 30 -i OUT_FOLDER/%04d.jpg -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p mosaic.mp4
```

### `mzq json-full`

```bash
mzq json-full TILE_FOLDER [OUTPUT_JSON] [OPTIONS]
```

Computes the best tile arrangement for **every** image in `TILE_FOLDER` (each image treated as its own master) and writes a `{image_stem: [tile_stem, …]}` map to `OUTPUT_JSON` (default `full_mzq.json`). This is a bulk lookup table, not a video description — `mzq frames` does not consume it.

| Option | Default | Meaning |
| --- | --- | --- |
| `--grid-size` | `16` | Tiles per row/column |

Requires at least `grid_size²` images in `TILE_FOLDER`.

## 🔧 How it works

1. **Normalize** the collection to uniform dimensions.
2. **Vectorize** each tile: downscale in LAB color space and flatten to a feature vector.
3. **Split** the master into a `grid_size × grid_size` set of cells.
4. **Assign** tiles to cells by minimizing total Euclidean distance in LAB space with the Hungarian algorithm.
5. **Animate**: `mzq frames` renders a cosine-eased zoom out from the start tile to the full mosaic, cross-fading toward the master image near the end of each transition.

Each output frame is the size of the normalized images, so a larger `mzq normalize` target produces a higher-resolution video.

## 🐛 Troubleshooting

| Message | Fix |
| --- | --- |
| `Need at least N images for GxG grid` | Add more images, or lower `--grid-size` |
| `num_tiles must be odd` | `mzq json` needs an odd `--grid-size` (9, 15, 21, …) |
| `No supported image files found` | Folder has no `.jpg/.png/…` files |
| `Could not read image from …` | A file is corrupt or not a real image |
| `ffmpeg: command not found` | Install ffmpeg (see Prerequisites) |

## 🔗 Links

- **Homepage**: [mozyq.org](https://mozyq.org)
- **Repository**: [github.com/mozyq/app](https://github.com/mozyq/app)
- **Issues**: [github.com/mozyq/app/issues](https://github.com/mozyq/app/issues)

---

**Created with ❤️ by [Gilad Kutiel](mailto:gilad.kutiel@gmail.com)**
