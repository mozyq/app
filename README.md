# mozyq

<video src="https://github.com/user-attachments/assets/c1ac633a-3eaa-405d-8a90-749d68fcd234" autoplay muted></video>

mozyq is a Python command line tool that takes a folder containing multiple photos and generates a video of photography mosaics from them. This tool is useful for creating stunning mosaic videos where each frame is a mosaic made from the photos in the input folder.

## Requirements
You need ffmpeg:
```
sudo apt install ffmpeg
```

## Installation
```
pip install mozyq
```

## Usage

First you need to have a folder with enough photos.
You should have at least 500 630x630 photos.
You can create one like this:
```
mkdir photos && seq 999 | xargs -I {} -n 1 -P 32 wget https://picsum.photos/630 -O photos/{}.jpg
```



To generate a video run:
```
mzq photos/1.jpg
```
