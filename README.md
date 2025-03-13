# Remove Moving Objects from Images

## Overview
**Remove Moving Objects** is a Python script that removes moving objects from a set of images taken from the same location. The script aligns the images and computes a static background by applying a median filter. This is particularly useful for scenarios such as:
- Capturing a clean view of a tourist attraction without people.
- Creating a background image of a street scene without cars.
- Enhancing time-lapse photography by removing transient objects.

## Features
- Uses ORB (Oriented FAST and Rotated BRIEF) for feature detection and FLANN for image alignment.
- Applies a median filter to remove moving objects and extract the static background.
- Processes images in parallel for improved performance.
- POSIX-compliant command-line interface.

## Installation
### Prerequisites
Ensure you have Python installed along with the following dependencies:
```sh
pip install opencv-python numpy imageio psutil tqdm
```

## Usage
Run the script with the following command:
```sh
python script.py /path/to/images output.jpg
```
### Parameters
- `/path/to/images` - Directory containing the images (.jpg format).
- `output.jpg` - Filename for the resulting background image.

If incorrect parameters are provided, the script will display usage instructions.

## Algorithm Explanation
1. **Image Alignment**
   - ORB (Oriented FAST and Rotated BRIEF) detects keypoints.
   - FLANN (Fast Library for Approximate Nearest Neighbors) matches keypoints.
   - Homography transformation is applied to align images.

2. **Background Extraction**
   - Images are stacked and split into color channels.
   - The median of each pixel value across images is computed.
   - The channels are merged to produce a clean background image.

## Example Use Case
Suppose you have a set of images taken from a busy square. Running:
```sh
python script.py images/ cleaned_background.jpg
```
Will generate `cleaned_background.jpg` with moving objects removed.

## License
This project is open-source and available under the MIT License.

