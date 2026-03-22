# OptiMeasure: Real-Time Computer Vision Dimensioning

![OptiMeasure Demo](assets/screenshot.png)

OptiMeasure is a computer vision application built with Python and OpenCV that performs real-time object measurement. The system utilizes a standard A4 reference sheet to calibrate spatial dimensions and extract real-world measurements from images or video feeds.

## Technical Implementation

The image processing pipeline is designed for robustness and accuracy:

- **Pre-processing:** Implements a 5x5 Gaussian Blur for noise reduction and Canny Edge Detection for contour identification.
- **Morphological Operations:** Uses a sequence of three dilation iterations followed by two erosion steps to ensure contour integrity.
- **Perspective Transformation:** Employs the `approxPolyDP` algorithm to isolate reference corners and calculates a homography matrix to warp the image into a normalized, top-down view.
- **Calibration:** Maps pixel distances to real-world metrics based on the known dimensions of an A4 reference (210mm x 297mm).
- **Dynamic Interface:** Includes an OpenCV-based control panel for real-time adjustment of thresholds and filters.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

## Usage

1. Place an A4 sheet of paper on a flat surface.
2. Place the object you wish to measure on the paper.
3. Run the script:
   ```bash
   python ObjMeasurement.py
   ```
4. Use the "Settings" window to adjust thresholds until the paper and object are clearly outlined.

## License

MIT License
