# Aruco Based Pose Detection (Work in Progress)

This repository currently contains a basic pose detection script using AruCo and OpenCV. It loads a camera calibration file and visualizes the detected marker's pose using 3D axes. 
This is a foundational step toward a full control panel overlay system.

---

## What's Included

- `normal_pose_detection.py`: A standalone script that detects Aruco markers and draws pose axes using OpenCV.
- `camera_calibration.py`: Camera calibration script that gives the intrinsic parameters of the camera using a chessboard pattern.
- `panel_config.json`: Configuration file for the full overlay system (not yet implemented in code).
- `dataset/`: Folder containing all chessboard images used for camera calibration.

---

## Installation

Install required Python packages:

```bash
pip install opencv-python numpy
