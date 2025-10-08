
# Camera Calibration with chessboard dataset

import cv2
import numpy as np
import glob
import os

#  Chessboard configuration
chessboard_size = (7, 11)  # Inner corners: one less than squares
square_size = 0.025       # Size of one square in meters (adjust if needed)

#  Path to your calibration images
image_folder = r'C:\Users\amaan\Downloads\data\imgs\rightcamera'
images = glob.glob(os.path.join(image_folder, '*.png'))

#  Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#  Prepare object points (3D points in real world space)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points
image_size = None

#  Process each image
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    print(f"{fname}: {' Detected' if ret else ' Not Detected'}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # Save valid image size

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(100)
    else:
        print(f" Chessboard not detected in: {fname}")

cv2.destroyAllWindows()

#  Calibrate the camera
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("\n Calibration successful!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    #  Save to YAML file
    cv_file = cv2.FileStorage("mono_calibration.yaml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", camera_matrix)
    cv_file.write("dist_coeffs", dist_coeffs)
    cv_file.release()
    print("\n Calibration saved to mono_calibration.yaml")
else:
    print("\n Calibration failed: No valid chessboard detections.")
