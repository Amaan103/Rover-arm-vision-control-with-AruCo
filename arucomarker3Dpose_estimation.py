import cv2
import numpy as np

# === Load camera calibration from OpenCV-style YAML ===
fs = cv2.FileStorage("mono_calibration.yml", cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    print("Failed to open file.")
else:
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coeffs:\n", dist_coeffs)
fs.release()

# === ArUco setup ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # or DICT_APRILTAG_36h11
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# === Marker size in meters ===
marker_length = 0.04  # 40 mm

# === Video capture ===
cap = cv2.VideoCapture(0)  # Use webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Detect markers ===
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        # === Estimate pose for each marker ===
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
