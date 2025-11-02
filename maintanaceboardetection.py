import cv2
import yaml
import numpy as np
import json
import os
from typing import List, Tuple


# --- 1. YAML LOADING FIX ---
def opencv_matrix_constructor(loader, node):
    """Tells PyYAML to treat the !!opencv-matrix tag as a regular dictionary mapping."""
    return loader.construct_mapping(node, deep=True)


yaml.SafeLoader.add_constructor('tag:yaml.org,2002:opencv-matrix', opencv_matrix_constructor)


# --- 2. CONFIGURATION ---
# Marker size in METERS (50 mm = 0.05 m)
TAG_SIZE = 0.05
REF_MARKER_ID = 1  # Marker ID designated as the origin for switch coordinates
AXIS_LENGTH = 0.01  # Fixed axis length in METERS (10mm) for a clean display


# --- 3. LOAD CONFIGURATION AND CALIBRATION DATA ---
try:
    # Load configuration
    # Correctly locate button_config.json relative to the script's location
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "button_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Correctly locate the calibration file relative to the script's location
    calib_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config["camera"]["calibration_file"])
    with open(calib_file_path, "r") as f:
        lines = f.readlines()
        # Handle the %YAML header that OpenCV sometimes adds
        content = ''.join(lines[1:]) if lines and lines[0].startswith("%YAML") else ''.join(lines)

    raw = yaml.safe_load(content)

    camera_matrix = np.array(raw["camera_matrix"]["data"]).reshape((3, 3)).astype(np.float32)
    dist_coeffs = np.array(raw["dist_coeffs"]["data"]).reshape((1, -1)).astype(np.float32)

except Exception as e:
    print(f"Error loading configuration or calibration: {e}")
    # In a real application, you might use placeholder values, but we exit here for safety.
    exit()


# --- 4. ARUCO SETUP ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Define the 3D object points for a single marker. The origin is the marker center.
# This is used for cv2.solvePnP.
marker_size_half = TAG_SIZE / 2.0
obj_points = np.array([
    [-marker_size_half,  marker_size_half, 0],  # Top-left
    [ marker_size_half,  marker_size_half, 0],  # Top-right
    [ marker_size_half, -marker_size_half, 0],  # Bottom-right
    [-marker_size_half, -marker_size_half, 0]   # Bottom-left
], dtype=np.float32)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


# --- HELPER FUNCTIONS ---
def project_box_corners(center_tvec: np.ndarray, rvec: np.ndarray, size_mm: List[float],
                        camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the 8 corners of the bounding box relative to its center,
    then projects them to the image plane using the component's full 3D pose.
    """
    W = size_mm[0] / 1000.0  # Width in meters
    H = size_mm[1] / 1000.0  # Height in meters
    D = 5.0 / 1000.0         # Depth/thickness in meters

    # Define the 8 corner points in the component's local frame (center is 0,0,0)
    box_obj_points = np.array([
        [-W/2,  H/2,  0], [-W/2, -H/2,  0], [W/2, -H/2,  0], [W/2,  H/2,  0],
        [-W/2,  H/2, -D], [-W/2, -H/2, -D], [W/2, -H/2, -D], [W/2,  H/2, -D]
    ], dtype=np.float32)

    # Project points using the component's full pose (rvec for rotation, center_tvec for translation)
    imgpts, _ = cv2.projectPoints(box_obj_points, rvec, center_tvec, camera_matrix, dist_coeffs)

    return imgpts.reshape(-1, 2).astype(int), box_obj_points


def draw_box(img, imgpts, color=(0, 255, 0)):
    """Draws a wireframe box given the projected 8 corner points."""
    # Front Face
    cv2.line(img, tuple(imgpts[0]), tuple(imgpts[3]), color, 2)
    cv2.line(img, tuple(imgpts[3]), tuple(imgpts[2]), color, 2)
    cv2.line(img, tuple(imgpts[2]), tuple(imgpts[1]), color, 2)
    cv2.line(img, tuple(imgpts[1]), tuple(imgpts[0]), color, 2)
    # Back Face
    cv2.line(img, tuple(imgpts[4]), tuple(imgpts[7]), color, 2)
    cv2.line(img, tuple(imgpts[7]), tuple(imgpts[6]), color, 2)
    cv2.line(img, tuple(imgpts[6]), tuple(imgpts[5]), color, 2)
    cv2.line(img, tuple(imgpts[5]), tuple(imgpts[4]), color, 2)
    # Connecting Edges
    cv2.line(img, tuple(imgpts[0]), tuple(imgpts[4]), color, 2)
    cv2.line(img, tuple(imgpts[1]), tuple(imgpts[5]), color, 2)
    cv2.line(img, tuple(imgpts[2]), tuple(imgpts[6]), color, 2)
    cv2.line(img, tuple(imgpts[3]), tuple(imgpts[7]), color, 2)


# --- 5. MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    # Check if any markers are detected
    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        ref_rvec, ref_tvec = None, None

        # --- PHASE 1: FIND REFERENCE MARKER POSE ---
        if REF_MARKER_ID in ids_flat:
            ref_index = np.where(ids_flat == REF_MARKER_ID)[0][0]
            ref_corners = corners[ref_index]

            # Estimate Pose of the Reference Marker using solvePnP
            # Note: cv2.aruco.estimatePoseSingleMarkers is deprecated
            success, rvec, tvec = cv2.solvePnP(obj_points, ref_corners, camera_matrix, dist_coeffs)

            if success:
                ref_rvec = rvec
                ref_tvec = tvec
                R_ref, _ = cv2.Rodrigues(ref_rvec)

                # --- PHASE 2: PROJECT SWITCH POSITIONS ---
                for switch in config["switches"]:
                    offset_mm = np.array(switch["offset_from_marker_mm"], dtype=np.float32)
                    rel_pos = (offset_mm / 1000.0).reshape((3, 1))

                    # 1. Transformation: Marker Frame -> Camera Frame (Calculates the 3D center of the component)
                    component_tvec = ref_tvec + R_ref @ rel_pos

                    # 2. Draw the Bounding Box Wireframe
                    # The box should have the same rotation as the reference marker
                    imgpts_box, _ = project_box_corners(component_tvec, ref_rvec, switch["size_mm"],
                                                        camera_matrix, dist_coeffs)
                    if imgpts_box.shape[0] == 8: # Ensure all points were projected
                        draw_box(frame, imgpts_box, color=(0, 255, 0))

                    # 3. Draw the Text Label near the component's projected center
                    # We project the 3D center point to find where to draw the text
                    center_point_3d = np.array([[[0,0,0]]], dtype=np.float32)
                    imgpts_center, _ = cv2.projectPoints(center_point_3d, ref_rvec, component_tvec, camera_matrix, dist_coeffs)

                    pt = tuple(imgpts_center[0][0].astype(int))

                    if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
                        cv2.putText(frame, switch["id"], (pt[0] + 10, pt[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # --- PHASE 3: DRAW AXIS ON ALL DETECTED MARKERS ---
        for i in range(len(ids)):
            current_corners = corners[i]

            # Recalculate pose for each marker to draw its own axis
            success, rvec_curr, tvec_curr = cv2.solvePnP(obj_points, current_corners, camera_matrix, dist_coeffs)

            if success:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                    rvec_curr, tvec_curr, AXIS_LENGTH)

    # Draw detected marker outlines
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("Button Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
