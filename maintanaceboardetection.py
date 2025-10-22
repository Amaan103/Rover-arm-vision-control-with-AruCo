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
REF_MARKER_ID = 1 # Marker ID designated as the origin for switch coordinates
AXIS_LENGTH = 0.01 # Fixed axis length in METERS (10mm) for a clean display

# --- 3. LOAD CONFIGURATION AND CALIBRATION DATA ---
try:
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "button_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    calib_file = config["camera"]["calibration_file"]

    # Load Calibration Data
    with open(calib_file, "r") as f:
        lines = f.readlines()
        content = ''.join(lines[1:]) if lines[0].startswith("%YAML") else ''.join(lines)
    
    raw = yaml.load(content, Loader=yaml.SafeLoader)

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

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# --- HELPER FUNCTION: Projects the 3D corners of a box into the 2D image plane ---
def project_box_corners(center_tvec: np.ndarray, R_camera: np.ndarray, size_mm: List[float], 
                        camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the 8 corners of the bounding box relative to its center, 
    transforms them to the camera frame, and projects them to the image plane.
    """
    W = size_mm[0] / 1000.0  # Width in meters
    H = size_mm[1] / 1000.0  # Height in meters
    # We assume Z (depth/thickness) is small, e.g., 5mm
    D = 5.0 / 1000.0 

    # Define the 8 corner points in the component's local frame (center is 0,0,0)
    # The order is: Front-Top-Left, Front-Top-Right, Front-Bottom-Right, Front-Bottom-Left
    # Followed by the same for the back face.
    obj_points = np.array([
        [-W/2, H/2, 0],   # 0: Front-Top-Left
        [ W/2, H/2, 0],   # 1: Front-Top-Right
        [ W/2,-H/2, 0],   # 2: Front-Bottom-Right
        [-W/2,-H/2, 0],   # 3: Front-Bottom-Left
        [-W/2, H/2, -D],  # 4: Back-Top-Left
        [ W/2, H/2, -D],  # 5: Back-Top-Right
        [ W/2,-H/2, -D],  # 6: Back-Bottom-Right
        [-W/2,-H/2, -D]   # 7: Back-Bottom-Left
    ], dtype=np.float32)
    
    # We use a zero rotation vector (no rotation relative to the component's frame, 
    # as the center_tvec already holds the correct world rotation/translation)
    rvec_zero = np.zeros((3, 1), dtype=np.float32) 
    
    # Project points using the center_tvec as the origin of the component's frame in the world
    imgpts, _ = cv2.projectPoints(obj_points, rvec_zero, center_tvec, camera_matrix, dist_coeffs)
    
    return imgpts.reshape(-1, 2).astype(int), obj_points # Return projected 2D points

def draw_box(img, imgpts, color=(0, 255, 0)):
    """Draws a wireframe box given the projected 8 corner points."""
    # Define connection lines for a 3D box (indices correspond to obj_points in project_box_corners)
    # Front Face
    cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]), color, 2)
    cv2.line(img, tuple(imgpts[1]), tuple(imgpts[2]), color, 2)
    cv2.line(img, tuple(imgpts[2]), tuple(imgpts[3]), color, 2)
    cv2.line(img, tuple(imgpts[3]), tuple(imgpts[0]), color, 2)
    
    # Connecting Edges (Front to Back)
    cv2.line(img, tuple(imgpts[0]), tuple(imgpts[4]), color, 2)
    cv2.line(img, tuple(imgpts[1]), tuple(imgpts[5]), color, 2)
    cv2.line(img, tuple(imgpts[2]), tuple(imgpts[6]), color, 2)
    cv2.line(img, tuple(imgpts[3]), tuple(imgpts[7]), color, 2)
    
    # Back Face
    cv2.line(img, tuple(imgpts[4]), tuple(imgpts[5]), color, 2)
    cv2.line(img, tuple(imgpts[5]), tuple(imgpts[6]), color, 2)
    cv2.line(img, tuple(imgpts[6]), tuple(imgpts[7]), color, 2)
    cv2.line(img, tuple(imgpts[7]), tuple(imgpts[4]), color, 2)


# --- 5. MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    # Check if any markers are detected
    if ids is not None and len(ids) > 0:
        ids_flat = ids.flatten()
        
        # Initialize variables for the reference marker
        ref_rvec, ref_tvec, R_ref = None, None, None
        
        # --- PHASE 1: FIND REFERENCE MARKER POSE (Needed for Switch Projection) ---
        if REF_MARKER_ID in ids_flat:
            ref_index = np.where(ids_flat == REF_MARKER_ID)[0][0]
            ref_corners = corners[ref_index] 
            
            # Estimate Pose of the Reference Marker
            rvec_ref_raw, tvec_ref_raw, _ = cv2.aruco.estimatePoseSingleMarkers(
                ref_corners, TAG_SIZE, camera_matrix, dist_coeffs
            )
            
            # **SHAPE FIX:** Explicitly reshape to (3, 1) column vectors
            ref_rvec = rvec_ref_raw[0][0].reshape((3, 1))
            ref_tvec = tvec_ref_raw[0][0].reshape((3, 1))

            R_ref, _ = cv2.Rodrigues(ref_rvec)

            # --- PHASE 2: PROJECT SWITCH POSITIONS (Now drawing bounding boxes) ---
            for switch in config["switches"]:
                
                offset_mm = np.array(switch["offset_from_marker_mm"], dtype=np.float32)
                # Convert to meters and ensure it's a (3, 1) column vector
                rel_pos = (offset_mm / 1000.0).reshape((3, 1)) 
                
                # 1. Transformation: Marker Frame -> Camera Frame (Calculates the 3D center of the component)
                component_tvec = ref_tvec + R_ref @ rel_pos

                # 2. Projection: Component Center to Image Plane (for text label placement)
                rvec_zero = np.zeros((3, 1), dtype=np.float32) 
                imgpts_center, _ = cv2.projectPoints(component_tvec.reshape(1, 3), 
                                                     rvec_zero, rvec_zero, # Use rvec_zero/tvec_zero for the camera frame projection
                                                     camera_matrix, dist_coeffs)
                
                pt = tuple(imgpts_center[0][0].astype(int))

                # 3. Draw the Bounding Box Wireframe
                imgpts_box, _ = project_box_corners(component_tvec, R_ref, switch["size_mm"], 
                                                    camera_matrix, dist_coeffs)
                
                draw_box(frame, imgpts_box, color=(0, 255, 0)) # Draw green box
                
                # 4. Draw the White Text Label (always drawn outside the box for clarity)
                if 0 <= pt[0] < frame.shape[1] and 0 <= pt[1] < frame.shape[0]:
                    cv2.putText(frame, switch["id"], (pt[0] + 5, pt[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- PHASE 3: DRAW AXIS ON ALL DETECTED MARKERS (Using reduced length) ---
        for i in range(len(ids_flat)):
            current_corners = corners[i]
            
            # Recalculate pose for the current marker
            rvec_curr_raw, tvec_curr_raw, _ = cv2.aruco.estimatePoseSingleMarkers(
                current_corners, TAG_SIZE, camera_matrix, dist_coeffs
            )
            
            rvec_curr = rvec_curr_raw[0][0]
            tvec_curr = tvec_curr_raw[0][0]
            
            # AXIS LENGTH FIX: Use the smaller AXIS_LENGTH (0.01m)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                              rvec_curr, tvec_curr, AXIS_LENGTH)

    # Draw detected markers (green squares on the tags themselves)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
    cv2.imshow("Button Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 6. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
