# mediapipe_face_detector.py

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any
import logging

# Initialize Mediapipe Face Detection
mp_face_detection = None
mp_drawing = mp.solutions.drawing_utils

def detect_faces(image: np.ndarray, conf_threshold: float = 0.85,
                model_selection: int = 1) -> List[Dict[str, Any]]:
    """
    Detect faces in an image using Mediapipe Face Detection.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        conf_threshold (float): Minimum confidence value ([0.0, 1.0]) for a detection to be considered successful.
        model_selection (int): 0 or 1. 0 for general model, 1 for short-range.

    Returns:
        list: A list of dictionaries, each containing 'bbox', 'landmarks', and 'confidence'.
    """
    global mp_face_detection

    if mp_face_detection is None:
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=conf_threshold)

    # Convert the BGR image to RGB before processing.
    results = mp_face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    faces = []
    h, w = image.shape[:2]

    if results.detections:
        for det in results.detections:
            # Extract bounding box
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)
            box_w = min(w - x, box_w)
            box_h = min(h - y, box_h)

            # Extract keypoints
            keypoints = det.location_data.relative_keypoints
            right_eye = keypoints[0]  # Right eye
            left_eye = keypoints[1]   # Left eye
            nose_tip = keypoints[2]   # Nose tip

            # Convert keypoints to absolute coordinates
            right_eye = (int(right_eye.x * w), int(right_eye.y * h))
            left_eye = (int(left_eye.x * w), int(left_eye.y * h))
            nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

            # Confidence score
            conf = det.score[0] if det.score else 0.0


            faces.append({
                'bbox': [x, y, box_w, box_h],
                'landmarks': [right_eye, left_eye, nose_tip],
                'confidence': conf
            })




    return faces

def visualize(image: np.ndarray, faces: List[Dict[str, Any]],
             box_color=(0, 255, 0), landmark_color=(0, 0, 255)):
    """
    Visualize detected faces on the image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        faces (list): List of detected faces with 'bbox', 'landmarks', and 'confidence'.
        box_color (tuple): Color for bounding boxes.
        landmark_color (tuple): Color for landmarks.

    Returns:
        numpy.ndarray: Image with visualizations.
    """
    output = image.copy()

    for face in faces:
        bbox = face['bbox']
        x, y, w, h = bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)

        landmarks = face['landmarks']
        for landmark in landmarks:
            cv2.circle(output, landmark, 2, landmark_color, 2)

        conf = face['confidence']
        cv2.putText(output, f'{conf:.4f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return output

def extract_faces(image: np.ndarray, threshold: float = 0.5, align: bool = True,
                 allow_upscaling: bool = True, expand_face_area: int = 0,
                 conf_threshold: float = 0.5, model_selection: int = 0) -> List[np.ndarray]:
    """
    Extract detected and aligned faces from an image using Mediapipe Face Detection.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold (float): Confidence threshold for face detection.
        align (bool): Whether to align the faces.
        allow_upscaling (bool): Allow upscaling of the image for detection.
        expand_face_area (int): Percentage to expand the facial area.
        conf_threshold (float): Confidence threshold for detection.
        model_selection (int): 0 for general model, 1 for short-range.

    Returns:
        list: List of extracted face images in RGB format.
    """
    extracted_faces = []

    try:
        faces = detect_faces(image, conf_threshold=conf_threshold,
                             model_selection=model_selection)

        for face in faces:
            bbox = face['bbox']
            x, y, w, h = bbox
            x2, y2 = x + w, y + h

            # Expand facial area if required
            if expand_face_area > 0:
                expanded_w = w + int(w * expand_face_area / 100)
                expanded_h = h + int(h * expand_face_area / 100)

                x = max(0, x - int((expanded_w - w) / 2))
                y = max(0, y - int((expanded_h - h) / 2))
                w = min(image.shape[1] - x, expanded_w)
                h = min(image.shape[0] - y, expanded_h)
                x2 = x + w
                y2 = y + h
            else:
                x2 = x + w
                y2 = y + h

            facial_img = image[y:y2, x:x2]

            if align and len(face['landmarks']) >= 3:
                right_eye = face['landmarks'][0]
                left_eye = face['landmarks'][1]
                nose_tip = face['landmarks'][2]

                # Define coordinates of the three points in the input image
                pts1 = np.float32([left_eye, right_eye, nose_tip])

                # Define coordinates of the three points in the aligned image
                desired_left_eye = (0.35 * 160, 0.35 * 160)
                desired_right_eye = (0.65 * 160, 0.35 * 160)
                desired_nose_tip = (0.5 * 160, 0.55 * 160)
                pts2 = np.float32([desired_left_eye, desired_right_eye, desired_nose_tip])

                # Compute the affine transformation matrix
                M = cv2.getAffineTransform(pts1, pts2)

                # Apply the affine transformation to the image
                aligned_face = cv2.warpAffine(image, M, (160, 160))

                # Extract the aligned facial area
                facial_img = aligned_face[0:160, 0:160]
            else:
                # If not aligning, just resize the cropped face
                facial_img = cv2.resize(facial_img, (160, 160))

            # Convert from BGR to RGB
            extracted_faces.append(facial_img[:, :, ::-1])

    except Exception as e:
        logging.error(f"Error in extract_faces_mediapipe: {e}")

    return extracted_faces
