# yunet_face_detector.py

import numpy as np
import cv2 as cv
from yunet import YuNet

# Global YuNet model instance
model_yunet = None

def detect_faces(image, model_path='weights/face_detection_yunet_2023mar.onnx',
                conf_threshold=0.90, nms_threshold=0.3, top_k=5000):
    """
    Detect faces in an image using YuNet.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        model_path (str): Path to the YuNet ONNX model.
        conf_threshold (float): Confidence threshold for detections.
        nms_threshold (float): Non-maximum suppression threshold.
        top_k (int): Maximum number of faces to detect.

    Returns:
        list: A list of dictionaries, each containing 'bbox', 'landmarks', and 'confidence'.
    """
    global model_yunet

    # Get image dimensions
    h, w = image.shape[:2]

    # Instantiate YuNet model if it doesn't exist
    if model_yunet is None:
        model_yunet = YuNet(modelPath=model_path,
                            inputSize=[w, h],
                            confThreshold=conf_threshold,
                            nmsThreshold=nms_threshold,
                            topK=top_k,
                            backendId=cv.dnn.DNN_BACKEND_OPENCV,  # Default backend: OpenCV + CPU
                            targetId=cv.dnn.DNN_TARGET_CPU)       # Default target: CPU
    else:
        # Update input size if the image size has changed
        if model_yunet.inputSize != [w, h]:
            model_yunet.setInputSize([w, h])

    # Perform face detection
    results = model_yunet.infer(image)


    # Collect bounding boxes and landmarks
    faces = []
    if results is not None:
        for det in results:
            bbox = det[0:4].astype(np.int32)  # Bounding box: [x, y, width, height]
            landmarks = det[4:14].reshape((5, 2)).astype(
                np.int32)  # Landmarks: 5 points (right eye, left eye, nose, mouth corners)
            conf = det[-1]  # Confidence score
            faces.append({'bbox': bbox, 'landmarks': landmarks, 'confidence': conf})

    return faces

def visualize(image, faces, box_color=(0, 255, 0), landmark_color=(0, 0, 255)):
    """
    Visualize detected faces on the image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        faces (list): List of detected faces with 'bbox' and 'landmarks'.
        box_color (tuple): Color for bounding boxes.
        landmark_color (tuple): Color for landmarks.

    Returns:
        numpy.ndarray: Image with visualizations.
    """
    output = image.copy()

    for face in faces:
        bbox = face['bbox']
        x, y, w, h = bbox
        cv.rectangle(output, (x, y), (x + w, y + h), box_color, 2)

        for landmark in face['landmarks']:
            cv.circle(output, tuple(landmark), 2, landmark_color, 2)

        conf = face['confidence']
        cv.putText(output, '{:.4f}'.format(conf), (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return output

def extract_faces(image, threshold=0.9, align=True, allow_upscaling=True,
                 expand_face_area=0, conf_threshold=0.9, nms_threshold=0.3,
                 top_k=5000):
    """
    Extract detected and aligned faces from an image using YuNet.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold (float): Confidence threshold for face detection.
        align (bool): Whether to align the faces.
        allow_upscaling (bool): Allow upscaling of the image for detection.
        expand_face_area (int): Percentage to expand the facial area.
        conf_threshold (float): Confidence threshold for detection.
        nms_threshold (float): Non-maximum suppression threshold.
        top_k (int): Maximum number of faces to detect.

    Returns:
        list: List of extracted face images in RGB format.
    """
    resp = []

    try:
        faces = detect_faces(image, conf_threshold=conf_threshold,
                             nms_threshold=nms_threshold, top_k=top_k)

        for face in faces:
            bbox = face['bbox']
            x, y, w, h = bbox

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

            if align:
                landmarks = face['landmarks']
                right_eye = tuple(landmarks[0])
                left_eye = tuple(landmarks[1])
                nose_tip = tuple(landmarks[2])

                # Define coordinates of the three points in the input image
                pts1 = np.float32([left_eye, right_eye, nose_tip])

                # Define coordinates of the three points in the aligned image
                desired_left_eye = (0.35 * 160, 0.35 * 160)
                desired_right_eye = (0.65 * 160, 0.35 * 160)
                desired_nose_tip = (0.5 * 160, 0.55 * 160)
                pts2 = np.float32([desired_left_eye, desired_right_eye, desired_nose_tip])

                # Compute the affine transformation matrix
                M = cv.getAffineTransform(pts1, pts2)

                # Apply the affine transformation to the image
                aligned_face = cv.warpAffine(image, M, (160, 160))

                # Extract the aligned facial area
                rotated_coords = (0, 0, 160, 160)
                rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotated_coords
                facial_img = aligned_face[rotated_y1:rotated_y2, rotated_x1:rotated_x2]
            else:
                # If not aligning, just resize the cropped face
                facial_img = cv.resize(facial_img, (160, 160))

            # Convert from BGR to RGB
            resp.append(facial_img[:, :, ::-1])

    except Exception as e:
        print(f"Error in extract_faces_yunet: {e}")

    return resp