# retinaface_face_detector.py

import warnings
import onnxruntime as ort
import numpy as np
from typing import Union, Any, Dict, List
import time
import cv2
import logging

from processing import preprocess, postprocess

# Suppress warnings
warnings.filterwarnings("ignore")

class ModelSingleton:
    _instances = {}

    def __new__(cls, mode="gpu_optimized"):
        if mode not in cls._instances:
            cls._instances[mode] = super(ModelSingleton, cls).__new__(cls)
        return cls._instances[mode]

    def __init__(self, mode="gpu_optimized"):
        if getattr(self, 'initialized', False):
            return
        self.mode = mode
        session_options = ort.SessionOptions()

        # Choose the execution mode
        if mode == "gpu":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif mode == "gpu_optimized":
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif mode == "cpu":
            providers = ['CPUExecutionProvider']
        elif mode == "cpu_optimized":
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CPUExecutionProvider']
        elif mode == "npu":
            providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        elif mode == "npu_optimized":
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        else:
            raise ValueError(
                f"Invalid mode selected: {mode}. Choose from 'gpu', 'gpu_optimized', 'cpu', 'cpu_optimized', 'npu', 'npu_optimized'.")

        # Load ONNX model with the chosen providers
        model_path = "weights/retinaface.onnx"
        self.model_session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.initialized = True

def detect_faces(image: np.ndarray, threshold: float = 0.9,
                allow_upscaling: bool = True, mode: str = "gpu_optimized") -> List[Dict[str, Any]]:
    """
    Detect faces in an image using RetinaFace.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold (float): Confidence threshold for detections.
        allow_upscaling (bool): Allow upscaling of the image.
        mode (str): Execution mode for ONNX runtime.

    Returns:
        List[Dict[str, Any]]: Detected faces with 'bbox', 'landmarks', and 'confidence'.
    """
    faces = []
    img = preprocess.get_image(image)

    # Use the singleton model instance
    model_instance = ModelSingleton(mode=mode)
    model_retinaface = model_instance.model_session

    nms_threshold = 0.4
    decay4 = 0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        "stride32": np.array(
            [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32
        ),
        "stride16": np.array(
            [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32
        ),
        "stride8": np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
    }

    _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    # Preprocess the image
    im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)

    # Run inference using the ONNX model
    inputs = {model_retinaface.get_inputs()[0].name: im_tensor}

    # Measure the time to extract faces
    start_time = time.time()

    net_out = model_retinaface.run(None, inputs)

    # Calculate execution time
    execution_time = time.time() - start_time

    # Adjust net_out order according to the permutation in the ONNX model
    net_out = [net_out[i] for i in [7, 1, 4, 8, 0, 3, 6, 2, 5]]  # Adjust output order

    proposals_list = []
    scores_list = []
    landmarks_list = []
    sym_idx = 0

    for stride in _feat_stride_fpn:
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors[f"stride{stride}"]:]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors[f"stride{stride}"]
        K = height * width
        anchors_fpn = _anchors_fpn[f"stride{stride}"]
        anchors = postprocess.anchors_plane(height, width, stride, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] *= bbox_stds[0]
        bbox_deltas[:, 1::4] *= bbox_stds[1]
        bbox_deltas[:, 2::4] *= bbox_stds[2]
        bbox_deltas[:, 3::4] *= bbox_stds[3]
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)
        proposals = postprocess.clip_boxes(proposals, im_info[:2])

        if stride == 4 and decay4 < 1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3] // A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)

    if proposals.shape[0] == 0:
        return faces

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
    keep = postprocess.cpu_nms(pre_det, nms_threshold)

    det = np.hstack((pre_det, proposals[:, 4:]))
    det = det[keep, :]
    landmarks = landmarks[keep]

    # **Reshape Landmarks to Ensure Correct Dimensions**
    # If landmarks have shape (num_faces, 5, 2), reshape to (num_faces * 5, 2)
    if landmarks.ndim == 3 and landmarks.shape[1] == 5 and landmarks.shape[2] == 2:
        landmarks = landmarks.reshape(-1, 2)
    elif landmarks.ndim == 2 and landmarks.shape[1] == 2:
        pass  # Already in correct shape
    else:
        logging.warning(f"Unexpected landmarks shape after processing: {landmarks.shape}")
        landmarks = landmarks.reshape(-1, 2)  # Attempt to flatten

    # Correctly assign landmarks to each face
    num_faces = det.shape[0]
    for i in range(num_faces):
        face = det[i]
        x1, y1, x2, y2, score = face[:5]
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # [x, y, w, h]

        # Each face has 5 landmarks: right_eye, left_eye, nose, mouth_right, mouth_left
        landmark_start = i * 5
        landmark_end = (i + 1) * 5
        landmark_points = landmarks[landmark_start:landmark_end]

        # Ensure that we have exactly 5 landmarks
        if landmark_points.shape[0] != 5:
            logging.warning(f"Unexpected number of landmarks for face {i+1}: {landmark_points.shape}")
            continue

        landmarks_dict = {
            "right_eye": list(landmark_points[0]),
            "left_eye": list(landmark_points[1]),
            "nose": list(landmark_points[2]),
            "mouth_right": list(landmark_points[3]),
            "mouth_left": list(landmark_points[4])
        }

        faces.append({
            "bbox": bbox,
            "landmarks": landmarks_dict,
            "confidence": float(score)
        })

    return faces

def visualize(image: np.ndarray, faces: List[Dict[str, Any]],
             box_color=(0, 255, 0), landmark_color=(0, 0, 255)):
    """
    Visualize detected faces on the image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        faces (list): Detected faces with 'bbox', 'landmarks', and 'confidence'.
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
        for key, point in landmarks.items():
            cv2.circle(output, tuple(point), 2, landmark_color, 2)

        conf = face['confidence']
        cv2.putText(output, f'{conf:.4f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return output

def extract_faces(image: np.ndarray, threshold: float = 0.9, align: bool = True,
                 allow_upscaling: bool = True, expand_face_area: int = 0,
                 mode: str = "gpu_optimized") -> List[np.ndarray]:
    """
    Extract detected and aligned faces from an image using RetinaFace.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold (float): Confidence threshold for face detection.
        align (bool): Whether to align the faces.
        allow_upscaling (bool): Allow upscaling of the image for detection.
        expand_face_area (int): Percentage to expand the facial area.
        mode (str): Execution mode for ONNX runtime.

    Returns:
        list: List of extracted face images in RGB format.
    """
    extracted_faces = []

    try:
        faces = detect_faces(image, threshold=threshold,
                             allow_upscaling=allow_upscaling, mode=mode)

        for face in faces:
            bbox = face["bbox"]
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

            facial_img = image[y:y2, x:x2]

            if align:
                landmarks = face["landmarks"]
                left_eye = tuple(landmarks["left_eye"])
                right_eye = tuple(landmarks["right_eye"])
                nose = tuple(landmarks["nose"])

                # Define coordinates of the three points in the input image
                pts1 = np.float32([left_eye, right_eye, nose])

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
        logging.error(f"Error in extract_faces_retinaface: {e}")

    return extracted_faces
