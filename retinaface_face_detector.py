import warnings
import onnxruntime as ort
import numpy as np
from typing import Any, Dict, List
import time
import cv2
import logging

from modules.interfaces.detector import Detector
from processing import preprocess, postprocess

# Suppress warnings
warnings.filterwarnings("ignore")

class RetinaFaceDetector(Detector):
    def __init__(self, mode="gpu_optimized", threshold: float = 0.9, nms_threshold: float = 0.4, decay4: float = 0.5):
        self.mode = mode
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        self.decay4 = decay4

        session_options = ort.SessionOptions()
        if "optimized" in mode:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if "gpu" in mode:
            providers.append('CUDAExecutionProvider')
        elif "cpu" in mode:
            providers.append('CPUExecutionProvider')
        elif "npu" in mode:
            providers.append('TensorrtExecutionProvider')
        else:
            providers.append('CPUExecutionProvider') # Default fallback

        model_path = "weights/retinaface.onnx"
        self.model_session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

        self._feat_stride_fpn = [32, 16, 8]
        self._anchors_fpn = {
            "stride32": np.array([[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32),
            "stride16": np.array([[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32),
            "stride8": np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
        }
        self._num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    def detect_faces(self, image: np.ndarray, allow_upscaling: bool = True) -> List[Dict[str, Any]]:
        faces = []
        img = preprocess.get_image(image)
        im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
        inputs = {self.model_session.get_inputs()[0].name: im_tensor}
        net_out = self.model_session.run(None, inputs)
        net_out = [net_out[i] for i in [7, 1, 4, 8, 0, 3, 6, 2, 5]]

        proposals_list = []
        scores_list = []
        landmarks_list = []
        sym_idx = 0

        for stride in self._feat_stride_fpn:
            scores = net_out[sym_idx]
            scores = scores[:, :, :, self._num_anchors[f"stride{stride}"]:]
            bbox_deltas = net_out[sym_idx + 1]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
            A = self._num_anchors[f"stride{stride}"]
            K = height * width
            anchors_fpn = self._anchors_fpn[f"stride{stride}"]
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
            if stride == 4 and self.decay4 < 1.0:
                scores *= self.decay4
            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= self.threshold)[0]
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
        keep = postprocess.cpu_nms(pre_det, self.nms_threshold)
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        landmarks = landmarks[keep]

        if landmarks.ndim == 3 and landmarks.shape[1] == 5 and landmarks.shape[2] == 2:
            landmarks = landmarks.reshape(-1, 2)
        elif landmarks.ndim == 2 and landmarks.shape[1] == 2:
            pass
        else:
            logging.warning(f"Unexpected landmarks shape after processing: {landmarks.shape}")
            landmarks = landmarks.reshape(-1, 2)

        num_faces = det.shape[0]
        for i in range(num_faces):
            face_data = det[i]
            x1, y1, x2, y2, score = face_data[:5]
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            landmark_start = i * 5
            landmark_end = (i + 1) * 5
            landmark_points = landmarks[landmark_start:landmark_end]
            if landmark_points.shape[0] != 5:
                logging.warning(f"Unexpected number of landmarks for face {i+1}: {landmark_points.shape}")
                continue
            landmarks_dict = {
                "right_eye": list(landmark_points[0]), "left_eye": list(landmark_points[1]),
                "nose": list(landmark_points[2]), "mouth_right": list(landmark_points[3]),
                "mouth_left": list(landmark_points[4])
            }
            faces.append({"bbox": bbox, "landmarks": landmarks_dict, "confidence": float(score)})
        return faces

    def extract_faces(self, image: np.ndarray, align: bool = True, allow_upscaling: bool = True, expand_face_area: int = 0) -> List[np.ndarray]:
        extracted_faces = []
        try:
            faces = self.detect_faces(image, allow_upscaling=allow_upscaling)
            for face in faces:
                bbox = face["bbox"]
                x, y, w, h = bbox
                if expand_face_area > 0:
                    ew, eh = w + int(w * expand_face_area / 100), h + int(h * expand_face_area / 100)
                    x, y = max(0, x - int((ew - w) / 2)), max(0, y - int((eh - h) / 2))
                    w, h = min(image.shape[1] - x, ew), min(image.shape[0] - y, eh)
                x2, y2 = x + w, y + h
                facial_img = image[y:y2, x:x2]
                if align:
                    landmarks = face["landmarks"]
                    pts1 = np.float32([tuple(landmarks["left_eye"]), tuple(landmarks["right_eye"]), tuple(landmarks["nose"])])
                    pts2 = np.float32([(0.35 * 160, 0.35 * 160), (0.65 * 160, 0.35 * 160), (0.5 * 160, 0.55 * 160)])
                    M = cv2.getAffineTransform(pts1, pts2)
                    aligned_face = cv2.warpAffine(image, M, (160, 160))
                    facial_img = aligned_face[0:160, 0:160]
                else:
                    facial_img = cv2.resize(facial_img, (160, 160))
                extracted_faces.append(facial_img[:, :, ::-1])
        except Exception as e:
            logging.error(f"Error in extract_faces_retinaface: {e}")
        return extracted_faces

def visualize(image: np.ndarray, faces: List[Dict[str, Any]], box_color=(0, 255, 0), landmark_color=(0, 0, 255)):
    output = image.copy()
    for face in faces:
        bbox = face['bbox']
        x, y, w, h = bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
        landmarks = face['landmarks']
        for key, point in landmarks.items():
            cv2.circle(output, tuple(point), 2, landmark_color, 2)
        conf = face['confidence']
        cv2.putText(output, f'{conf:.4f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return output
