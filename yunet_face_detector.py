import numpy as np
import cv2 as cv
from typing import List, Dict, Any
from modules.interfaces.detector import Detector
from yunet import YuNet

class YuNetDetector(Detector):
    def __init__(self, model_path='weights/face_detection_yunet_2023mar.onnx',
                 conf_threshold=0.9, nms_threshold=0.3, top_k=5000,
                 backend_id=cv.dnn.DNN_BACKEND_OPENCV, target_id=cv.dnn.DNN_TARGET_CPU):
        self.model = YuNet(modelPath=model_path,
                           inputSize=[640, 480],  # Default, will be updated
                           confThreshold=conf_threshold,
                           nmsThreshold=nms_threshold,
                           topK=top_k,
                           backendId=backend_id,
                           targetId=target_id)

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        # YuNet's inputSize is a tuple, so we compare with a tuple.
        if self.model.inputSize != (w, h):
            self.model.setInputSize([w, h])

        results = self.model.infer(image)

        faces = []
        if results is not None:
            for det in results:
                bbox = det[0:4].astype(np.int32)
                landmarks = det[4:14].reshape((5, 2)).astype(np.int32)
                conf = det[-1]
                faces.append({'bbox': bbox.tolist(), 'landmarks': landmarks.tolist(), 'confidence': conf})
        return faces

    def extract_faces(self, image: np.ndarray, align: bool = True, expand_face_area: int = 0) -> List[np.ndarray]:
        extracted_faces = []
        try:
            faces = self.detect_faces(image)
            for face in faces:
                bbox = face['bbox']
                x, y, w, h = bbox

                if expand_face_area > 0:
                    ew = w + int(w * expand_face_area / 100)
                    eh = h + int(h * expand_face_area / 100)
                    x = max(0, x - int((ew - w) / 2))
                    y = max(0, y - int((eh - h) / 2))
                    w, h = min(image.shape[1] - x, ew), min(image.shape[0] - y, eh)

                x2, y2 = x + w, y + h
                facial_img = image[y:y2, x:x2]

                if align:
                    landmarks = np.array(face['landmarks'])
                    right_eye = tuple(landmarks[0])
                    left_eye = tuple(landmarks[1])
                    nose_tip = tuple(landmarks[2])

                    pts1 = np.float32([left_eye, right_eye, nose_tip])
                    pts2 = np.float32([(0.35 * 160, 0.35 * 160), (0.65 * 160, 0.35 * 160), (0.5 * 160, 0.55 * 160)])

                    M = cv.getAffineTransform(pts1, pts2)
                    aligned_face = cv.warpAffine(image, M, (160, 160))
                    facial_img = aligned_face[0:160, 0:160]
                else:
                    facial_img = cv.resize(facial_img, (160, 160))

                extracted_faces.append(facial_img[:, :, ::-1])
        except Exception as e:
            print(f"Error in extract_faces_yunet: {e}")
        return extracted_faces

def visualize(image, faces, box_color=(0, 255, 0), landmark_color=(0, 0, 255)):
    output = image.copy()
    for face in faces:
        bbox = face['bbox']
        x, y, w, h = bbox
        cv.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
        landmarks = np.array(face['landmarks'])
        for landmark in landmarks:
            cv.circle(output, tuple(landmark), 2, landmark_color, 2)
        conf = face['confidence']
        cv.putText(output, f'{conf:.4f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return output