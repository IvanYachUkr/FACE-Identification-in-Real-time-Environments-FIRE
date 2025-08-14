import logging
from modules.interfaces.detector import Detector
from mediapipe_face_detector import MediapipeDetector
from retinaface_face_detector import RetinaFaceDetector
from yunet_face_detector import YuNetDetector

def initialize_detector(detector_type: str) -> Detector:
    detector_type = detector_type.lower()
    if detector_type == 'yunet':
        logging.info("Initialized Yunet face detector.")
        return YuNetDetector()
    elif detector_type == 'retinaface':
        logging.info("Initialized RetinaFace detector.")
        return RetinaFaceDetector()
    elif detector_type == 'mediapipe':
        logging.info("Initialized Mediapipe Face Detection.")
        return MediapipeDetector()
    else:
        raise ValueError("Invalid detector_type. Choose from 'yunet', 'retinaface', 'mediapipe'.")
