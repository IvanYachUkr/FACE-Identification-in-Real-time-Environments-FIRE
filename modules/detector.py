# --------------------------------------------
# File: modules/detector.py
# --------------------------------------------
import logging

def initialize_detector(detector_type: str):
    detector_type = detector_type.lower()
    if detector_type == 'yunet':
        from yunet_face_detector import detect_faces as detect_faces_impl, extract_faces as extract_faces_impl
        logging.info("Initialized Yunet face detector.")
    elif detector_type == 'retinaface':
        import retinaface_face_detector
        detect_faces_impl = retinaface_face_detector.detect_faces
        extract_faces_impl = retinaface_face_detector.extract_faces
        logging.info("Initialized RetinaFace detector.")
    elif detector_type == 'mediapipe':
        import mediapipe_face_detector
        detect_faces_impl = mediapipe_face_detector.detect_faces
        extract_faces_impl = mediapipe_face_detector.extract_faces
        logging.info("Initialized Mediapipe Face Detection.")
    else:
        raise ValueError("Invalid detector_type. Choose from 'yunet', 'retinaface', 'mediapipe'.")
    return detect_faces_impl, extract_faces_impl
