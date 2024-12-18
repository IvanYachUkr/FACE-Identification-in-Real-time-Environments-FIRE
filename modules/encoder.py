# --------------------------------------------
# File: modules/encoder.py
# --------------------------------------------
import logging
import cv2
import numpy as np
from facenet_gpu import FaceNetClient

class Encoder:
    def __init__(self, encoder_model_type: str, encoder_mode: str):
        self.encoder = FaceNetClient(model_type=encoder_model_type, mode=encoder_mode)
        self.input_shape = self.encoder.input_shape
        self.output_shape = self.encoder.output_shape
        logging.info(f"Initialized FaceNet-{self.encoder.output_shape} encoder in {encoder_mode} mode.")

    def encode(self, face_img: np.ndarray) -> np.ndarray:
        return self.encoder(face_img)

    def preprocess_for_encoder(self, face_img: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(face_img, self.input_shape, interpolation=cv2.INTER_AREA)
        img = resized_img.astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            raise ValueError("Face image has incorrect shape for encoder.")
        img = np.expand_dims(img, axis=0)
        return img
