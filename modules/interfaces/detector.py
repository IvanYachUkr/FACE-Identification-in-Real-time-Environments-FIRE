from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class Detector(ABC):
    """
    Abstract base class for face detectors.
    """
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in an image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                 represents a detected face and contains keys
                                 like 'bbox', 'landmarks', and 'confidence'.
        """
        pass

    @abstractmethod
    def extract_faces(self, image: np.ndarray, align: bool = True) -> List[np.ndarray]:
        """
        Extracts face images from a larger image.

        Args:
            image (np.ndarray): The input image.
            align (bool): Whether to perform face alignment.

        Returns:
            List[np.ndarray]: A list of extracted face images.
        """
        pass
