from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Tracker(ABC):
    """
    Abstract base class for face trackers.
    """
    @abstractmethod
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates the tracker with new detections.

        Args:
            detections (List[Dict[str, Any]]): A list of detections from the
                                               face detector.

        Returns:
            List[Dict[str, Any]]: A list of tracked objects, where each object
                                 is a dictionary containing information like
                                 'id' and 'bbox'.
        """
        pass
