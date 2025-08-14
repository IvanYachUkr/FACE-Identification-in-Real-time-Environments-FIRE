from sort_UKF import SortTracker
from modules.interfaces.tracker import Tracker

def initialize_tracker() -> Tracker:
    return SortTracker(max_age=4, min_hits=4, iou_threshold=0.3)
