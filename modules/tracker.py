# --------------------------------------------
# File: modules/tracker.py
# --------------------------------------------
from sort_UKF import Sort

def initialize_tracker():
    return Sort(max_age=4, min_hits=4, iou_threshold=0.3)
