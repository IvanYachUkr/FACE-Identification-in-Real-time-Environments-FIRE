# import cv2
# import time
# import logging
# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import MerweScaledSigmaPoints
# from yunet_face_detector import detect_faces  # Ensure this is properly implemented or imported
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# def compute_iou(bb_test, bb_gt):
#     """
#     Computes IOU between two bounding boxes.
#     """
#     xx1 = np.maximum(bb_test[0], bb_gt[0])
#     yy1 = np.maximum(bb_test[1], bb_gt[1])
#     xx2 = np.minimum(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
#     yy2 = np.minimum(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
#     w = np.maximum(0., xx2 - xx1)
#     h = np.maximum(0., yy2 - yy1)
#     wh = w * h
#     o = wh / ((bb_test[2] * bb_test[3]) + (bb_gt[2] * bb_gt[3]) - wh)
#     return o
#
# def non_max_suppression(detections, iou_threshold=0.3):
#     """
#     Performs Non-Max Suppression (NMS) on the detections.
#
#     Args:
#         detections (list): List of detections, each a dict with 'bbox' and 'confidence'.
#         iou_threshold (float): IOU threshold for suppression.
#
#     Returns:
#         list: Detections after NMS.
#     """
#     if len(detections) == 0:
#         return []
#
#     # Extract bounding boxes and confidence scores
#     bboxes = np.array([det['bbox'] for det in detections])
#     scores = np.array([det['confidence'] for det in detections])
#
#     x1 = bboxes[:, 0]
#     y1 = bboxes[:, 1]
#     x2 = bboxes[:, 0] + bboxes[:, 2]
#     y2 = bboxes[:, 1] + bboxes[:, 3]
#
#     areas = (x2 - x1) * (y2 - y1)
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#
#         # Compute IoU of the kept box with the rest
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#         # Keep boxes with IoU less than the threshold
#         inds = np.where(ovr <= iou_threshold)[0]
#         order = order[inds + 1]
#
#     # Return the filtered detections
#     return [detections[idx] for idx in keep]
#
# def filter_detections_near_edges(detections, frame_width, frame_height, margin=50):
#     """
#     Filters out detections that are too close to the frame edges.
#
#     Args:
#         detections (list): List of detections, each a dict with 'bbox' and 'confidence'.
#         frame_width (int): Width of the frame in pixels.
#         frame_height (int): Height of the frame in pixels.
#         margin (int): Margin in pixels from each frame edge.
#
#     Returns:
#         list: Filtered detections.
#     """
#     filtered = []
#     for det in detections:
#         x, y, w, h = det['bbox']
#         if (x >= margin and
#             y >= margin and
#             (x + w) <= (frame_width - margin) and
#             (y + h) <= (frame_height - margin)):
#             filtered.append(det)
#     return filtered
#
# class Track:
#     def __init__(self, bbox, track_id, max_age=30, min_area=np.log(10), min_aspect_ratio=np.log(0.5)):
#         """
#         Initializes a Track object with a bounding box.
#
#         Args:
#             bbox (list): Bounding box [x, y, w, h].
#             track_id (int): Unique identifier for the track.
#             max_age (int): Maximum number of frames to keep the track without updates.
#             min_area (float): Minimum logarithmic area to prevent collapse.
#             min_aspect_ratio (float): Minimum logarithmic aspect ratio to prevent collapse.
#         """
#         # Define UKF for tracking with log-transformed size variables and constant acceleration
#         points = MerweScaledSigmaPoints(n=10, alpha=0.1, beta=2., kappa=0.)
#         self.ukf = UKF(dim_x=10, dim_z=4, fx=self.f_process, hx=self.h_measurement, dt=1., points=points)
#
#         # Process noise and measurement noise covariance matrices
#         # Increased process noise for velocity and acceleration components
#         q_pos = 1.0    # Position process noise
#         q_vel = 10.0   # Velocity process noise
#         q_acc = 100.0  # Acceleration process noise
#         self.ukf.Q = np.diag([
#             q_pos, q_pos,      # cx, cy
#             q_acc, q_acc,      # log_s, log_r
#             q_vel, q_vel,      # vx, vy
#             q_acc, q_acc,      # v_log_s, (additional process noise if needed)
#             q_acc, q_acc       # ax, ay
#         ])
#
#         self.ukf.R = np.eye(4) * 10    # Measurement covariance
#
#         # Initialize state with the first bounding box, using log(s) and log(r)
#         x, y, w, h = bbox
#         s = w * h
#         r = w / h if h != 0 else 1.0  # Prevent division by zero
#         log_s = np.log(s) if s > 0 else self.min_area
#         log_r = np.log(r) if r > 0 else self.min_aspect_ratio
#         self.ukf.x = np.array([
#             x + w / 2,          # cx
#             y + h / 2,          # cy
#             log_s,              # log_s
#             log_r,              # log_r
#             0,                  # vx
#             0,                  # vy
#             0,                  # v_log_s
#             0,                  # ax
#             0,                  # ay
#             0                   # a_log_s
#         ])  # [cx, cy, log_s, log_r, vx, vy, v_log_s, ax, ay, a_log_s]
#
#         self.id = track_id
#         self.age = 0
#         self.time_since_update = 0
#         self.max_age = max_age
#         self.hits = 0
#         self.hit_streak = 0
#         self.min_area = min_area
#         self.min_aspect_ratio = min_aspect_ratio
#
#         # Define maximum allowed velocities and accelerations to prevent extreme movements
#         self.max_velocity = 1000  # pixels per frame
#         self.max_acceleration = 5000  # pixels per frame^2
#
#     def f_process(self, x, dt):
#         """
#         Process model function for the UKF.
#         Assumes a constant acceleration model with no control input.
#         """
#         F = np.array([
#             [1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
#             [0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
#             [0, 0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2],
#             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
#             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#         ])
#         return np.dot(F, x)
#
#     def h_measurement(self, x):
#         """
#         Measurement function for the UKF.
#         Maps the state vector to the measurement space (cx, cy, log_s, log_r).
#         """
#         return np.array([x[0], x[1], x[2], x[3]])
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         self.ukf.predict()
#         self.age += 1
#         self.time_since_update += 1
#
#         # Clamp velocity and acceleration to prevent extreme predictions
#         self.clamp_state()
#
#         return self.get_state()
#
#     def update(self, bbox):
#         """
#         Updates the state vector with observed bbox.
#         """
#         x, y, w, h = bbox
#         s = w * h
#         r = w / h if h != 0 else 1.0  # Prevent division by zero
#         log_s = np.log(s) if s > 0 else self.min_area
#         log_r = np.log(r) if r > 0 else self.min_aspect_ratio
#         z = np.array([x + w / 2, y + h / 2, log_s, log_r])
#         self.ukf.update(z)
#         self.time_since_update = 0
#         self.hits += 1
#         self.hit_streak += 1
#
#         # Clamp velocity and acceleration after update
#         self.clamp_state()
#
#     def clamp_state(self):
#         """
#         Clamps the velocity and acceleration components of the state vector
#         to prevent extreme values.
#         """
#         # Extract velocity and acceleration components
#         vx, vy, v_log_s, ax, ay, a_log_s = self.ukf.x[4:10]
#
#         # Clamp velocities
#         vx = np.clip(vx, -self.max_velocity, self.max_velocity)
#         vy = np.clip(vy, -self.max_velocity, self.max_velocity)
#         v_log_s = np.clip(v_log_s, -self.max_velocity, self.max_velocity)
#
#         # Clamp accelerations
#         ax = np.clip(ax, -self.max_acceleration, self.max_acceleration)
#         ay = np.clip(ay, -self.max_acceleration, self.max_acceleration)
#         a_log_s = np.clip(a_log_s, -self.max_acceleration, self.max_acceleration)
#
#         # Update the state vector with clamped values
#         self.ukf.x[4:10] = np.array([vx, vy, v_log_s, ax, ay, a_log_s])
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         Ensures that the bounding box size does not collapse below minimum thresholds.
#         """
#         x = self.ukf.x[0]
#         y = self.ukf.x[1]
#         log_s = self.ukf.x[2]
#         log_r = self.ukf.x[3]
#
#         # Enforce minimum log_s and log_r
#         log_s = max(log_s, self.min_area)
#         log_r = max(log_r, self.min_aspect_ratio)
#
#         s = np.exp(log_s)
#         r = np.exp(log_r)
#
#         # Prevent extremely small or large sizes
#         min_w = 10   # Minimum width
#         min_h = 10   # Minimum height
#         max_w = 1000 # Maximum width
#         max_h = 1000 # Maximum height
#
#         w = np.sqrt(s * r)
#         h = s / w
#
#         # Clamp width and height to reasonable bounds
#         w = np.clip(w, min_w, max_w)
#         h = np.clip(h, min_h, max_h)
#
#         return [int(x - w / 2), int(y - h / 2), int(w), int(h)]
#
# class Sort:
#     def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age  # Frames to keep alive a track without associated detections
#         self.min_hits = min_hits  # Minimum number of associated detections before track is confirmed
#         self.iou_threshold = iou_threshold
#         self.tracks = []
#         self.next_id = 0
#
#         # Define maximum allowable distance for measurement gating
#         self.max_distance = 200  # pixels
#
#     def update(self, detections):
#         """
#         Params:
#           detections - a list of detections, each in the format:
#                        {'bbox': [x, y, w, h], 'confidence': float}
#         Requires:
#           this method must be called once for each frame even with empty detections.
#         Returns:
#           a list of tracks, each track is a dict with 'id', 'bbox', and 'age'
#         """
#         # Predict new locations of existing tracks
#         for trk in self.tracks:
#             trk.predict()
#
#         trks = [trk.get_state() for trk in self.tracks]
#
#         # Create the cost matrix (1 - IoU) with measurement gating
#         matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)
#
#         # Update matched tracks with assigned detections
#         for t, trk in enumerate(self.tracks):
#             if t not in unmatched_trks:
#                 d = matched[t]
#                 trk.update(detections[d]['bbox'])
#
#         # Create and initialize new tracks for unmatched detections
#         for i in unmatched_dets:
#             trk = Track(detections[i]['bbox'], self.next_id, self.max_age)
#             self.next_id += 1
#             self.tracks.append(trk)
#
#         # Remove dead tracks
#         self.tracks = [trk for trk in self.tracks if trk.time_since_update <= self.max_age]
#
#         # Prepare output
#         output_tracks = []
#         for trk in self.tracks:
#             if trk.hits >= self.min_hits:  # Only output confirmed tracks
#                 bbox = trk.get_state()
#                 output_tracks.append({
#                     'id': trk.id,
#                     'bbox': bbox,
#                     'age': trk.age
#                 })
#
#         return output_tracks
#
#     def associate_detections_to_trackers(self, detections, trackers):
#         """
#         Assigns detections to tracked objects (both represented as bounding boxes).
#         Implements measurement gating to limit associations based on maximum allowable distance.
#
#         Returns:
#             matched_indices: dict mapping tracker index to detection index
#             unmatched_detections: list of detection indices not matched
#             unmatched_trackers: list of tracker indices not matched
#         """
#         if len(trackers) == 0:
#             unmatched_detections = list(range(len(detections)))
#             return {}, unmatched_detections, []
#
#         iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
#         confidence_scores = np.array([det['confidence'] for det in detections])
#
#         for d, det in enumerate(detections):
#             for t, trk in enumerate(trackers):
#                 iou = compute_iou(det['bbox'], trk)
#                 iou_matrix[d, t] = iou * confidence_scores[d]
#
#         # Apply linear assignment
#         matched_indices = linear_sum_assignment(-iou_matrix)
#         matched_indices = np.array(matched_indices).T
#
#         # Initialize matched, unmatched detections and trackers
#         matches = {}
#         unmatched_detections = []
#         unmatched_trackers = []
#
#         for d, t in matched_indices:
#             # Compute the Euclidean distance between detection and tracker centers
#             det_bbox = detections[d]['bbox']
#             trk_bbox = trackers[t]
#             det_center = np.array([det_bbox[0] + det_bbox[2] / 2, det_bbox[1] + det_bbox[3] / 2])
#             trk_center = np.array([trk_bbox[0] + trk_bbox[2] / 2, trk_bbox[1] + trk_bbox[3] / 2])
#             distance = np.linalg.norm(det_center - trk_center)
#
#             # Define maximum allowable distance
#             max_distance = self.max_distance  # pixels
#
#             if distance > max_distance:
#                 # If the detection is too far from the tracker, consider it unmatched
#                 unmatched_detections.append(d)
#                 unmatched_trackers.append(t)
#             else:
#                 matches[t] = d
#
#         # Detections not matched
#         for d in range(len(detections)):
#             if d not in matched_indices[:, 0]:
#                 unmatched_detections.append(d)
#
#         # Trackers not matched
#         for t in range(len(trackers)):
#             if t not in matched_indices[:, 1]:
#                 unmatched_trackers.append(t)
#
#         return matches, unmatched_detections, unmatched_trackers
#
#
#
import cv2
#
# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import MerweScaledSigmaPoints
#
# def compute_iou(bb_test, bb_gt):
#     """
#     Computes IOU between two bounding boxes.
#     """
#     xx1 = np.maximum(bb_test[0], bb_gt[0])
#     yy1 = np.maximum(bb_test[1], bb_gt[1])
#     xx2 = np.minimum(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
#     yy2 = np.minimum(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
#     w = np.maximum(0., xx2 - xx1)
#     h = np.maximum(0., yy2 - yy1)
#     wh = w * h
#     o = wh / ((bb_test[2] * bb_test[3]) + (bb_gt[2] * bb_gt[3]) - wh)
#     return o
#
# class Track:
#     def __init__(self, bbox, track_id, max_age=30, min_area=np.log(10), min_aspect_ratio=np.log(0.5)):
#         """
#         Initializes a Track object with a bounding box.
#
#         Args:
#             bbox (list): Bounding box [x, y, w, h].
#             track_id (int): Unique identifier for the track.
#             max_age (int): Maximum number of frames to keep the track without updates.
#             min_area (float): Minimum logarithmic area to prevent collapse.
#             min_aspect_ratio (float): Minimum logarithmic aspect ratio to prevent collapse.
#         """
#         # Define UKF for tracking with log-transformed size variables and constant acceleration
#         points = MerweScaledSigmaPoints(n=10, alpha=0.1, beta=2., kappa=0.)
#         self.ukf = UKF(dim_x=10, dim_z=4, fx=self.f_process, hx=self.h_measurement, dt=1., points=points)
#
#         # Process noise and measurement noise covariance matrices
#         # Increased process noise for velocity and acceleration components
#         q_pos = 1.0    # Position process noise
#         q_vel = 100.0   # Velocity process noise
#         q_acc = 25.0  # Acceleration process noise
#         self.ukf.Q = np.diag([
#             q_pos, q_pos,      # cx, cy
#             q_acc, q_acc,      # log_s, log_r
#             q_vel, q_vel,      # vx, vy
#             q_acc, q_acc,      # v_log_s, (additional process noise if needed)
#             q_acc, q_acc       # ax, ay
#         ])
#
#         self.ukf.R = np.eye(4) * 10    # Measurement covariance
#
#         # Initialize state with the first bounding box, using log(s) and log(r)
#         x, y, w, h = bbox
#         s = w * h
#         r = w / h if h != 0 else 1.0  # Prevent division by zero
#         log_s = np.log(s) if s > 0 else np.log(min_area)
#         log_r = np.log(r) if r > 0 else np.log(min_aspect_ratio)
#         self.ukf.x = np.array([
#             x + w / 2,          # cx
#             y + h / 2,          # cy
#             log_s,              # log_s
#             log_r,              # log_r
#             0,                  # vx
#             0,                  # vy
#             0,                  # v_log_s
#             0,                  # ax
#             0,                  # ay
#             0                   # a_log_s
#         ])  # [cx, cy, log_s, log_r, vx, vy, v_log_s, ax, ay, a_log_s]
#
#         self.id = track_id
#         self.age = 0
#         self.time_since_update = 0
#         self.max_age = max_age
#         self.hits = 0
#         self.hit_streak = 0
#         self.min_area = min_area
#         self.min_aspect_ratio = min_aspect_ratio
#
#         # Define maximum allowed velocities and accelerations to prevent extreme movements
#         self.max_velocity = 1000  # pixels per frame
#         self.max_acceleration = 5000  # pixels per frame^2
#
#     def f_process(self, x, dt):
#         """
#         Process model function for the UKF.
#         Assumes a constant acceleration model with no control input.
#         """
#         F = np.array([
#             [1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
#             [0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
#             [0, 0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2],
#             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
#             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#         ])
#         return np.dot(F, x)
#
#     def h_measurement(self, x):
#         """
#         Measurement function for the UKF.
#         Maps the state vector to the measurement space (cx, cy, log_s, log_r).
#         """
#         return np.array([x[0], x[1], x[2], x[3]])
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         self.ukf.predict()
#         self.age += 1
#         self.time_since_update += 1
#
#         # Clamp velocity and acceleration to prevent extreme predictions
#         self.clamp_state()
#
#         return self.get_state()
#
#     def update(self, bbox):
#         """
#         Updates the state vector with observed bbox.
#         """
#         x, y, w, h = bbox
#         s = w * h
#         r = w / h if h != 0 else 1.0  # Prevent division by zero
#         log_s = np.log(s) if s > 0 else self.min_area
#         log_r = np.log(r) if r > 0 else self.min_aspect_ratio
#         z = np.array([x + w / 2, y + h / 2, log_s, log_r])
#         self.ukf.update(z)
#         self.time_since_update = 0
#         self.hits += 1
#         self.hit_streak += 1
#
#         # Clamp velocity and acceleration after update
#         self.clamp_state()
#
#     def clamp_state(self):
#         """
#         Clamps the velocity and acceleration components of the state vector
#         to prevent extreme values.
#         """
#         # Extract velocity and acceleration components
#         vx, vy, v_log_s, ax, ay, a_log_s = self.ukf.x[4:10]
#
#         # Clamp velocities
#         vx = np.clip(vx, -self.max_velocity, self.max_velocity)
#         vy = np.clip(vy, -self.max_velocity, self.max_velocity)
#         v_log_s = np.clip(v_log_s, -self.max_velocity, self.max_velocity)
#
#         # Clamp accelerations
#         ax = np.clip(ax, -self.max_acceleration, self.max_acceleration)
#         ay = np.clip(ay, -self.max_acceleration, self.max_acceleration)
#         a_log_s = np.clip(a_log_s, -self.max_acceleration, self.max_acceleration)
#
#         # Update the state vector with clamped values
#         self.ukf.x[4:10] = np.array([vx, vy, v_log_s, ax, ay, a_log_s])
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         Ensures that the bounding box size does not collapse below minimum thresholds.
#         """
#         x = self.ukf.x[0]
#         y = self.ukf.x[1]
#         log_s = self.ukf.x[2]
#         log_r = self.ukf.x[3]
#
#         # Enforce minimum log_s and log_r
#         log_s = max(log_s, self.min_area)
#         log_r = max(log_r, self.min_aspect_ratio)
#
#         s = np.exp(log_s)
#         r = np.exp(log_r)
#
#         # Prevent extremely small or large sizes
#         min_w = 10   # Minimum width
#         min_h = 10   # Minimum height
#         max_w = 1000 # Maximum width
#         max_h = 1000 # Maximum height
#
#         w = np.sqrt(s * r)
#         h = s / w
#
#         # Clamp width and height to reasonable bounds
#         w = np.clip(w, min_w, max_w)
#         h = np.clip(h, min_h, max_h)
#
#         return [int(x - w / 2), int(y - h / 2), int(w), int(h)]
#
# class Sort:
#     def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age  # Frames to keep alive a track without associated detections
#         self.min_hits = min_hits  # Minimum number of associated detections before track is confirmed
#         self.iou_threshold = iou_threshold
#         self.tracks = []
#         self.next_id = 0
#
#         # Define maximum allowable distance for measurement gating
#         self.max_distance = 200  # pixels
#
#     def update(self, detections):
#         """
#         Params:
#           detections - a list of detections, each in the format:
#                        {'bbox': [x, y, w, h], 'confidence': float}
#         Requires:
#           this method must be called once for each frame even with empty detections.
#         Returns:
#           a list of tracks, each track is a dict with 'id', 'bbox', and 'age'
#         """
#         # Predict new locations of existing tracks
#         for trk in self.tracks:
#             trk.predict()
#
#         trks = [trk.get_state() for trk in self.tracks]
#
#         # Create the cost matrix (1 - IoU) with measurement gating
#         matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)
#
#         # Update matched tracks with assigned detections
#         for t, trk in enumerate(self.tracks):
#             if t not in unmatched_trks:
#                 d = matched[t]
#                 trk.update(detections[d]['bbox'])
#
#         # Create and initialize new tracks for unmatched detections
#         for i in unmatched_dets:
#             trk = Track(detections[i]['bbox'], self.next_id, self.max_age)
#             self.next_id += 1
#             self.tracks.append(trk)
#
#         # Remove dead tracks
#         self.tracks = [trk for trk in self.tracks if trk.time_since_update <= self.max_age]
#
#         # Prepare output
#         output_tracks = []
#         for trk in self.tracks:
#             if trk.hits >= self.min_hits or trk.age <= self.min_hits:
#                 bbox = trk.get_state()
#                 output_tracks.append({
#                     'id': trk.id,
#                     'bbox': bbox,
#                     'age': trk.age
#                 })
#
#         return output_tracks
#
#     def associate_detections_to_trackers(self, detections, trackers):
#         """
#         Assigns detections to tracked objects (both represented as bounding boxes).
#         Implements measurement gating to limit associations based on maximum allowable distance.
#
#         Returns:
#             matched_indices: dict mapping tracker index to detection index
#             unmatched_detections: list of detection indices not matched
#             unmatched_trackers: list of tracker indices not matched
#         """
#         if len(trackers) == 0:
#             unmatched_detections = list(range(len(detections)))
#             return {}, unmatched_detections, []
#
#         iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
#         confidence_scores = np.array([det['confidence'] for det in detections])
#
#         for d, det in enumerate(detections):
#             for t, trk in enumerate(trackers):
#                 iou = compute_iou(det['bbox'], trk)
#                 iou_matrix[d, t] = iou * confidence_scores[d]
#
#         # Apply linear assignment
#         matched_indices = linear_sum_assignment(-iou_matrix)
#         matched_indices = np.array(matched_indices).T
#
#         # Initialize matched, unmatched detections and trackers
#         matches = {}
#         unmatched_detections = []
#         unmatched_trackers = []
#
#         for d, t in matched_indices:
#             # Compute the Euclidean distance between detection and tracker centers
#             det_bbox = detections[d]['bbox']
#             trk_bbox = trackers[t]
#             det_center = np.array([det_bbox[0] + det_bbox[2] / 2, det_bbox[1] + det_bbox[3] / 2])
#             trk_center = np.array([trk_bbox[0] + trk_bbox[2] / 2, trk_bbox[1] + trk_bbox[3] / 2])
#             distance = np.linalg.norm(det_center - trk_center)
#
#             if distance > self.max_distance:
#                 # If the detection is too far from the tracker, consider it unmatched
#                 unmatched_detections.append(d)
#                 unmatched_trackers.append(t)
#             else:
#                 matches[t] = d
#
#         # Detections not matched
#         for d in range(len(detections)):
#             if d not in matched_indices[:, 0]:
#                 unmatched_detections.append(d)
#
#         # Trackers not matched
#         for t in range(len(trackers)):
#             if t not in matched_indices[:, 1]:
#                 unmatched_trackers.append(t)
#
#         return matches, unmatched_detections, unmatched_trackers
#






#
#
#
# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import MerweScaledSigmaPoints
#
# def compute_iou(bb_test, bb_gt):
#     """
#     Computes IOU between two bounding boxes.
#     """
#     # Ensure positive width and height
#     if bb_test[2] <= 0 or bb_test[3] <= 0 or bb_gt[2] <= 0 or bb_gt[3] <= 0:
#         return 0.0
#
#     xx1 = np.maximum(bb_test[0], bb_gt[0])
#     yy1 = np.maximum(bb_test[1], bb_gt[1])
#     xx2 = np.minimum(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
#     yy2 = np.minimum(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
#     w = np.maximum(0., xx2 - xx1)
#     h = np.maximum(0., yy2 - yy1)
#     wh = w * h
#     denominator = (bb_test[2] * bb_test[3]) + (bb_gt[2] * bb_gt[3]) - wh
#     o = wh / denominator if denominator > 0 else 0.0
#     return o
#
# class Track:
#     def __init__(self, bbox, track_id, max_age=30, min_log_area=np.log(10), min_log_aspect_ratio=np.log(0.5)):
#         """
#         Initializes a Track object with a bounding box.
#
#         Args:
#             bbox (list): Bounding box [x, y, w, h].
#             track_id (int): Unique identifier for the track.
#             max_age (int): Maximum number of frames to keep the track without updates.
#             min_log_area (float): Minimum logarithmic area to prevent collapse.
#             min_log_aspect_ratio (float): Minimum logarithmic aspect ratio to prevent collapse.
#         """
#         # Define UKF for tracking with log-transformed size variables and constant acceleration
#         points = MerweScaledSigmaPoints(n=10, alpha=0.1, beta=2., kappa=0.)
#         self.ukf = UKF(dim_x=10, dim_z=4, fx=self.f_process, hx=self.h_measurement, dt=1., points=points)
#
#         # Process noise and measurement noise covariance matrices
#         # Adjusted process noise for stability
#         q_pos = 1.0    # Position process noise
#         q_vel = 75.0 #100 # Velocity process noise
#         q_acc = 15.0  #25 # Acceleration process noise
#         self.ukf.Q = np.diag([
#             q_pos, q_pos,      # cx, cy
#             q_acc, q_acc,      # log_s, log_r
#             q_vel, q_vel,      # vx, vy
#             q_acc, q_acc,      # v_log_s, ax
#             q_acc, q_acc       # ay, a_log_s
#         ])
#
#         # self.ukf.R = np.eye(4) * 15 #10    # Measurement covariance
#         self.ukf.R = np.diag([10.0, 10.0, 5.0, 5.0])   # Measurement covariance
#
#         # Initialize state with the first bounding box, using log(s) and log(r)
#         x, y, w, h = bbox
#         s = w * h
#         r = w / h if h != 0 else 1.0  # Prevent division by zero
#         log_s = np.log(s) if s > 0 else min_log_area
#         log_r = np.log(r) if r > 0 else min_log_aspect_ratio
#         self.ukf.x = np.array([
#             x + w / 2,          # cx
#             y + h / 2,          # cy
#             log_s,              # log_s
#             log_r,              # log_r
#             0,                  # vx
#             0,                  # vy
#             0,                  # v_log_s
#             0,                  # ax
#             0,                  # ay
#             0                   # a_log_s
#         ])  # [cx, cy, log_s, log_r, vx, vy, v_log_s, ax, ay, a_log_s]
#
#         self.id = track_id
#         self.age = 0
#         self.time_since_update = 0
#         self.max_age = max_age
#         self.hits = 0
#         self.hit_streak = 0
#         self.min_log_area = min_log_area  # Renamed for clarity
#         self.min_log_aspect_ratio = min_log_aspect_ratio  # Renamed for clarity
#
#         # Define maximum allowed velocities and accelerations to prevent extreme movements
#         self.max_velocity = 200 # 1000  # pixels per frame
#         self.max_acceleration = 500 #0  # pixels per frame^2
#
#     def f_process(self, x, dt):
#         """
#         Process model function for the UKF.
#         Assumes a constant acceleration model with no control input.
#         """
#         F = np.array([
#             [1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
#             [0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
#             [0, 0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2],
#             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
#             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#         ])
#         return np.dot(F, x)
#
#     def h_measurement(self, x):
#         """
#         Measurement function for the UKF.
#         Maps the state vector to the measurement space (cx, cy, log_s, log_r).
#         """
#         return np.array([x[0], x[1], x[2], x[3]])
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         self.ukf.predict()
#         self.age += 1
#         self.time_since_update += 1
#
#         # Clamp velocity and acceleration to prevent extreme predictions
#         self.clamp_state()
#
#         return self.get_state()
#
#     def update(self, bbox):
#         """
#         Updates the state vector with observed bbox.
#         """
#         x, y, w, h = bbox
#         s = w * h
#         r = w / h if h != 0 else 1.0  # Prevent division by zero
#         log_s = np.log(s) if s > 0 else self.min_log_area
#         log_r = np.log(r) if r > 0 else self.min_log_aspect_ratio
#         z = np.array([x + w / 2, y + h / 2, log_s, log_r])
#         self.ukf.update(z)
#         self.time_since_update = 0
#         self.hits += 1
#         self.hit_streak += 1
#
#         # Clamp velocity and acceleration after update
#         self.clamp_state()
#
#     def clamp_state(self):
#         """
#         Clamps the velocity and acceleration components of the state vector
#         to prevent extreme values.
#         """
#         # Extract velocity and acceleration components
#         vx, vy, v_log_s, ax, ay, a_log_s = self.ukf.x[4:10]
#
#         # Clamp velocities
#         vx = np.clip(vx, -self.max_velocity, self.max_velocity)
#         vy = np.clip(vy, -self.max_velocity, self.max_velocity)
#         v_log_s = np.clip(v_log_s, -self.max_velocity, self.max_velocity)
#
#         # Clamp accelerations
#         ax = np.clip(ax, -self.max_acceleration, self.max_acceleration)
#         ay = np.clip(ay, -self.max_acceleration, self.max_acceleration)
#         a_log_s = np.clip(a_log_s, -self.max_acceleration, self.max_acceleration)
#
#         # Update the state vector with clamped values
#         self.ukf.x[4:10] = np.array([vx, vy, v_log_s, ax, ay, a_log_s])
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         Ensures that the bounding box size does not collapse below minimum thresholds.
#         """
#         x = self.ukf.x[0]
#         y = self.ukf.x[1]
#         log_s = self.ukf.x[2]
#         log_r = self.ukf.x[3]
#
#         # Enforce minimum log_s and log_r
#         log_s = max(log_s, self.min_log_area)
#         log_r = max(log_r, self.min_log_aspect_ratio)
#
#         s = np.exp(log_s)
#         r = np.exp(log_r)
#
#         # Prevent extremely small or large sizes
#         min_w = 10   # Minimum width
#         min_h = 10   # Minimum height
#         max_w = 720 # Maximum width
#         max_h = 720 # Maximum height
#
#         w = np.sqrt(s * r)
#         h = s / w
#
#         # Clamp width and height to reasonable bounds
#         w = np.clip(w, min_w, max_w)
#         h = np.clip(h, min_h, max_h)
#
#         return [int(x - w / 2), int(y - h / 2), int(w), int(h)]
#
# class Sort:
#     def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
#         """
#         Sets key parameters for SORT
#         """
#         self.max_age = max_age  # Frames to keep alive a track without associated detections
#         self.min_hits = min_hits  # Minimum number of associated detections before track is confirmed
#         self.iou_threshold = iou_threshold
#         self.tracks = []
#         self.next_id = 0
#
#         # Define maximum allowable distance for measurement gating
#         self.max_distance = 200  # pixels
#
#     def update(self, detections):
#         """
#         Params:
#           detections - a list of detections, each in the format:
#                        {'bbox': [x, y, w, h], 'confidence': float}
#         Requires:
#           this method must be called once for each frame even with empty detections.
#         Returns:
#           a list of tracks, each track is a dict with 'id', 'bbox', and 'age'
#         """
#         # Predict new locations of existing tracks
#         for trk in self.tracks:
#             trk.predict()
#
#         trks = [trk.get_state() for trk in self.tracks]
#
#         # Create the cost matrix (IoU * confidence)
#         matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)
#
#         # Update matched tracks with assigned detections
#         for t, d in matched.items():
#             self.tracks[t].update(detections[d]['bbox'])
#
#         # Create and initialize new tracks for unmatched detections
#         for i in unmatched_dets:
#             trk = Track(detections[i]['bbox'], self.next_id, self.max_age)
#             self.next_id += 1
#             self.tracks.append(trk)
#
#         # Remove dead tracks
#         self.tracks = [trk for trk in self.tracks if trk.time_since_update <= self.max_age]
#
#         # Prepare output
#         output_tracks = []
#         for trk in self.tracks:
#             # **Improvement 2:** Only output tracks that have been confirmed
#             if trk.hits >= self.min_hits:
#                 bbox = trk.get_state()
#                 output_tracks.append({
#                     'id': trk.id,
#                     'bbox': bbox,
#                     'age': trk.age
#                 })
#
#         return output_tracks
#
#     def associate_detections_to_trackers(self, detections, trackers):
#         """
#         Assigns detections to tracked objects (both represented as bounding boxes).
#         Implements measurement gating to limit associations based on maximum allowable distance.
#
#         Returns:
#             matched_indices: dict mapping tracker index to detection index
#             unmatched_detections: list of detection indices not matched
#             unmatched_trackers: list of tracker indices not matched
#         """
#         if len(trackers) == 0:
#             unmatched_detections = list(range(len(detections)))
#             return {}, unmatched_detections, []
#
#         iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
#         confidence_scores = np.array([det['confidence'] for det in detections])
#
#         for d, det in enumerate(detections):
#             for t, trk in enumerate(trackers):
#                 iou = compute_iou(det['bbox'], trk)
#                 iou_matrix[d, t] = iou * confidence_scores[d]
#
#         # Apply linear assignment
#         matched_indices = linear_sum_assignment(-iou_matrix)
#         matched_indices = np.array(matched_indices).T
#
#         # **Improvement 3:** Prevent duplication by using sets
#         matches = {}
#         unmatched_detections = set(range(len(detections)))
#         unmatched_trackers = set(range(len(trackers)))
#
#         for d, t in matched_indices:
#             # Compute the Euclidean distance between detection and tracker centers
#             det_bbox = detections[d]['bbox']
#             trk_bbox = trackers[t]
#             det_center = np.array([det_bbox[0] + det_bbox[2] / 2, det_bbox[1] + det_bbox[3] / 2])
#             trk_center = np.array([trk_bbox[0] + trk_bbox[2] / 2, trk_bbox[1] + trk_bbox[3] / 2])
#             distance = np.linalg.norm(det_center - trk_center)
#
#             if distance > self.max_distance:
#                 # If the detection is too far from the tracker, consider it unmatched
#                 continue
#             else:
#                 matches[t] = d
#                 unmatched_detections.discard(d)
#                 unmatched_trackers.discard(t)
#
#         # Convert sets to lists
#         unmatched_detections = list(unmatched_detections)
#         unmatched_trackers = list(unmatched_trackers)
#
#         return matches, unmatched_detections, unmatched_trackers






















import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from typing import List, Dict, Tuple, Optional

BoundingBox = List[float]
Detection = Dict[str, float]
TrackOutput = Dict[str, any]


def compute_iou(bb_test: BoundingBox, bb_gt: BoundingBox) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.

    Args:
        bb_test (BoundingBox): Bounding box in the format [x, y, w, h].
        bb_gt (BoundingBox): Ground truth bounding box in the format [x, y, w, h].

    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    x1, y1, w1, h1 = bb_test
    x2, y2, w2, h2 = bb_gt

    # Ensure positive width and height
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    # Calculate intersection coordinates
    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1 + w1, x2 + w2)
    yy2 = min(y1 + h1, y2 + h2)

    # Calculate intersection area
    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter_area = inter_w * inter_h

    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class Track:
    """
    Represents an individual track with its own Unscented Kalman Filter (UKF).

    Attributes:
        ukf (UKF): Unscented Kalman Filter instance for state estimation.
        id (int): Unique identifier for the track.
        age (int): Total number of frames since the track was created.
        time_since_update (int): Number of frames since the last update.
        max_age (int): Maximum allowed frames without an update before the track is deleted.
        hits (int): Total number of successful updates.
        hit_streak (int): Current number of consecutive hits.
        min_log_area (float): Minimum logarithmic area to prevent state collapse.
        min_log_aspect_ratio (float): Minimum logarithmic aspect ratio to prevent state collapse.
        max_velocity (float): Maximum allowed velocity to prevent extreme movements.
        max_acceleration (float): Maximum allowed acceleration to prevent extreme changes.
    """

    def __init__(
        self,
        bbox: BoundingBox,
        track_id: int,
        max_age: int = 30,
        min_log_area: float = np.log(10.0),
        min_log_aspect_ratio: float = np.log(0.5)
    ) -> None:
        """
        Initializes a Track object with a bounding box.

        Args:
            bbox (BoundingBox): Initial bounding box [x, y, w, h].
            track_id (int): Unique identifier for the track.
            max_age (int, optional): Maximum number of frames to keep the track without updates.
                                      Defaults to 30.
            min_log_area (float, optional): Minimum logarithmic area to prevent collapse.
                                             Defaults to np.log(10.0).
            min_log_aspect_ratio (float, optional): Minimum logarithmic aspect ratio to prevent collapse.
                                                    Defaults to np.log(0.5).
        """
        # Define UKF for tracking with log-transformed size variables and constant acceleration
        points = MerweScaledSigmaPoints(n=10, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UKF(
            dim_x=10,
            dim_z=4,
            fx=self.f_process,
            hx=self.h_measurement,
            dt=1.0,
            points=points
        )

        # Process noise covariance matrix
        q_pos = 5.0    # Position process noise
        q_vel = 100.0   # Velocity process noise
        q_acc = 5.0   # Acceleration process noise
        self.ukf.Q = np.diag([
            q_pos, q_pos,      # cx, cy
            q_acc, q_acc,      # log_s, log_r
            q_vel, q_vel,      # vx, vy
            q_acc, q_acc,      # v_log_s, ax
            q_acc, q_acc       # ay, a_log_s
        ])

        # Measurement noise covariance matrix
        # self.ukf.R = np.diag([10.0, 10.0, 15.0, 15.0])  # [cx, cy, log_s, log_r]
        self.ukf.R = np.eye(4) * 10
        # Initialize state with the first bounding box, using log(s) and log(r)
        x, y, w, h = bbox
        s = w * h
        r = w / h if h != 0 else 1.0  # Prevent division by zero
        log_s = np.log(s) if s > 0 else min_log_area
        log_r = np.log(r) if r > 0 else min_log_aspect_ratio

        self.ukf.x = np.array([
            x + w / 2,          # cx
            y + h / 2,          # cy
            log_s,              # log_s
            log_r,              # log_r
            0.0,                # vx
            0.0,                # vy
            0.0,                # v_log_s
            0.0,                # ax
            0.0,                # ay
            0.0                 # a_log_s
        ])  # [cx, cy, log_s, log_r, vx, vy, v_log_s, ax, ay, a_log_s]

        # Track identifiers and status
        self.id: int = track_id
        self.age: int = 0
        self.time_since_update: int = 0
        self.max_age: int = max_age
        self.hits: int = 0
        self.hit_streak: int = 0

        # Minimum thresholds to prevent state collapse
        self.min_log_area: float = min_log_area
        self.min_log_aspect_ratio: float = min_log_aspect_ratio

        # Constraints to prevent extreme movements
        self.max_velocity: float = 200.0  # pixels per frame
        self.max_acceleration: float = 500.0  # pixels per frame^2

    def f_process(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Process model function for the UKF.
        Assumes a constant acceleration model with no control input.

        Args:
            x (np.ndarray): Current state vector.
            dt (float): Time step.

        Returns:
            np.ndarray: Predicted state vector after time step dt.
        """
        F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
            [0, 0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        return F @ x

    def h_measurement(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function for the UKF.
        Maps the state vector to the measurement space (cx, cy, log_s, log_r).

        Args:
            x (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Measurement vector.
        """
        return np.array([x[0], x[1], x[2], x[3]])

    def predict(self) -> BoundingBox:
        """
        Advances the state vector and returns the predicted bounding box estimate.

        Returns:
            BoundingBox: Predicted bounding box [x, y, w, h].
        """
        self.ukf.predict()
        self.age += 1
        self.time_since_update += 1

        # Clamp velocity and acceleration to prevent extreme predictions
        self.clamp_state()

        return self.get_state()

    def update(self, bbox: BoundingBox) -> None:
        """
        Updates the state vector with the observed bounding box.

        Args:
            bbox (BoundingBox): Observed bounding box [x, y, w, h].
        """
        x, y, w, h = bbox
        s = w * h
        r = w / h if h != 0 else 1.0  # Prevent division by zero
        log_s = np.log(s) if s > 0 else self.min_log_area
        log_r = np.log(r) if r > 0 else self.min_log_aspect_ratio
        z = np.array([x + w / 2, y + h / 2, log_s, log_r])
        self.ukf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Clamp velocity and acceleration after update
        self.clamp_state()

    def clamp_state(self) -> None:
        """
        Clamps the velocity and acceleration components of the state vector
        to prevent extreme values.
        """
        # Extract velocity and acceleration components
        vx, vy, v_log_s, ax, ay, a_log_s = self.ukf.x[4:10]

        # Clamp velocities
        vx = np.clip(vx, -self.max_velocity, self.max_velocity)
        vy = np.clip(vy, -self.max_velocity, self.max_velocity)
        v_log_s = np.clip(v_log_s, -self.max_velocity, self.max_velocity)

        # Clamp accelerations
        ax = np.clip(ax, -self.max_acceleration, self.max_acceleration)
        ay = np.clip(ay, -self.max_acceleration, self.max_acceleration)
        a_log_s = np.clip(a_log_s, -self.max_acceleration, self.max_acceleration)

        # Update the state vector with clamped values
        self.ukf.x[4:10] = np.array([vx, vy, v_log_s, ax, ay, a_log_s])

    def get_state(self) -> BoundingBox:
        """
        Returns the current bounding box estimate.
        Ensures that the bounding box size does not collapse below minimum thresholds.

        Returns:
            BoundingBox: Estimated bounding box [x, y, w, h].
        """
        cx, cy, log_s, log_r = self.ukf.x[:4]

        # Enforce minimum log_s and log_r
        log_s = max(log_s, self.min_log_area)
        log_r = max(log_r, self.min_log_aspect_ratio)

        s = np.exp(log_s)
        r = np.exp(log_r)

        # Calculate width and height from area and aspect ratio
        w = np.sqrt(s * r)
        h = s / w

        # Prevent extremely small or large sizes
        min_w, min_h = 10.0, 10.0    # Minimum width and height
        max_w, max_h = 720.0, 720.0  # Maximum width and height

        w = np.clip(w, min_w, max_w)
        h = np.clip(h, min_h, max_h)

        # Convert center coordinates back to top-left corner
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        w = int(w)
        h = int(h)

        return [x, y, w, h]


class Sort:
    """
    SORT (Simple Online and Realtime Tracking) Tracker.

    Attributes:
        max_age (int): Maximum number of frames to keep a track without updates.
        min_hits (int): Minimum number of associated detections before a track is confirmed.
        iou_threshold (float): IoU threshold for data association.
        tracks (List[Track]): List of active tracks.
        next_id (int): Next unique identifier for a new track.
        max_distance (float): Maximum allowable Euclidean distance for measurement gating.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 200.0
    ) -> None:
        """
        Initializes the SORT tracker with specified parameters.

        Args:
            max_age (int, optional): Maximum number of frames to keep a track without updates.
                                      Defaults to 30.
            min_hits (int, optional): Minimum number of associated detections before a track is confirmed.
                                      Defaults to 3.
            iou_threshold (float, optional): IoU threshold for data association.
                                             Defaults to 0.3.
            max_distance (float, optional): Maximum allowable Euclidean distance for measurement gating.
                                            Defaults to 200.0.
        """
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.iou_threshold: float = iou_threshold
        self.max_distance: float = max_distance

        self.tracks: List[Track] = []
        self.next_id: int = 0

    def update(self, detections: List[Detection]) -> List[TrackOutput]:
        """
        Updates the tracker with the current frame's detections.

        Args:
            detections (List[Detection]): List of detections, each as a dictionary with keys:
                                          'bbox' (BoundingBox) and 'confidence' (float).

        Returns:
            List[TrackOutput]: List of confirmed tracks with 'id', 'bbox', and 'age'.
        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()

        # Get current state estimates for all tracks
        current_track_bboxes = [track.get_state() for track in self.tracks]

        # Associate detections to existing tracks
        matches, unmatched_detections, unmatched_trackers = self.associate_detections_to_trackers(
            detections, current_track_bboxes
        )

        # Update matched tracks with assigned detections
        for track_idx, det_idx in matches.items():
            self.tracks[track_idx].update(detections[det_idx]['bbox'])

        # Initialize new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = Track(
                bbox=detections[det_idx]['bbox'],
                track_id=self.next_id,
                max_age=self.max_age
            )
            self.tracks.append(new_track)
            self.next_id += 1

        # Remove dead tracks that have exceeded max_age without updates
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_age
        ]

        # Compile list of confirmed tracks for output
        confirmed_tracks: List[TrackOutput] = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                confirmed_tracks.append({
                    'id': track.id,
                    'bbox': track.get_state(),
                    'age': track.age
                })

        return confirmed_tracks

    def associate_detections_to_trackers(
        self,
        detections: List[Detection],
        trackers: List[BoundingBox]
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Assigns detections to tracked objects using a combined cost metric based on IoU and Euclidean distance.

        Args:
            detections (List[Detection]): List of detections with 'bbox' and 'confidence'.
            trackers (List[BoundingBox]): List of current tracked bounding boxes.

        Returns:
            Tuple[Dict[int, int], List[int], List[int]]:
                - matched_indices: Dictionary mapping tracker indices to detection indices.
                - unmatched_detections: List of detection indices that did not match any tracker.
                - unmatched_trackers: List of tracker indices that did not match any detection.
        """
        num_detections = len(detections)
        num_trackers = len(trackers)

        if num_trackers == 0:
            return {}, list(range(num_detections)), []

        # Initialize cost matrix
        cost_matrix = np.zeros((num_detections, num_trackers), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                # Compute IoU
                iou = compute_iou(det['bbox'], trk)

                # Compute Euclidean distance between centers
                det_center = np.array([
                    det['bbox'][0] + det['bbox'][2] / 2,
                    det['bbox'][1] + det['bbox'][3] / 2
                ])
                trk_center = np.array([
                    trk[0] + trk[2] / 2,
                    trk[1] + trk[3] / 2
                ])
                distance = np.linalg.norm(det_center - trk_center)

                # Combine IoU and distance into a single cost metric
                # Lower cost for higher IoU and lower distance
                # Adjust weighting factors as necessary
                cost_matrix[d, t] = (1.0 - iou) + (distance / self.max_distance)

        # Perform linear assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_indices: Dict[int, int] = {}
        unmatched_detections: set = set(range(num_detections))
        unmatched_trackers: set = set(range(num_trackers))

        for d, t in zip(row_indices, col_indices):
            if cost_matrix[d, t] > (1.0 - self.iou_threshold) + 1.0:
                # Cost too high; do not match
                continue
            matched_indices[t] = d
            unmatched_detections.discard(d)
            unmatched_trackers.discard(t)

        return matched_indices, list(unmatched_detections), list(unmatched_trackers)


def visualize_tracks(frame: np.ndarray, tracks: List[TrackOutput]) -> np.ndarray:
    """
    Draws bounding boxes and track IDs on the frame.

    Args:
        frame (np.ndarray): The video frame in which to draw.
        tracks (List[TrackOutput]): List of tracks with 'id' and 'bbox'.

    Returns:
        np.ndarray: The frame with drawn bounding boxes and IDs.
    """
    for track in tracks:
        bbox = track['bbox']
        track_id = track['id']
        x, y, w, h = bbox
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Put track ID text
        cv2.putText(
            frame,
            f'ID: {track_id}',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    return frame





# # sort_UKF.py
#
# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import MerweScaledSigmaPoints
# from typing import List, Dict, Tuple, Optional
#
# import logging
#
# BoundingBox = List[float]
# Detection = Dict[str, float]
# TrackOutput = Dict[str, any]
#
#
# def compute_iou(bb_test: BoundingBox, bb_gt: BoundingBox) -> float:
#     """
#     Computes Intersection over Union (IoU) between two bounding boxes.
#
#     Args:
#         bb_test (BoundingBox): Bounding box in the format [x, y, w, h].
#         bb_gt (BoundingBox): Ground truth bounding box in the format [x, y, w, h].
#
#     Returns:
#         float: IoU value between 0.0 and 1.0.
#     """
#     x1, y1, w1, h1 = bb_test
#     x2, y2, w2, h2 = bb_gt
#
#     # Ensure positive width and height
#     if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
#         return 0.0
#
#     # Calculate intersection coordinates
#     xx1 = max(x1, x2)
#     yy1 = max(y1, y2)
#     xx2 = min(x1 + w1, x2 + w2)
#     yy2 = min(y1 + h1, y2 + h2)
#
#     # Calculate intersection area
#     inter_w = max(0.0, xx2 - xx1)
#     inter_h = max(0.0, yy2 - yy1)
#     inter_area = inter_w * inter_h
#
#     # Calculate union area
#     area1 = w1 * h1
#     area2 = w2 * h2
#     union_area = area1 + area2 - inter_area
#
#     return inter_area / union_area if union_area > 0 else 0.0
#
#
# class Track:
#     """
#     Represents an individual track with its own Unscented Kalman Filter (UKF).
#
#     Attributes:
#         ukf (UKF): Unscented Kalman Filter instance for state estimation.
#         id (int): Unique identifier for the track.
#         age (int): Total number of frames since the track was created.
#         time_since_update (int): Number of frames since the last update.
#         max_age (int): Maximum allowed frames without an update before the track is deleted.
#         hits (int): Total number of successful updates.
#         hit_streak (int): Current number of consecutive hits.
#     """
#
#     def __init__(
#         self,
#         bbox: BoundingBox,
#         track_id: int,
#         max_age: int = 30,
#         min_log_area: float = np.log(10.0),
#         min_log_aspect_ratio: float = np.log(0.5)
#     ) -> None:
#         """
#         Initializes a Track object with a bounding box.
#
#         Args:
#             bbox (BoundingBox): Initial bounding box [x, y, w, h].
#             track_id (int): Unique identifier for the track.
#             max_age (int, optional): Maximum number of frames to keep the track without updates.
#                                       Defaults to 30.
#             min_log_area (float, optional): Minimum logarithmic area to prevent collapse.
#                                              Defaults to np.log(10.0).
#             min_log_aspect_ratio (float, optional): Minimum logarithmic aspect ratio to prevent collapse.
#                                                     Defaults to np.log(0.5).
#         """
#         # Define UKF for tracking with log-transformed size variables and constant acceleration
#         points = MerweScaledSigmaPoints(n=10, alpha=0.1, beta=2.0, kappa=0.0)
#         self.ukf = UKF(
#             dim_x=10,
#             dim_z=4,
#             fx=self.f_process,
#             hx=self.h_measurement,
#             dt=1.0,
#             points=points
#         )
#
#         # Process noise covariance matrix
#         q_pos = 5.0    # Position process noise
#         q_vel = 100.0  # Velocity process noise
#         q_acc = 5.0    # Acceleration process noise
#         self.ukf.Q = np.diag([
#             q_pos, q_pos,       # cx, cy
#             q_acc, q_acc,       # log_s, log_r
#             q_vel, q_vel,       # vx, vy
#             q_acc, q_acc,       # v_log_s, ax
#             q_acc, q_acc        # ay, a_log_s
#         ])
#
#         # Measurement noise covariance matrix
#         self.ukf.R = np.eye(4) * 10  # [cx, cy, log_s, log_r]
#
#         # Initialize state with the first bounding box, using log(s) and log(r)
#         x, y, w, h = bbox
#         s = w * h
#         r = w / h if h != 0 else 1.0  # Prevent division by zero
#         log_s = np.log(s) if s > 0 else min_log_area
#         log_r = np.log(r) if r > 0 else min_log_aspect_ratio
#
#         self.ukf.x = np.array([
#             x + w / 2,          # cx
#             y + h / 2,          # cy
#             log_s,              # log_s
#             log_r,              # log_r
#             0.0,                # vx
#             0.0,                # vy
#             0.0,                # v_log_s
#             0.0,                # ax
#             0.0,                # ay
#             0.0                 # a_log_s
#         ])  # [cx, cy, log_s, log_r, vx, vy, v_log_s, ax, ay, a_log_s]
#
#         # Track identifiers and status
#         self.id: int = track_id
#         self.age: int = 0
#         self.time_since_update: int = 0
#         self.max_age: int = max_age
#         self.hits: int = 0
#         self.hit_streak: int = 0
#
#         # Minimum thresholds to prevent state collapse
#         self.min_log_area: float = min_log_area
#         self.min_log_aspect_ratio: float = min_log_aspect_ratio
#
#         # Constraints to prevent extreme movements
#         self.max_velocity: float = 200.0  # pixels per frame
#         self.max_acceleration: float = 500.0  # pixels per frame^2
#
#     def f_process(self, x: np.ndarray, dt: float) -> np.ndarray:
#         """
#         Process model function for the UKF.
#         Assumes a constant acceleration model with no control input.
#
#         Args:
#             x (np.ndarray): Current state vector.
#             dt (float): Time step.
#
#         Returns:
#             np.ndarray: Predicted state vector after time step dt.
#         """
#         F = np.array([
#             [1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],
#             [0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],
#             [0, 0, 1, 0, 0, 0, dt, 0, 0, 0.5 * dt**2],
#             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
#             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#         ])
#         return F @ x
#
#     def h_measurement(self, x: np.ndarray) -> np.ndarray:
#         """
#         Measurement function for the UKF.
#         Maps the state vector to the measurement space (cx, cy, log_s, log_r).
#
#         Args:
#             x (np.ndarray): Current state vector.
#
#         Returns:
#             np.ndarray: Measurement vector.
#         """
#         return np.array([x[0], x[1], x[2], x[3]])
#
#     def predict(self) -> BoundingBox:
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#
#         Returns:
#             BoundingBox: Predicted bounding box [x, y, w, h].
#         """
#         self.ukf.predict()
#         self.age += 1
#         self.time_since_update += 1
#
#         # Clamp velocity and acceleration to prevent extreme predictions
#         self.clamp_state()
#
#         return self.get_state()
#
#     def update(self, bbox: BoundingBox) -> None:
#         """
#         Updates the state vector with the observed bounding box.
#
#         Args:
#             bbox (BoundingBox): Observed bounding box [x, y, w, h].
#         """
#         x, y, w, h = bbox
#         s = w * h
#         r = w / h if h != 0 else 1.0  # Prevent division by zero
#         log_s = np.log(s) if s > 0 else self.min_log_area
#         log_r = np.log(r) if r > 0 else self.min_log_aspect_ratio
#         z = np.array([x + w / 2, y + h / 2, log_s, log_r])
#         self.ukf.update(z)
#         self.time_since_update = 0
#         self.hits += 1
#         self.hit_streak += 1
#
#         # Clamp velocity and acceleration after update
#         self.clamp_state()
#
#     def clamp_state(self) -> None:
#         """
#         Clamps the velocity and acceleration components of the state vector
#         to prevent extreme values.
#         """
#         # Extract velocity and acceleration components
#         vx, vy, v_log_s, ax, ay, a_log_s = self.ukf.x[4:10]
#
#         # Clamp velocities
#         vx = np.clip(vx, -self.max_velocity, self.max_velocity)
#         vy = np.clip(vy, -self.max_velocity, self.max_velocity)
#         v_log_s = np.clip(v_log_s, -self.max_velocity, self.max_velocity)
#
#         # Clamp accelerations
#         ax = np.clip(ax, -self.max_acceleration, self.max_acceleration)
#         ay = np.clip(ay, -self.max_acceleration, self.max_acceleration)
#         a_log_s = np.clip(a_log_s, -self.max_acceleration, self.max_acceleration)
#
#         # Update the state vector with clamped values
#         self.ukf.x[4:10] = np.array([vx, vy, v_log_s, ax, ay, a_log_s])
#
#     def get_state(self) -> BoundingBox:
#         """
#         Returns the current bounding box estimate.
#         Ensures that the bounding box size does not collapse below minimum thresholds.
#
#         Returns:
#             BoundingBox: Estimated bounding box [x, y, w, h].
#         """
#         cx, cy, log_s, log_r = self.ukf.x[:4]
#
#         # Enforce minimum log_s and log_r
#         log_s = max(log_s, self.min_log_area)
#         log_r = max(log_r, self.min_log_aspect_ratio)
#
#         s = np.exp(log_s)
#         r = np.exp(log_r)
#
#         # Calculate width and height from area and aspect ratio
#         w = np.sqrt(s * r)
#         h = s / w
#
#         # Prevent extremely small or large sizes
#         min_w, min_h = 10.0, 10.0    # Minimum width and height
#         max_w, max_h = 720.0, 720.0  # Maximum width and height
#
#         w = np.clip(w, min_w, max_w)
#         h = np.clip(h, min_h, max_h)
#
#         # Convert center coordinates back to top-left corner
#         x = int(cx - w / 2)
#         y = int(cy - h / 2)
#         w = int(w)
#         h = int(h)
#
#         return [x, y, w, h]
#
#
# class Sort:
#     """
#     SORT (Simple Online and Realtime Tracking) Tracker.
#
#     Attributes:
#         max_age (int): Maximum number of frames to keep a track without updates.
#         min_hits (int): Minimum number of associated detections before a track is confirmed.
#         iou_threshold (float): IoU threshold for data association.
#         tracks (List[Track]): List of active tracks.
#         next_id (int): Next unique identifier for a new track.
#         max_distance (float): Maximum allowable Euclidean distance for measurement gating.
#         track_id_to_track (Dict[int, Track]): Mapping from track ID to Track object.
#     """
#
#     def __init__(
#         self,
#         max_age: int = 30,
#         min_hits: int = 3,
#         iou_threshold: float = 0.3,
#         max_distance: float = 200.0
#     ) -> None:
#         """
#         Initializes the SORT tracker with specified parameters.
#
#         Args:
#             max_age (int, optional): Maximum number of frames to keep a track without updates.
#                                       Defaults to 30.
#             min_hits (int, optional): Minimum number of associated detections before a track is confirmed.
#                                       Defaults to 3.
#             iou_threshold (float, optional): IoU threshold for data association.
#                                              Defaults to 0.3.
#             max_distance (float, optional): Maximum allowable Euclidean distance for measurement gating.
#                                             Defaults to 200.0.
#         """
#         self.max_age: int = max_age
#         self.min_hits: int = min_hits
#         self.iou_threshold: float = iou_threshold
#         self.max_distance: float = max_distance
#
#         self.tracks: List[Track] = []
#         self.next_id: int = 0
#
#         # Mapping from track ID to Track object for easy access
#         self.track_id_to_track: Dict[int, Track] = {}
#
#     def update(self, detections: List[Detection]) -> List[TrackOutput]:
#         """
#         Updates the tracker with the current frame's detections.
#
#         Args:
#             detections (List[Detection]): List of detections, each as a dictionary with keys:
#                                           'bbox' (BoundingBox) and 'confidence' (float).
#
#         Returns:
#             List[TrackOutput]: List of confirmed tracks with 'id', 'bbox', and 'age'.
#         """
#         # Predict new locations of existing tracks
#         for track in self.tracks:
#             track.predict()
#
#         # Get current state estimates for all tracks
#         current_track_bboxes = [track.get_state() for track in self.tracks]
#
#         # Associate detections to existing tracks
#         matches, unmatched_detections, unmatched_trackers = self.associate_detections_to_trackers(
#             detections, current_track_bboxes
#         )
#
#         # Update matched tracks with assigned detections
#         for tracker_idx, det_idx in matches.items():
#             self.tracks[tracker_idx].update(detections[det_idx]['bbox'])
#
#         # Initialize new tracks for unmatched detections
#         for det_idx in unmatched_detections:
#             new_track = Track(
#                 bbox=detections[det_idx]['bbox'],
#                 track_id=self.next_id,
#                 max_age=self.max_age
#             )
#             self.tracks.append(new_track)
#             self.track_id_to_track[self.next_id] = new_track
#             self.next_id += 1
#
#         # Remove dead tracks that have exceeded max_age without updates
#         removed_tracks = []
#         for track in self.tracks:
#             if track.time_since_update > self.max_age:
#                 removed_tracks.append(track)
#                 del self.track_id_to_track[track.id]
#
#         self.tracks = [
#             track for track in self.tracks
#             if track.time_since_update <= self.max_age
#         ]
#
#         # Compile list of confirmed tracks for output
#         confirmed_tracks: List[TrackOutput] = []
#         for track in self.tracks:
#             if track.hits >= self.min_hits:
#                 confirmed_tracks.append({
#                     'id': track.id,
#                     'bbox': track.get_state(),
#                     'age': track.age
#                 })
#
#         return confirmed_tracks
#
#     def associate_detections_to_trackers(
#         self,
#         detections: List[Detection],
#         trackers: List[BoundingBox]
#     ) -> Tuple[Dict[int, int], List[int], List[int]]:
#         """
#         Assigns detections to tracked objects using a combined cost metric based on IoU and Euclidean distance.
#
#         Args:
#             detections (List[Detection]): List of detections with 'bbox' and 'confidence'.
#             trackers (List[BoundingBox]): List of current tracked bounding boxes.
#
#         Returns:
#             Tuple[Dict[int, int], List[int], List[int]]:
#                 - matched_indices: Dictionary mapping tracker indices to detection indices.
#                 - unmatched_detections: List of detection indices that did not match any tracker.
#                 - unmatched_trackers: List of tracker indices that did not match any detection.
#         """
#         num_detections = len(detections)
#         num_trackers = len(trackers)
#
#         if num_trackers == 0:
#             return {}, list(range(num_detections)), []
#
#         # Initialize cost matrix
#         cost_matrix = np.zeros((num_detections, num_trackers), dtype=np.float32)
#
#         for d, det in enumerate(detections):
#             for t, trk in enumerate(trackers):
#                 # Compute IoU
#                 iou = compute_iou(det['bbox'], trk)
#
#                 # Compute Euclidean distance between centers
#                 det_center = np.array([
#                     det['bbox'][0] + det['bbox'][2] / 2,
#                     det['bbox'][1] + det['bbox'][3] / 2
#                 ])
#                 trk_center = np.array([
#                     trk[0] + trk[2] / 2,
#                     trk[1] + trk[3] / 2
#                 ])
#                 distance = np.linalg.norm(det_center - trk_center)
#
#                 # Combine IoU and distance into a single cost metric
#                 # Lower cost for higher IoU and lower distance
#                 # Adjust weighting factors as necessary
#                 cost_matrix[d, t] = (1.0 - iou) + (distance / self.max_distance)
#
#         # Perform linear assignment
#         row_indices, col_indices = linear_sum_assignment(cost_matrix)
#
#         matched_indices: Dict[int, int] = {}
#         unmatched_detections: set = set(range(num_detections))
#         unmatched_trackers: set = set(range(num_trackers))
#
#         for d, t in zip(row_indices, col_indices):
#             if cost_matrix[d, t] > (1.0 - self.iou_threshold) + 1.0:
#                 # Cost too high; do not match
#                 continue
#             matched_indices[t] = d
#             unmatched_detections.discard(d)
#             unmatched_trackers.discard(t)
#
#         return matched_indices, list(unmatched_detections), list(unmatched_trackers)
#
#     def get_bbox(self, track_id: int) -> Optional[BoundingBox]:
#         """
#         Retrieve the bounding box for a given track ID.
#
#         Args:
#             track_id (int): The track ID.
#
#         Returns:
#             Optional[BoundingBox]: Bounding box coordinates [x, y, w, h] if found, else None.
#         """
#         track = self.track_id_to_track.get(track_id, None)
#         if track:
#             return track.get_state()
#         else:
#             return None
#
#



































































































































































# usage example:
#
# import cv2
# import time
# import logging
# import numpy as np
# from yunet_face_detector import detect_faces
# from sort_UKF import Sort
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# def non_max_suppression(detections, iou_threshold=0.3):
#     """
#     Performs Non-Max Suppression (NMS) on the detections.
#
#     Args:
#         detections (list): List of detections, each a dict with 'bbox' and 'confidence'.
#         iou_threshold (float): IOU threshold for suppression.
#
#     Returns:
#         list: Detections after NMS.
#     """
#     if len(detections) == 0:
#         return []
#
#     # Extract bounding boxes and confidence scores
#     bboxes = np.array([det['bbox'] for det in detections])
#     scores = np.array([det['confidence'] for det in detections])
#
#     x1 = bboxes[:, 0]
#     y1 = bboxes[:, 1]
#     x2 = bboxes[:, 0] + bboxes[:, 2]
#     y2 = bboxes[:, 1] + bboxes[:, 3]
#
#     areas = (x2 - x1) * (y2 - y1)
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#
#         # Compute IoU of the kept box with the rest
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#         # Keep boxes with IoU less than the threshold
#         inds = np.where(ovr <= iou_threshold)[0]
#         order = order[inds + 1]
#
#     # Return the filtered detections
#     return [detections[idx] for idx in keep]
#
#
# def main():
#     # Parameters
#     DETECTION_INTERVAL = 5  # Perform face detection every 5 frames
#     verbose = True  # Set verbose mode (calculate and display FPS)
#     # video_source = "4_25fps.mp4" #0  # Change to video file path if needed, e.g., "video.mp4"
#     video_source = 0 #"4_25fps.mp4" #0  # Change to video file path if needed, e.g., "video.mp4"
#
#     # Initialize the SORT tracker with adjusted parameters
#     face_tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)  # Adjusted parameters
#
#     # Open video capture
#     cap = cv2.VideoCapture(video_source)
#
#     if not cap.isOpened():
#         logger.error("Error opening video stream or file")
#         return
#
#     frame_index = 0
#     prev_time = time.time()
#     fps = 0
#
#     # Initialize a set to keep track of current face IDs
#     current_ids = set()
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.info("End of video stream")
#             break
#
#         start_time = time.time()
#
#         # Decide whether to perform face detection
#         if frame_index % DETECTION_INTERVAL == 0:
#             # Perform face detection
#             detections = detect_faces(frame)
#
#             # Apply Non-Max Suppression
#             detections = non_max_suppression(detections, iou_threshold=0.3)
#
#             # Convert detections to the expected format
#             formatted_detections = []
#             for det in detections:
#                 formatted_detections.append({
#                     'bbox': det['bbox'],
#                     'confidence': det['confidence']
#                 })
#             # Update the tracker with current frame detections
#             tracks = face_tracker.update(formatted_detections)
#         else:
#             # No detections, just update trackers (they will predict their new positions)
#             tracks = face_tracker.update([])
#
#         # Keep track of previous IDs
#         previous_ids = current_ids.copy()
#         current_ids = set([trk['id'] for trk in tracks])
#
#         # Identify new faces added
#         new_ids = current_ids - previous_ids
#         if new_ids:
#             logger.info(f"Added {len(new_ids)} new face(s), now tracking {len(current_ids)} face(s).")
#
#         # Identify faces lost
#         lost_ids = previous_ids - current_ids
#         if lost_ids:
#             logger.info(f"Lost {len(lost_ids)} face(s), now tracking {len(current_ids)} face(s).")
#
#         # Visualize the tracks
#         for trk in tracks:
#             bbox = trk['bbox']
#             x, y, w, h = bbox
#             track_id = trk['id']
#             age = trk['age']
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f'ID: {track_id}', (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             # Perform face recognition for new faces
#             if age == 1:
#                 face_img = frame[y:y + h, x:x + w]
#                 # TODO: Add your face recognition code here
#                 logger.info(f"Processing face recognition for new face ID: {track_id}")
#
#         # Calculate and display FPS if verbose mode is True
#         if verbose:
#             current_time = time.time()
#             time_diff = current_time - prev_time
#             if time_diff > 0:
#                 instantaneous_fps = 1 / time_diff
#                 fps = 0.9 * fps + 0.1 * instantaneous_fps
#                 prev_time = current_time
#                 cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#         # Display the frame with tracking
#         cv2.imshow('Face Tracking', frame)
#
#         # Handle key events
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q') or key == 27:
#             logger.info("Exit requested by user")
#             break  # Exit on 'q' key or ESC
#
#         frame_index += 1
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()
