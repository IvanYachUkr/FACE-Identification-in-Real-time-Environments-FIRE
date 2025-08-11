# --------------------------------------------
# File: sort_UKF.py
# --------------------------------------------


import cv2
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
        points = MerweScaledSigmaPoints(n=10, alpha=1e-3, beta=2.0, kappa=0.0)
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
        q_vel = 70.0   # Velocity process noise
        q_acc = 3.0   # Acceleration process noise
        self.ukf.Q = np.diag([
            q_pos, q_pos,      # cx, cy
            q_acc, q_acc,      # log_s, log_r
            q_vel, q_vel,      # vx, vy
            q_acc, q_acc,      # v_log_s, ax
            q_acc, q_acc       # ay, a_log_s
        ])

        # Measurement noise covariance matrix
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
        self.max_velocity: float = 30.0  # pixels per frame
        self.max_acceleration: float = 20.0  # pixels per frame^2

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
