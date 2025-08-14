# --------------------------------------------
# File: modules/face_recognition.py
# --------------------------------------------
import logging
import tempfile
import time
import uuid
import cv2
import numpy as np
import os
from screeninfo import get_monitors

from .detector import initialize_detector
from .encoder import Encoder
from .encryption import Encryptor
from .database import DatabaseManager
from .hnsw_manager import HNSWManager
from .tracker import initialize_tracker

def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

class FaceRecognition:
    def __init__(self,
                 detector_type: str = 'yunet',
                 align: bool = True,
                 encoder_model_type: str = '128',
                 encoder_mode: str = 'gpu_optimized',
                 similarity_threshold: float = 0.5,
                 unknown_similarity_threshold: float = 0.6,
                 unknown_trigger_count: int = 3,
                 enable_logging: bool = True,
                 show: bool = False,
                 detection_interval: int = 3,
                 hnsw_index_path: str = None,
                 hnsw_labels_path: str = None,
                 hnsw_db_ids_path: str = None,
                 hnsw_ef_construction: int = 200,
                 hnsw_m: int = 16,
                 max_recent: int = 200,
                 max_new: int = 250,
                 sqlite_db_path: str = None,
                 sqlite_db_encrypted_path: str = None,
                 encryption_password: str = None,
                 interested_label: str = None):
        """
        Note: We have added a structured storage system:
        storage/
            <detector_type>_<encoder_model_type>_<encryption_status>/
                db/
                    face_embeddings.db or face_embeddings.db.enc
                hnsw/
                    hnsw_index.bin (or encrypted)
                    hnsw_labels.pkl (or encrypted)
                    hnsw_db_ids.pkl (or encrypted)
        """
        self.encoder_model_type = encoder_model_type
        self.detector_type = detector_type.lower()
        self.align = align
        self.similarity_threshold = similarity_threshold
        self.unknown_similarity_threshold = unknown_similarity_threshold
        self.unknown_trigger_count = unknown_trigger_count
        self.enable_logging = enable_logging
        self.show = show
        self.detection_interval = detection_interval
        self.frame_index = 0
        self.interested_label = interested_label

        # Logging setup
        if self.enable_logging:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

        # Encryption
        self.encryption_password = encryption_password
        self.encryptor = None
        encryption_status = "encrypted" if self.encryption_password else "unencrypted"
        if self.encryption_password:
            self.encryptor = Encryptor(self.encryption_password)
            logging.info("Encryption is enabled for file operations.")
        else:
            logging.info("Encryption is disabled.")

        # Create the base storage directory if it doesn't exist
        base_storage_dir = "storage"
        os.makedirs(base_storage_dir, exist_ok=True)

        # Create structured directory: storage/<detector>_<encoder>_<encryption_status>
        pipeline_dir_name = f"{self.detector_type}_{self.encoder_model_type}_{encryption_status}"
        pipeline_dir = os.path.join(base_storage_dir, pipeline_dir_name)
        os.makedirs(pipeline_dir, exist_ok=True)

        # Create db and hnsw directories inside the pipeline_dir
        db_dir = os.path.join(pipeline_dir, "db")
        hnsw_dir = os.path.join(pipeline_dir, "hnsw")
        os.makedirs(db_dir, exist_ok=True)
        os.makedirs(hnsw_dir, exist_ok=True)

        # Initialize detector
        self.detector = initialize_detector(self.detector_type)

        # Initialize encoder
        self.encoder = Encoder(encoder_model_type, encoder_mode)
        self.embedding_dim = self.encoder.output_shape

        # Determine default file paths if not provided
        if hnsw_index_path is None:
            hnsw_index_path = os.path.join(hnsw_dir, f"hnsw_index_{self.detector_type}_{self.encoder_model_type}.bin")
        if hnsw_labels_path is None:
            hnsw_labels_path = os.path.join(hnsw_dir, f"hnsw_labels_{self.detector_type}_{self.encoder_model_type}.pkl")
        if hnsw_db_ids_path is None:
            hnsw_db_ids_path = os.path.join(hnsw_dir, f"hnsw_db_ids_{self.detector_type}_{self.encoder_model_type}.pkl")

        if self.encryption_password:
            # Encrypted DB path
            if sqlite_db_encrypted_path is None:
                sqlite_db_encrypted_path = os.path.join(db_dir, f"face_embeddings_{self.detector_type}_{self.encoder_model_type}.db.enc")
            self.sqlite_db_encrypted_path = sqlite_db_encrypted_path
            self.sqlite_db_path = None
        else:
            # Unencrypted DB path
            if sqlite_db_path is None:
                sqlite_db_path = os.path.join(db_dir, f"face_embeddings_{self.detector_type}_{self.encoder_model_type}.db")
            self.sqlite_db_encrypted_path = None
            self.sqlite_db_path = sqlite_db_path

        # Initialize Database Manager
        self.db_manager = DatabaseManager(
            sqlite_db_path=self.sqlite_db_path,
            sqlite_db_encrypted_path=self.sqlite_db_encrypted_path,
            encryptor=self.encryptor,
            embedding_dim=self.embedding_dim,
            detector_type=self.detector_type,
            encoder_model_type=self.encoder_model_type
        )

        # Initialize HNSW Manager
        self.hnsw_manager = HNSWManager(
            embedding_dim=self.embedding_dim,
            hnsw_index_path=hnsw_index_path,
            hnsw_labels_path=hnsw_labels_path,
            hnsw_db_ids_path=hnsw_db_ids_path,
            encryptor=self.encryptor,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_m=hnsw_m
        )

        # Load from DB if HNSW index is empty
        if self.hnsw_manager.hnsw_id_counter == 0:
            rows = self.db_manager.load_all_embeddings()
            self.hnsw_manager.load_embeddings_into_hnswlib(rows)
            self.hnsw_manager.save_hnswlib_index()

        self.recent_embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
        self.recent_labels = []
        self.max_recent = max_recent

        self.new_embeddings = []
        self.new_labels = []
        self.max_new = max_new

        self.total_detection_time = 0.0
        self.total_encoding_time = 0.0
        self.frame_count = 0
        self.start_time = None

        self.unknown_faces = {}
        self.face_tracker = initialize_tracker()
        self.track_id_to_label = {}

    def _add_to_sqlite(self, label: str, embedding: np.ndarray) -> int:
        return self.db_manager.add_face_embedding(label, embedding)

    def _flush_new_embeddings(self):
        try:
            for label, embedding in zip(self.new_labels, self.new_embeddings):
                db_id = self._add_to_sqlite(label, embedding)
                if db_id == -1:
                    continue
                self.hnsw_manager.add_embedding(embedding, label, db_id)
            self.new_embeddings = []
            self.new_labels = []
            self.hnsw_manager.save_hnswlib_index()
        except Exception as e:
            logging.error(f"Error flushing new embeddings: {e}")

    def save_database_to_sqlite(self):
        try:
            if self.new_embeddings:
                self._flush_new_embeddings()
            logging.info("Saved new embeddings to SQLite and HNSWlib index.")
        except Exception as e:
            logging.error(f"Error in save_database_to_sqlite: {e}")

    def add_face(self, image: np.ndarray, label: str) -> bool:
        """
        Adds a face to the database.
        Note: This function buffers new faces and adds them to the database in batches.
        The buffer is flushed when it reaches `max_new` or when `save_database_to_sqlite` is called.
        """
        try:
            faces = self.detector.extract_faces(image, align=self.align)
            if not faces:
                logging.warning("No faces detected to add.")
                return False

            success = False
            for face_img in faces:
                preprocessed_face = self.encoder.preprocess_for_encoder(face_img)
                start_encoding = time.time()
                embedding = self.encoder.encode(preprocessed_face)
                encoding_time = time.time() - start_encoding
                self.total_encoding_time += encoding_time

                if len(embedding.shape) > 1:
                    embedding = embedding.squeeze()

                if embedding.shape[0] != self.embedding_dim:
                    logging.error(f"Invalid embedding size: expected {self.embedding_dim}, got {embedding.shape[0]}")
                    continue

                norm = np.linalg.norm(embedding)
                if norm == 0:
                    logging.error("Received zero vector from encoder. Skipping this face.")
                    continue
                embedding = embedding / norm

                if self.hnsw_manager.hnsw_index.get_current_count() > 0:
                    labels, distances = self.hnsw_manager.query(embedding, k=1)
                    if labels is not None and labels.size > 0:
                        cosine_similarity = 1 - distances[0][0]
                        if cosine_similarity > self.similarity_threshold:
                            logging.info(
                                f"Face is too similar to an existing face (Label: {self.hnsw_manager.hnsw_labels[labels[0][0]]}). Not adding.")
                            continue

                self.new_embeddings.append(embedding)
                self.new_labels.append(label)
                logging.info(f"Added face for label '{label}' to the new embeddings buffer.")
                success = True

            if len(self.new_embeddings) >= self.max_new:
                self._flush_new_embeddings()

            return success
        except Exception as e:
            logging.error(f"Error in add_face: {e}")
            return False

    def _add_to_recent_embeddings(self, embedding: np.ndarray, label: str):
        self.recent_embeddings = np.vstack([self.recent_embeddings, embedding])
        self.recent_labels.append(label)
        if self.recent_embeddings.shape[0] > self.max_recent:
            self.recent_embeddings = self.recent_embeddings[1:]
            self.recent_labels.pop(0)


    def update_label(self, hnsw_id: int, new_label: str):
        self.hnsw_manager.update_label(hnsw_id, new_label, self.db_manager.cursor, self.db_manager.conn,
                                       similarity_threshold=self.similarity_threshold)

    def shrink_db_ids(self, similarity_threshold: float = 0.75):
        """
        Attempts to unify labels for all embeddings in the DB that are similar enough.
        Algorithm:
        1. Load all embeddings from HNSW (already loaded in memory).
        2. Track processed hnsw_ids to avoid double processing.
        3. For each hnsw_id, find similar embeddings.
        4. Determine final label:
           - If multiple known labels differ => skip
           - Else unify under a single known label if present, else choose one label (or a custom chosen label).
        """
        processed = set()
        total_unifications = 0
        for hid in range(len(self.hnsw_manager.hnsw_labels)):
            if hid in processed:
                continue
            label = self.hnsw_manager.hnsw_labels[hid]
            db_id = self.hnsw_manager.hnsw_db_ids[hid]

            embedding = self.hnsw_manager._get_embedding_from_db_id(db_id, self.db_manager.cursor)
            if embedding is None:
                continue

            similar_ids = self.hnsw_manager.find_similar_embeddings(embedding, similarity_threshold, k=50)
            if len(similar_ids) <= 1:
                processed.add(hid)
                continue

            # Check their labels
            current_labels = [self.hnsw_manager.hnsw_labels[sid] for sid in similar_ids]
            known_labels = [lbl for lbl in current_labels if not lbl.lower().startswith("unknown")]

            if len(set(known_labels)) > 1:
                # Conflict
                for sid in similar_ids:
                    processed.add(sid)
                continue

            # Decide final label
            if known_labels:
                final_label = known_labels[0]
            else:
                # All unknown, pick the label of the first
                final_label = label

            # Unify
            self.hnsw_manager.unify_labels(similar_ids, final_label, self.db_manager.cursor, self.db_manager.conn)
            total_unifications += 1
            for sid in similar_ids:
                processed.add(sid)
        logging.info(f"DB ID shrinking completed with {total_unifications} unification operations.")

    def _generate_unique_label(self) -> str:
        unique_id = uuid.uuid4().hex[:8]
        unique_label = f"Unknown_{unique_id}"
        return unique_label

    def _handle_unknown_embedding(self, track_id: int, embedding: np.ndarray, rename_label: str = None) -> str:
        if rename_label:
            self.new_embeddings.append(embedding)
            self.new_labels.append(rename_label)
            logging.info(f"Added face with label '{rename_label}' to the new embeddings buffer.")
            if self.hnsw_manager.hnsw_id_counter < 100000:
                db_id = self._add_to_sqlite(rename_label, embedding)
                if db_id != -1:
                    self.hnsw_manager.add_embedding(embedding, rename_label, db_id)
            else:
                logging.warning("HNSWlib index has reached its maximum capacity. Cannot add more embeddings.")
            self._flush_new_embeddings()
            return rename_label
        else:
            if track_id not in self.unknown_faces:
                self.unknown_faces[track_id] = {'embeddings': [embedding], 'count': 1}
            else:
                self.unknown_faces[track_id]['embeddings'].append(embedding)
                self.unknown_faces[track_id]['count'] += 1

            if self.unknown_faces[track_id]['count'] >= self.unknown_trigger_count:
                unique_label = self._generate_unique_label()
                avg_embedding = np.mean(self.unknown_faces[track_id]['embeddings'], axis=0)
                if self.hnsw_manager.hnsw_index.get_current_count() > 0:
                    labels, distances = self.hnsw_manager.query(avg_embedding, k=1)
                    if labels is not None and labels.size > 0:
                        cosine_similarity = 1 - distances[0][0]
                        if cosine_similarity > self.similarity_threshold:
                            existing_label = self.hnsw_manager.hnsw_labels[labels[0][0]] if labels[0][0] < len(self.hnsw_manager.hnsw_labels) else "Unknown"
                            logging.info("Unknown face is too similar to an existing face. Not adding.")
                            return existing_label

                self.new_embeddings.append(avg_embedding)
                self.new_labels.append(unique_label)
                logging.info(f"Added unknown face as '{unique_label}' to the new embeddings buffer.")

                if self.hnsw_manager.hnsw_id_counter < 100000:
                    db_id = self._add_to_sqlite(unique_label, avg_embedding)
                    if db_id != -1:
                        self.hnsw_manager.add_embedding(avg_embedding, unique_label, db_id)
                else:
                    logging.warning("HNSWlib index has reached its maximum capacity. Cannot add more embeddings.")

                self._flush_new_embeddings()
                del self.unknown_faces[track_id]
                return unique_label
            else:
                return "Unknown"

    def recognize_faces(self, image: np.ndarray, rename_label: str = None):
        results = []
        if self.start_time is None:
            self.start_time = time.time()

        self.frame_index += 1

        # Only run face detection every self.detection_interval frames
        if (self.frame_index % self.detection_interval == 0):
            start_detection = time.time()
            detected_faces = self.detector.detect_faces(image)
            detection_time = time.time() - start_detection
            self.total_detection_time += detection_time

            formatted_detections = []
            for bbox_dict in detected_faces:
                bbox = bbox_dict.get('bbox', [0, 0, 0, 0])
                detection_confidence = bbox_dict.get('confidence', 1.0)
                formatted_detections.append({'bbox': bbox, 'confidence': detection_confidence})

            tracks = self.face_tracker.update(formatted_detections)
        else:
            tracks = self.face_tracker.update([])

        # Clean up inactive track IDs
        active_track_ids = set([trk['id'] for trk in tracks])
        inactive_track_ids = set(self.track_id_to_label.keys()) - active_track_ids
        for track_id in inactive_track_ids:
            del self.track_id_to_label[track_id]
            if track_id in self.unknown_faces:
                del self.unknown_faces[track_id]

        # Process each track
        for trk in tracks:
            track_id = trk['id']
            bbox = trk['bbox']

            if track_id in self.track_id_to_label:
                label = self.track_id_to_label[track_id]
                confidence = 1.0
            else:
                x, y, w, h = bbox
                x = max(0, x)
                y = max(0, y)
                w = max(0, w)
                h = max(0, h)
                face_img = image[y:y+h, x:x+w]
                if face_img.size == 0:
                    logging.warning(f"Face image has zero size for track ID {track_id}. Skipping.")
                    continue

                try:
                    preprocessed_face = self.encoder.preprocess_for_encoder(face_img)
                except Exception as e:
                    logging.error(f"Error preprocessing face for track ID {track_id}: {e}")
                    continue

                start_encoding = time.time()
                embedding = self.encoder.encode(preprocessed_face)
                encoding_time = time.time() - start_encoding
                self.total_encoding_time += encoding_time

                if len(embedding.shape) > 1:
                    embedding = embedding.squeeze()

                if embedding.shape[0] != self.embedding_dim:
                    logging.error(f"Invalid embedding size: expected {self.embedding_dim}, got {embedding.shape[0]}")
                    continue

                norm = np.linalg.norm(embedding)
                if norm == 0:
                    logging.error("Received zero vector from encoder. Skipping this face.")
                    continue
                embedding = embedding / norm

                label = "Unknown"
                confidence = 0.0

                # Check recent embeddings first
                if self.recent_embeddings.shape[0] > 0:
                    similarities = np.dot(self.recent_embeddings, embedding.T).flatten()
                    max_similarity = np.max(similarities)
                    max_index = np.argmax(similarities)
                    if max_similarity > self.similarity_threshold:
                        label = self.recent_labels[max_index]
                        confidence = float(max_similarity)

                # Query HNSW index if still unknown
                if label == "Unknown":
                    labels, distances = self.hnsw_manager.query(embedding, k=1)
                    if labels is not None and labels.size > 0:
                        cosine_similarity = 1 - distances[0][0]
                        if cosine_similarity > self.similarity_threshold:
                            hnsw_id = labels[0][0]
                            label = self.hnsw_manager.hnsw_labels[hnsw_id]
                            confidence = float(cosine_similarity)
                            if rename_label:
                                self.update_label(hnsw_id, rename_label)
                                label = rename_label

                # Handle unknown logic
                if label == "Unknown":
                    label = self._handle_unknown_embedding(track_id, embedding, rename_label)
                    confidence = 1.0

                self.track_id_to_label[track_id] = label
                self._add_to_recent_embeddings(embedding, label)

            if self.interested_label is not None and label != self.interested_label:
                continue

            results.append({
                'label': self.track_id_to_label[track_id],
                'confidence': float(confidence),
                'bbox': bbox
            })

        self.frame_count += 1
        return results

    def process_image(self, image_path: str, annotate: bool = True, save_path: str = None, label: str = None):
        try:
            timing = {}
            start_time = time.time()
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Image not found at path: {image_path}")
                return
            timing['Image Loading'] = time.time() - start_time

            start_time = time.time()
            detected_faces = self.detector.detect_faces(image)
            detection_time = time.time() - start_time
            self.total_detection_time += detection_time
            timing['Face Detection'] = detection_time

            recognized_faces = []
            new_embeddings_to_add = []
            new_labels_to_add = []

            if label:
                # Update existing embeddings
                for face_data in detected_faces:
                    bbox = face_data.get('bbox', [0, 0, 0, 0])
                    x, y, w, h = bbox
                    x = max(x, 0)
                    y = max(y, 0)
                    w = max(w, 0)
                    h = max(h, 0)
                    if w == 0 or h == 0:
                        logging.warning("Detected face with zero width or height.")
                        continue

                    start_time = time.time()
                    face_img = image[y:y + h, x:x + w]
                    if face_img.size == 0:
                        logging.warning("Extracted face image is empty, skipping.")
                        continue
                    timing_face_extraction = time.time() - start_time

                    start_time = time.time()
                    try:
                        preprocessed_face = self.encoder.preprocess_for_encoder(face_img)
                    except Exception as e:
                        logging.error(f"Error preprocessing face: {e}")
                        continue
                    timing_preprocessing = time.time() - start_time

                    start_time = time.time()
                    embedding = self.encoder.encode(preprocessed_face)
                    encoding_time = time.time() - start_time
                    self.total_encoding_time += encoding_time
                    timing['Face Encoding'] = timing.get('Face Encoding', 0) + encoding_time

                    if embedding.ndim > 1:
                        embedding = embedding.squeeze()
                    norm = np.linalg.norm(embedding)
                    if norm == 0:
                        logging.error("Received zero vector from encoder. Skipping this face.")
                        continue
                    embedding = embedding / norm

                    matched = False
                    if self.hnsw_manager.hnsw_index.get_current_count() > 0:
                        labels, distances = self.hnsw_manager.query(embedding, k=1)
                        if labels is not None and labels.size > 0:
                            cosine_similarity = 1 - distances[0][0]
                            if cosine_similarity > self.similarity_threshold:
                                hnsw_id = labels[0][0]
                                self.update_label(hnsw_id, label)
                                logging.info(f"Updated label for hnsw_id {hnsw_id} to '{label}'.")
                                matched = True

                    if not matched:
                        logging.warning("No matching face found to update with the provided label.")

                if save_path:
                    _ensure_parent_dir(save_path)
                    if self.encryption_password:
                        _, buffer = cv2.imencode('.jpg', image)
                        image_bytes = buffer.tobytes()
                        self.encryptor.encrypt_and_write(save_path, image_bytes)
                    else:
                        cv2.imwrite(save_path, image)
                        logging.info(f"Processed image saved to {save_path}")

                print("\n--- Image Processing Timings ---")
                for step, duration in timing.items():
                    print(f"{step}: {duration:.4f} seconds")
                total_time = sum(timing.values())
                print(f"Total Processing Time: {total_time:.4f} seconds\n")

            else:
                # Standard recognition
                for face_data in detected_faces:
                    bbox = face_data.get('bbox', [0, 0, 0, 0])
                    x, y, w, h = bbox
                    x = max(x, 0)
                    y = max(y, 0)
                    w = max(w, 0)
                    h = max(h, 0)

                    if w == 0 or h == 0:
                        logging.warning("Detected face with zero width or height.")
                        continue

                    start_time = time.time()
                    face_img = image[y:y + h, x:x + w]
                    if face_img.size == 0:
                        logging.warning("Extracted face image is empty, skipping.")
                        continue
                    timing_face_extraction = time.time() - start_time

                    start_time = time.time()
                    try:
                        preprocessed_face = self.encoder.preprocess_for_encoder(face_img)
                    except Exception as e:
                        logging.error(f"Error preprocessing face: {e}")
                        continue
                    timing_preprocessing = time.time() - start_time

                    start_time = time.time()
                    embedding = self.encoder.encode(preprocessed_face)
                    encoding_time = time.time() - start_time
                    self.total_encoding_time += encoding_time
                    timing['Face Encoding'] = timing.get('Face Encoding', 0) + encoding_time

                    if embedding.ndim > 1:
                        embedding = embedding.squeeze()
                    norm = np.linalg.norm(embedding)
                    if norm == 0:
                        logging.error("Received zero vector from encoder. Skipping this face.")
                        continue
                    embedding = embedding / norm

                    start_time = time.time()
                    label_found = None
                    confidence = 0.0
                    if self.hnsw_manager.hnsw_index.get_current_count() > 0:
                        labels, distances = self.hnsw_manager.query(embedding, k=1)
                        if labels is not None and labels.size > 0:
                            cosine_similarity = 1 - distances[0][0]
                            if cosine_similarity > self.similarity_threshold:
                                hnsw_id = labels[0][0]
                                label_found = self.hnsw_manager.hnsw_labels[hnsw_id]
                                confidence = float(cosine_similarity)
                    timing['Face Recognition'] = timing.get('Face Recognition', 0) + (time.time() - start_time)

                    start_time = time.time()
                    if label_found is None:
                        unique_label = self._generate_unique_label()
                        label_found = unique_label
                        new_embeddings_to_add.append(embedding)
                        new_labels_to_add.append(label_found)
                    timing['Unknown Handling'] = time.time() - start_time

                    recognized_faces.append({
                        'label': label_found,
                        'bbox': bbox
                    })

                start_time = time.time()
                if new_embeddings_to_add:
                    for lbl, emb in zip(new_labels_to_add, new_embeddings_to_add):
                        db_id = self._add_to_sqlite(lbl, emb)
                        if db_id != -1:
                            if self.hnsw_manager.hnsw_id_counter < 100000:
                                self.hnsw_manager.add_embedding(emb, lbl, db_id)
                            else:
                                logging.warning("HNSWlib index has reached its maximum capacity, cannot add more embeddings.")
                    self.hnsw_manager.save_hnswlib_index()
                timing['Flushing Embeddings'] = time.time() - start_time

                start_time = time.time()
                annotated_image = image.copy()
                if annotate:
                    for face in recognized_faces:
                        bbox = face['bbox']
                        lbl = face['label']
                        x, y, w, h = bbox
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        text = f"{lbl}"
                        cv2.putText(annotated_image, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                timing['Image Annotation'] = time.time() - start_time

                if self.show:
                    cv2.imshow('Face Recognition - Image', annotated_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                start_time = time.time()
                if save_path:
                    _ensure_parent_dir(save_path)
                    if self.encryption_password:
                        _, buffer = cv2.imencode('.jpg', annotated_image)
                        image_bytes = buffer.tobytes()
                        self.encryptor.encrypt_and_write(save_path, image_bytes)
                    else:
                        cv2.imwrite(save_path, annotated_image)
                        logging.info(f"Annotated image saved to {save_path}")
                timing['Image Saving'] = time.time() - start_time

                total_time = sum(timing.values())
                print("\n--- Image Processing Timings ---")
                for step, duration in timing.items():
                    print(f"{step}: {duration:.4f} seconds")
                print(f"Total Processing Time: {total_time:.4f} seconds\n")

        except Exception as e:
            logging.error(f"Error in process_image: {e}")

    def resize_frame_to_screen(self, frame):
        try:
            # Get the primary monitor's dimensions
            monitor = get_monitors()[0]
            screen_width, screen_height = monitor.width, monitor.height
        except Exception as e:
            logging.warning(f"Could not get screen resolution using screeninfo: {e}. Defaulting to 1920x1080.")
            screen_width, screen_height = 1920, 1080

        original_height, original_width = frame.shape[:2]

        # Add a check for zero dimensions
        if original_height == 0 or original_width == 0:
            logging.warning("Cannot resize a frame with zero height or width.")
            return frame

        frame_aspect_ratio = original_width / original_height
        screen_aspect_ratio = screen_width / screen_height

        if frame_aspect_ratio > screen_aspect_ratio:
            # Frame is wider than the screen, fit to screen width
            new_width = screen_width
            new_height = int(new_width / frame_aspect_ratio)
        else:
            # Frame is taller than or equal to the screen, fit to screen height
            new_height = screen_height
            new_width = int(new_height * frame_aspect_ratio)

        # Ensure new dimensions are not zero
        if new_width <= 0 or new_height <= 0:
            logging.warning(f"Calculated new dimensions are invalid: {new_width}x{new_height}. Skipping resize.")
            return frame

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame

    def _process_stream(self, cap, annotate: bool = True, save_path: str = None, duration: int = 0, name: str = None, stream_type: str = "video"):
        try:
            if save_path:
                _ensure_parent_dir(save_path)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                temp_video_path = None

                if self.encryption_password:
                    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=".avi")
                    os.close(temp_video_fd)
                    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                    if not out.isOpened():
                        logging.error("Failed to open temporary video writer.")
                        return
                    logging.info(f"Writing annotated frames to temporary file: {temp_video_path}")
                else:
                    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                    if not out.isOpened():
                        logging.error(f"Failed to open video writer for {save_path}.")
                        return
                    logging.info(f"Saving annotated video to {save_path}")
            else:
                out = None

            self.total_detection_time = 0.0
            self.total_encoding_time = 0.0
            self.frame_count = 0
            self.start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    if stream_type == "webcam":
                        logging.error("Failed to grab frame from webcam.")
                    break

                recognized_faces = self.recognize_faces(frame, rename_label=name)

                annotated_frame = frame.copy()
                if annotate:
                    for face in recognized_faces:
                        bbox = face['bbox']
                        label = face['label']
                        confidence = face['confidence']
                        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 2)
                        text = f"{label} ({confidence:.2f})"
                        cv2.putText(annotated_frame, text, (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if self.show:
                    display_frame = self.resize_frame_to_screen(annotated_frame)
                    cv2.imshow(f'Face Recognition - {stream_type.capitalize()}', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info(f"User requested to quit {stream_type} processing.")
                        break

                if out:
                    out.write(annotated_frame)

                if duration > 0 and (time.time() - self.start_time) >= duration:
                    logging.info(f"Duration of {duration} seconds reached. Stopping.")
                    break

            cap.release()
            if out:
                out.release()
                if self.encryption_password and save_path and temp_video_path:
                    try:
                        with open(temp_video_path, 'rb') as tmp_video:
                            video_bytes = tmp_video.read()
                        self.encryptor.encrypt_and_write(save_path, video_bytes)
                        logging.info(f"Encrypted video saved to {save_path}")
                        os.remove(temp_video_path)
                        logging.info(f"Temporary video file {temp_video_path} removed.")
                    except Exception as e:
                        logging.error(f"Error during encryption of video: {e}")
                elif not self.encryption_password and save_path:
                    logging.info(f"Annotated video saved to {save_path}")

            if self.show:
                cv2.destroyAllWindows()

        except Exception as e:
            logging.error(f"Error in _process_stream: {e}")
        finally:
            if cap:
                cap.release()
            if self.show:
                cv2.destroyAllWindows()

    def process_video(self, video_path: str, annotate: bool = True, save_path: str = None):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video file: {video_path}")
                return
            self._process_stream(cap, annotate, save_path, stream_type="video")
        except Exception as e:
            logging.error(f"Error in process_video: {e}")

    def process_webcam(self, annotate: bool = True, save_path: str = None, duration: int = 0, name: str = None):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("Cannot open webcam.")
                return
            self._process_stream(cap, annotate, save_path, duration, name, stream_type="webcam")
        except Exception as e:
            logging.error(f"Error in process_webcam: {e}")

    def close(self):
        try:
            self.save_database_to_sqlite()
            self.hnsw_manager.save_hnswlib_index()
            self.db_manager.save()
            logging.info("Closed FaceRecognition system and saved all data.")
        except Exception as e:
            logging.error(f"Error closing FaceRecognition system: {e}")

        if self.enable_logging and self.frame_count > 0 and self.start_time is not None:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            avg_detection_time = self.total_detection_time / self.frame_count
            avg_encoding_time = self.total_encoding_time / self.frame_count
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            logging.info(f"Total frames processed: {self.frame_count}")
            logging.info(f"Total processing time: {elapsed_time:.2f} seconds")
            logging.info(f"Average FPS: {fps:.2f}")
            logging.info(f"Average Detection Time: {avg_detection_time * 1000:.2f} ms/frame")
            logging.info(f"Average Encoding Time: {avg_encoding_time * 1000:.2f} ms/frame")
