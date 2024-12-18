import logging
import sqlite3
import time
import uuid
import os
import pickle
import tempfile
import base64
from typing import List, Dict, Any
import cv2
import numpy as np
import hnswlib
import psutil
from facenet_gpu import FaceNetClient
from yunet_face_detector import detect_faces as detect_faces_yunet, extract_faces as extract_faces_yunet
from sort_UKF import Sort
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import ctypes

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
        :param interested_label: If set, only this label will be maintained/tracked in recognition results.
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

        # New attribute to filter recognition results for a specific label of interest
        self.interested_label = interested_label

        if self.enable_logging:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

        self.encryption_password = encryption_password
        if self.encryption_password:
            self._initialize_encryption()
            logging.info("Encryption is enabled for file operations.")
        else:
            logging.info("Encryption is disabled.")

        # Initialize face detector
        if self.detector_type == 'yunet':
            self.detect_faces = detect_faces_yunet
            self.extract_faces = extract_faces_yunet
            logging.info("Initialized Yunet face detector.")
        elif self.detector_type == 'retinaface':
            import retinaface_face_detector
            self.detect_faces = retinaface_face_detector.detect_faces
            self.extract_faces = retinaface_face_detector.extract_faces
            logging.info("Initialized RetinaFace detector.")
        elif self.detector_type == 'mediapipe':
            import mediapipe_face_detector
            self.detect_faces = mediapipe_face_detector.detect_faces
            self.extract_faces = mediapipe_face_detector.extract_faces
            logging.info("Initialized Mediapipe Face Detection.")
        else:
            raise ValueError("Invalid detector_type. Choose from 'yunet', 'retinaface', 'mediapipe'.")

        # Initialize FaceNet encoder
        self.encoder = FaceNetClient(model_type=encoder_model_type, mode=encoder_mode)
        logging.info(f"Initialized FaceNet-{self.encoder.output_shape} encoder in {encoder_mode} mode.")

        if hnsw_index_path is None:
            hnsw_index_path = f"hnsw_index_{self.detector_type}_{self.encoder_model_type}.bin"
        if hnsw_labels_path is None:
            hnsw_labels_path = f"hnsw_labels_{self.detector_type}_{self.encoder_model_type}.pkl"
        if hnsw_db_ids_path is None:
            hnsw_db_ids_path = f"hnsw_db_ids_{self.detector_type}_{self.encoder_model_type}.pkl"

        if self.encryption_password:
            if sqlite_db_encrypted_path is None:
                sqlite_db_encrypted_path = f"face_embeddings_{self.detector_type}_{self.encoder_model_type}.db.enc"
            self.sqlite_db_encrypted_path = sqlite_db_encrypted_path
            self.sqlite_db_path = None
        else:
            if sqlite_db_path is None:
                sqlite_db_path = f"face_embeddings_{self.detector_type}_{self.encoder_model_type}.db"
            self.sqlite_db_encrypted_path = None
            self.sqlite_db_path = sqlite_db_path

        self.sqlite_temp_path = None
        self._initialize_sqlite(sqlite_db_path=self.sqlite_db_path, sqlite_db_encrypted_path=self.sqlite_db_encrypted_path)

        # Initialize HNSW index
        self.hnsw_index_path = hnsw_index_path
        self.hnsw_labels_path = hnsw_labels_path
        self.hnsw_db_ids_path = hnsw_db_ids_path
        self.embedding_dim = self.encoder.output_shape
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.hnsw_labels = []
        self.hnsw_db_ids = []
        self.hnsw_id_counter = 0

        if self._files_exist([self.hnsw_index_path, self.hnsw_labels_path, self.hnsw_db_ids_path]):
            self._load_hnswlib_index()
            logging.info("Loaded existing HNSWlib index and mappings from disk.")
        else:
            self.hnsw_index.init_index(max_elements=100000, ef_construction=hnsw_ef_construction, M=hnsw_m)
            self.hnsw_index.set_ef(50)
            logging.info("Initialized new HNSWlib index.")
            self._load_embeddings_into_hnswlib()
            self._save_hnswlib_index()

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

        self.face_tracker = Sort(max_age=4, min_hits=4, iou_threshold=0.3)
        self.track_id_to_label = {}

    def _initialize_encryption(self):
        self.backend = default_backend()
        self.iterations = 100000
        self.key_length = 32

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.key_length,
            salt=salt,
            iterations=self.iterations,
            backend=self.backend
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _encrypt_data(self, data: bytes) -> bytes:
        salt = os.urandom(16)
        key = self._derive_key(self.encryption_password, salt)
        f = Fernet(key)
        encrypted = f.encrypt(data)
        return salt + encrypted

    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        key = self._derive_key(self.encryption_password, salt)
        f = Fernet(key)
        return f.decrypt(encrypted)

    def _files_exist(self, paths: List[str]) -> bool:
        return all(os.path.exists(path) for path in paths)

    def _encrypt_and_write(self, file_path: str, data: bytes):
        encrypted_data = self._encrypt_data(data)
        with open(file_path, 'wb') as f:
            f.write(encrypted_data)
        logging.info(f"Encrypted and saved data to {file_path}.")

    def _read_and_decrypt(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = self._decrypt_data(encrypted_data)
        logging.info(f"Decrypted and loaded data from {file_path}.")
        return decrypted_data

    def _initialize_sqlite(self, sqlite_db_path: str, sqlite_db_encrypted_path: str):
        if self.encryption_password:
            if os.path.exists(sqlite_db_encrypted_path):
                self.sqlite_temp_fd, self.sqlite_temp_path = tempfile.mkstemp(suffix=".db")
                with os.fdopen(self.sqlite_temp_fd, 'wb') as tmp:
                    decrypted_data = self._decrypt_data(open(sqlite_db_encrypted_path, 'rb').read())
                    tmp.write(decrypted_data)
                self.conn = sqlite3.connect(self.sqlite_temp_path)
                logging.info("Decrypted and connected to the temporary SQLite database.")
            else:
                self.sqlite_temp_fd, self.sqlite_temp_path = tempfile.mkstemp(suffix=".db")
                self.conn = sqlite3.connect(self.sqlite_temp_path)
                logging.info("Initialized a new temporary SQLite database.")
        else:
            if sqlite_db_path is None:
                sqlite_db_path = f"face_embeddings_{self.detector_type}_{self.encoder_model_type}.db"
            self.conn = sqlite3.connect(sqlite_db_path)
            logging.info("Initialized SQLite database for persistent storage.")

        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        self.conn.commit()
        logging.info("Initialized SQLite database for persistent storage.")

    def _save_sqlite_database(self):
        if self.encryption_password and self.sqlite_temp_path:
            self.conn.commit()
            self.conn.close()
            with open(self.sqlite_temp_path, 'rb') as tmp:
                decrypted_data = tmp.read()
            self._encrypt_and_write(self.sqlite_db_encrypted_path, decrypted_data)
            os.remove(self.sqlite_temp_path)
            self.sqlite_temp_path = None
            logging.info("Encrypted and saved the SQLite database.")
        elif not self.encryption_password:
            self.conn.commit()
            self.conn.close()
            logging.info("Saved and closed the unencrypted SQLite database.")

    def _load_hnswlib_index(self):
        try:
            if self.encryption_password:
                index_data = self._read_and_decrypt(self.hnsw_index_path)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_index:
                    tmp_index.write(index_data)
                    tmp_index_path = tmp_index.name
                self.hnsw_index.load_index(tmp_index_path, max_elements=100000)
                os.remove(tmp_index_path)

                labels_data = self._read_and_decrypt(self.hnsw_labels_path)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_labels:
                    tmp_labels.write(labels_data)
                    tmp_labels_path = tmp_labels.name
                with open(tmp_labels_path, 'rb') as f:
                    self.hnsw_labels = pickle.load(f)
                os.remove(tmp_labels_path)

                db_ids_data = self._read_and_decrypt(self.hnsw_db_ids_path)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_db_ids:
                    tmp_db_ids.write(db_ids_data)
                    tmp_db_ids_path = tmp_db_ids.name
                with open(tmp_db_ids_path, 'rb') as f:
                    self.hnsw_db_ids = pickle.load(f)
                os.remove(tmp_db_ids_path)
            else:
                self.hnsw_index.load_index(self.hnsw_index_path, max_elements=100000)
                with open(self.hnsw_labels_path, 'rb') as f:
                    self.hnsw_labels = pickle.load(f)
                with open(self.hnsw_db_ids_path, 'rb') as f:
                    self.hnsw_db_ids = pickle.load(f)
            self.hnsw_id_counter = len(self.hnsw_labels)
            logging.info("Loaded HNSWlib index and mappings from disk.")
        except Exception as e:
            logging.error(f"Error loading HNSWlib index: {e}")
            self.hnsw_index.init_index(max_elements=100000, ef_construction=200, M=16)
            self.hnsw_index.set_ef(50)
            self.hnsw_labels = []
            self.hnsw_db_ids = []
            self.hnsw_id_counter = 0
            logging.info("Initialized a new HNSWlib index due to loading failure.")

    def _save_hnswlib_index(self):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_index:
                self.hnsw_index.save_index(tmp_index.name)
                tmp_index_path = tmp_index.name
            with open(tmp_index_path, 'rb') as f:
                index_data = f.read()
            os.remove(tmp_index_path)

            if self.encryption_password:
                self._encrypt_and_write(self.hnsw_index_path, index_data)
            else:
                with open(self.hnsw_index_path, 'wb') as f:
                    f.write(index_data)
                logging.info(f"Saved HNSWlib index to {self.hnsw_index_path}.")

            labels_data = pickle.dumps(self.hnsw_labels)
            if self.encryption_password:
                self._encrypt_and_write(self.hnsw_labels_path, labels_data)
            else:
                with open(self.hnsw_labels_path, 'wb') as f:
                    f.write(labels_data)
                logging.info(f"Saved HNSWlib labels to {self.hnsw_labels_path}.")

            db_ids_data = pickle.dumps(self.hnsw_db_ids)
            if self.encryption_password:
                self._encrypt_and_write(self.hnsw_db_ids_path, db_ids_data)
            else:
                with open(self.hnsw_db_ids_path, 'wb') as f:
                    f.write(db_ids_data)
                logging.info(f"Saved HNSWlib DB IDs to {self.hnsw_db_ids_path}.")

            logging.info("Saved HNSWlib index and mappings to disk.")
        except Exception as e:
            logging.error(f"Error saving HNSWlib index: {e}")

    def _load_embeddings_into_hnswlib(self):
        try:
            self.cursor.execute('SELECT id, label, embedding FROM faces')
            rows = self.cursor.fetchall()
            for row in rows:
                db_id, label, embedding_blob = row
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                if embedding.shape[0] != self.embedding_dim:
                    logging.warning(f"Embedding size mismatch for label '{label}'. Skipping.")
                    continue
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    logging.warning(f"Zero vector found for label '{label}'. Skipping.")
                    continue
                embedding = embedding / norm
                self.hnsw_index.add_items(embedding, self.hnsw_id_counter)
                self.hnsw_labels.append(label)
                self.hnsw_db_ids.append(db_id)
                self.hnsw_id_counter += 1
            logging.info("Loaded embeddings into HNSWlib index from SQLite database.")
        except Exception as e:
            logging.error(f"Error loading embeddings into HNSWlib: {e}")

    def add_face(self, image: np.ndarray, label: str) -> bool:
        try:
            faces = self.extract_faces(image, align=self.align)
            if not faces:
                logging.warning("No faces detected to add.")
                return False

            success = False
            for face_img in faces:
                preprocessed_face = self._preprocess_for_encoder(face_img)
                start_encoding = time.time()
                embedding = self.encoder(preprocessed_face)
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

                if self.hnsw_index.get_current_count() > 0:
                    labels, distances = self.hnsw_index.knn_query(embedding, k=1)
                    if labels.size > 0:
                        cosine_similarity = 1 - distances[0][0]
                        if cosine_similarity > self.similarity_threshold:
                            logging.info(
                                f"Face is too similar to an existing face (Label: {self.hnsw_labels[labels[0][0]]}). Not adding.")
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

    def _add_to_sqlite(self, label: str, embedding: np.ndarray) -> int:
        try:
            embedding_blob = embedding.tobytes()
            self.cursor.execute('INSERT INTO faces (label, embedding) VALUES (?, ?)', (label, embedding_blob))
            self.conn.commit()
            db_id = self.cursor.lastrowid
            logging.info(f"Added face for label '{label}' to SQLite database with id {db_id}.")
            return db_id
        except Exception as e:
            logging.error(f"Error adding face to SQLite: {e}")
            return -1

    def _flush_new_embeddings(self):
        try:
            for label, embedding in zip(self.new_labels, self.new_embeddings):
                db_id = self._add_to_sqlite(label, embedding)
                if db_id == -1:
                    continue
                if self.hnsw_id_counter < 100000:
                    self.hnsw_index.add_items(embedding, self.hnsw_id_counter)
                    self.hnsw_labels.append(label)
                    self.hnsw_db_ids.append(db_id)
                    logging.info(f"Added '{label}' to HNSWlib index with hnsw_id {self.hnsw_id_counter}.")
                    self.hnsw_id_counter += 1
                else:
                    logging.warning("HNSWlib index has reached its maximum capacity. Cannot add more embeddings.")

            self.new_embeddings = []
            self.new_labels = []
            self._save_hnswlib_index()
        except Exception as e:
            logging.error(f"Error flushing new embeddings: {e}")

    def save_database_to_sqlite(self):
        try:
            if self.new_embeddings:
                self._flush_new_embeddings()
            logging.info("Saved new embeddings to SQLite and HNSWlib index.")
        except Exception as e:
            logging.error(f"Error in save_database_to_sqlite: {e}")

    def _preprocess_for_encoder(self, face_img: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(face_img, self.encoder.input_shape, interpolation=cv2.INTER_AREA)
        img = resized_img.astype(np.float32) / 255.0

        if img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            raise ValueError("Face image has incorrect shape for encoder.")

        img = np.expand_dims(img, axis=0)
        return img

    def recognize_faces(self, image: np.ndarray, rename_label: str = None) -> List[Dict[str, Any]]:
        results = []
        if self.start_time is None:
            self.start_time = time.time()

        self.frame_index += 1

        # Only run face detection every self.detection_interval frames
        if (self.frame_index % self.detection_interval == 0):
            start_detection = time.time()
            detected_faces = self.detect_faces(image)
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
                    preprocessed_face = self._preprocess_for_encoder(face_img)
                except Exception as e:
                    logging.error(f"Error preprocessing face for track ID {track_id}: {e}")
                    continue

                start_encoding = time.time()
                embedding = self.encoder(preprocessed_face)
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
                    if self.hnsw_index.get_current_count() > 0:
                        labels, distances = self.hnsw_index.knn_query(embedding, k=1)
                        if labels.size > 0:
                            cosine_similarity = 1 - distances[0][0]
                            if cosine_similarity > self.similarity_threshold:
                                hnsw_id = labels[0][0]
                                label = self.hnsw_labels[hnsw_id]
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

            # NEW FILTERING: If there's an interested_label set, ignore faces not matching it
            if self.interested_label is not None and label != self.interested_label:
                continue

            results.append({
                'label': self.track_id_to_label[track_id],
                'confidence': float(confidence),
                'bbox': bbox
            })

        self.frame_count += 1
        return results

    def _handle_unknown_embedding(self, track_id: int, embedding: np.ndarray, rename_label: str = None) -> str:
        if rename_label:
            self.new_embeddings.append(embedding)
            self.new_labels.append(rename_label)
            logging.info(f"Added face with label '{rename_label}' to the new embeddings buffer.")
            if self.hnsw_id_counter < 100000:
                db_id = self._add_to_sqlite(rename_label, embedding)
                if db_id != -1:
                    self.hnsw_index.add_items(embedding, self.hnsw_id_counter)
                    self.hnsw_labels.append(rename_label)
                    self.hnsw_db_ids.append(db_id)
                    logging.info(f"Added '{rename_label}' to HNSWlib index with hnsw_id {self.hnsw_id_counter}.")
                    self.hnsw_id_counter += 1
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
                if self.hnsw_index.get_current_count() > 0:
                    labels, distances = self.hnsw_index.knn_query(avg_embedding, k=1)
                    if labels.size > 0:
                        cosine_similarity = 1 - distances[0][0]
                        if cosine_similarity > self.similarity_threshold:
                            existing_label = self.hnsw_labels[labels[0][0]] if labels[0][0] < len(self.hnsw_labels) else "Unknown"
                            logging.info("Unknown face is too similar to an existing face. Not adding.")
                            return existing_label

                self.new_embeddings.append(avg_embedding)
                self.new_labels.append(unique_label)
                logging.info(f"Added unknown face as '{unique_label}' to the new embeddings buffer.")

                if self.hnsw_id_counter < 100000:
                    db_id = self._add_to_sqlite(unique_label, avg_embedding)
                    if db_id != -1:
                        self.hnsw_index.add_items(avg_embedding, self.hnsw_id_counter)
                        self.hnsw_labels.append(unique_label)
                        self.hnsw_db_ids.append(db_id)
                        logging.info(f"Added '{unique_label}' to HNSWlib index with hnsw_id {self.hnsw_id_counter}.")
                        self.hnsw_id_counter += 1
                else:
                    logging.warning("HNSWlib index has reached its maximum capacity. Cannot add more embeddings.")

                self._flush_new_embeddings()
                del self.unknown_faces[track_id]
                return unique_label
            else:
                return "Unknown"

    def _generate_unique_label(self) -> str:
        unique_id = uuid.uuid4().hex[:8]
        unique_label = f"Unknown_{unique_id}"
        return unique_label

    def _add_to_recent_embeddings(self, embedding: np.ndarray, label: str):
        self.recent_embeddings = np.vstack([self.recent_embeddings, embedding])
        self.recent_labels.append(label)
        if self.recent_embeddings.shape[0] > self.max_recent:
            self.recent_embeddings = self.recent_embeddings[1:]
            self.recent_labels.pop(0)

    def update_label(self, hnsw_id: int, new_label: str):
        try:
            if hnsw_id < 0 or hnsw_id >= len(self.hnsw_db_ids):
                logging.error("Invalid hnsw_id for update_label.")
                return
            db_id = self.hnsw_db_ids[hnsw_id]
            self.hnsw_labels[hnsw_id] = new_label
            self.cursor.execute('UPDATE faces SET label = ? WHERE id = ?', (new_label, db_id))
            self.conn.commit()
            logging.info(f"Updated label for hnsw_id {hnsw_id} (db_id {db_id}) to '{new_label}'.")
            self._save_hnswlib_index()
        except Exception as e:
            logging.error(f"Error updating label: {e}")

    def process_image(self, image_path: str, annotate: bool = True, save_path: str = None, label: str = None):
        try:
            timing = {}

            # Step 1: Read the Image
            start_time = time.time()
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Image not found at path: {image_path}")
                return
            timing['Image Loading'] = time.time() - start_time

            # Step 2: Detect Faces
            start_time = time.time()
            detected_faces = self.detect_faces(image)
            detection_time = time.time() - start_time
            self.total_detection_time += detection_time
            timing['Face Detection'] = detection_time

            recognized_faces = []
            new_embeddings_to_add = []
            new_labels_to_add = []

            if label:
                # If label is provided, update existing embeddings with this label
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

                    # Extract Face Image
                    start_time = time.time()
                    face_img = image[y:y + h, x:x + w]
                    if face_img.size == 0:
                        logging.warning("Extracted face image is empty, skipping.")
                        continue
                    timing_face_extraction = time.time() - start_time

                    # Preprocess for Encoder
                    start_time = time.time()
                    try:
                        preprocessed_face = self._preprocess_for_encoder(face_img)
                    except Exception as e:
                        logging.error(f"Error preprocessing face: {e}")
                        continue
                    timing_preprocessing = time.time() - start_time

                    # Encode the Face
                    start_time = time.time()
                    embedding = self.encoder(preprocessed_face)
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

                    # Attempt to find matching embeddings
                    matched = False
                    if self.hnsw_index.get_current_count() > 0:
                        labels, distances = self.hnsw_index.knn_query(embedding, k=1)
                        if labels.size > 0:
                            cosine_similarity = 1 - distances[0][0]
                            if cosine_similarity > self.similarity_threshold:
                                hnsw_id = labels[0][0]
                                self.update_label(hnsw_id, label)
                                logging.info(f"Updated label for hnsw_id {hnsw_id} to '{label}'.")
                                matched = True

                    if not matched:
                        logging.warning("No matching face found to update with the provided label.")

                # Save and annotate if needed
                if save_path:
                    if self.encryption_password:
                        _, buffer = cv2.imencode('.jpg', image)
                        image_bytes = buffer.tobytes()
                        self._encrypt_and_write(save_path, image_bytes)
                    else:
                        cv2.imwrite(save_path, image)
                        logging.info(f"Processed image saved to {save_path}")

                print("\n--- Image Processing Timings ---")
                for step, duration in timing.items():
                    print(f"{step}: {duration:.4f} seconds")
                total_time = sum(timing.values())
                print(f"Total Processing Time: {total_time:.4f} seconds\n")

            else:
                # If no label is provided, perform standard recognition and annotation
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

                    # Extract Face Image
                    start_time = time.time()
                    face_img = image[y:y + h, x:x + w]
                    if face_img.size == 0:
                        logging.warning("Extracted face image is empty, skipping.")
                        continue
                    timing_face_extraction = time.time() - start_time

                    # Preprocess for Encoder
                    start_time = time.time()
                    try:
                        preprocessed_face = self._preprocess_for_encoder(face_img)
                    except Exception as e:
                        logging.error(f"Error preprocessing face: {e}")
                        continue
                    timing_preprocessing = time.time() - start_time

                    # Encode the Face
                    start_time = time.time()
                    embedding = self.encoder(preprocessed_face)
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

                    # Attempt Recognition using HNSW Index
                    start_time = time.time()
                    label_found = None
                    confidence = 0.0
                    if self.hnsw_index.get_current_count() > 0:
                        labels, distances = self.hnsw_index.knn_query(embedding, k=1)
                        if labels.size > 0:
                            cosine_similarity = 1 - distances[0][0]
                            if cosine_similarity > self.similarity_threshold:
                                hnsw_id = labels[0][0]
                                label_found = self.hnsw_labels[hnsw_id]
                                confidence = float(cosine_similarity)
                    timing['Face Recognition'] = timing.get('Face Recognition', 0) + (time.time() - start_time)

                    # Handle Unknown Faces
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

                # Step 9: Flush New Embeddings
                start_time = time.time()
                if new_embeddings_to_add:
                    for label, emb in zip(new_labels_to_add, new_embeddings_to_add):
                        db_id = self._add_to_sqlite(label, emb)
                        if db_id != -1:
                            if self.hnsw_id_counter < 100000:
                                self.hnsw_index.add_items(emb, self.hnsw_id_counter)
                                self.hnsw_labels.append(label)
                                self.hnsw_db_ids.append(db_id)
                                logging.info(f"Added '{label}' to HNSWlib index with hnsw_id {self.hnsw_id_counter}.")
                                self.hnsw_id_counter += 1
                            else:
                                logging.warning("HNSWlib index has reached its maximum capacity, cannot add more embeddings.")
                    self._save_hnswlib_index()
                timing['Flushing Embeddings'] = time.time() - start_time

                # Step 10: Annotate the Image
                start_time = time.time()
                annotated_image = image.copy()
                if annotate:
                    for face in recognized_faces:
                        bbox = face['bbox']
                        label = face['label']
                        x, y, w, h = bbox
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        text = f"{label}"
                        cv2.putText(annotated_image, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                timing['Image Annotation'] = time.time() - start_time

                # Step 11: Display the Image (Optional)
                if self.show:
                    cv2.imshow('Face Recognition - Image', annotated_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # Step 12: Save the Annotated Image
                start_time = time.time()
                if save_path:
                    if self.encryption_password:
                        _, buffer = cv2.imencode('.jpg', annotated_image)
                        image_bytes = buffer.tobytes()
                        self._encrypt_and_write(save_path, image_bytes)
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
        """
        Resize the input frame to fit within the current screen resolution while maintaining the aspect ratio.

        Parameters:
        frame (numpy.ndarray): The input frame to be resized.

        Returns:
        numpy.ndarray: The resized frame.
        """
        # Get the screen resolution
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        # Get the original dimensions of the frame
        original_height, original_width = frame.shape[:2]

        # Calculate the aspect ratios
        frame_aspect_ratio = original_width / original_height
        screen_aspect_ratio = screen_width / screen_height

        # Determine the resizing dimensions
        if frame_aspect_ratio > screen_aspect_ratio:
            # Fit by width
            new_width = screen_width
            new_height = int(screen_width / frame_aspect_ratio)
        else:
            # Fit by height
            new_height = screen_height
            new_width = int(screen_height * frame_aspect_ratio)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return resized_frame

    def process_video(self, video_path: str, annotate: bool = True, save_path: str = None):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video file: {video_path}")
                return

            if save_path:
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
                    logging.info(f"Writing annotated video frames to temporary file: {temp_video_path}")
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
                    break

                if self.show:
                    frame = self.resize_frame_to_screen(frame)

                recognized_faces = self.recognize_faces(frame)

                annotated_frame = frame.copy()
                if annotate:
                    for face in recognized_faces:
                        bbox = face['bbox']
                        label = face['label']
                        confidence = face['confidence']
                        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (255, 0, 0), 2)
                        text = f"{label} ({confidence:.2f})"
                        cv2.putText(annotated_frame, text, (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if self.show:
                    cv2.imshow('Face Recognition - Video', annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("User requested to quit video processing.")
                        break

                if out:
                    out.write(annotated_frame)

            cap.release()
            if out:
                out.release()
                if self.encryption_password and temp_video_path:
                    try:
                        with open(temp_video_path, 'rb') as tmp_video:
                            video_bytes = tmp_video.read()
                        self._encrypt_and_write(save_path, video_bytes)
                        logging.info(f"Encrypted video saved to {save_path}")
                        os.remove(temp_video_path)
                        logging.info(f"Temporary video file {temp_video_path} removed.")
                    except Exception as e:
                        logging.error(f"Error during encryption of video: {e}")
                elif not self.encryption_password:
                    logging.info(f"Annotated video saved to {save_path}")

            if self.show:
                cv2.destroyAllWindows()

            end_time = time.time()
            elapsed_time = end_time - self.start_time if self.start_time else 0
            avg_detection_time = self.total_detection_time / self.frame_count if self.frame_count else 0
            avg_encoding_time = self.total_encoding_time / self.frame_count if self.frame_count else 0
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            logging.info(f"Processed {self.frame_count} frames in {elapsed_time:.2f} seconds.")
            logging.info(f"Average Detection Time: {avg_detection_time * 1000:.2f} ms/frame")
            logging.info(f"Average Encoding Time: {avg_encoding_time * 1000:.2f} ms/frame")
            logging.info(f"Pipeline FPS: {fps:.2f}")

        except Exception as e:
            logging.error(f"Error in process_video: {e}")

    def process_webcam(self, annotate: bool = True, save_path: str = None, duration: int = 0, name: str = None):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open webcam.")
            return

        if save_path:
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
                logging.info(f"Writing annotated webcam frames to temporary file: {temp_video_path}")
            else:
                out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    logging.error(f"Failed to open video writer for {save_path}.")
                    return
                logging.info(f"Saving annotated webcam video to {save_path}")
        else:
            out = None

        self.total_detection_time = 0.0
        self.total_encoding_time = 0.0
        self.frame_count = 0
        self.start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to grab frame from webcam.")
                    break

                recognized_faces = self.recognize_faces(frame, rename_label=name)

                annotated_frame = frame.copy()
                if annotate:
                    for face in recognized_faces:
                        bbox = face['bbox']
                        label = face['label']
                        confidence = face['confidence']
                        cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 2)
                        text = f"{label} ({confidence:.2f})"
                        cv2.putText(annotated_frame, text, (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if self.show:
                    cv2.imshow('Face Recognition - Webcam', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("User requested to quit webcam processing.")
                        break

                if out:
                    out.write(annotated_frame)

                if duration > 0:
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time >= duration:
                        logging.info(f"Duration of {duration} seconds reached. Stopping webcam.")
                        break

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Stopping webcam.")
        finally:
            cap.release()
            if out:
                out.release()
                if self.encryption_password and save_path and temp_video_path:
                    try:
                        with open(temp_video_path, 'rb') as tmp_video:
                            video_bytes = tmp_video.read()
                        self._encrypt_and_write(save_path, video_bytes)
                        logging.info(f"Encrypted webcam video saved to {save_path}")
                        os.remove(temp_video_path)
                        logging.info(f"Temporary video file {temp_video_path} removed.")
                    except Exception as e:
                        logging.error(f"Error during encryption of webcam video: {e}")
                elif not self.encryption_password and save_path:
                    logging.info(f"Annotated webcam video saved to {save_path}")

            if self.show:
                cv2.destroyAllWindows()

            end_time = time.time()
            elapsed_time = end_time - self.start_time if self.start_time else 0
            avg_detection_time = self.total_detection_time / self.frame_count if self.frame_count else 0
            avg_encoding_time = self.total_encoding_time / self.frame_count if self.frame_count else 0
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            logging.info(f"Processed {self.frame_count} frames in {elapsed_time:.2f} seconds.")
            logging.info(f"Average Detection Time: {avg_detection_time * 1000:.2f} ms/frame")
            logging.info(f"Average Encoding Time: {avg_encoding_time * 1000:.2f} ms/frame")
            logging.info(f"Pipeline FPS: {fps:.2f}")

    def close(self):
        try:
            self.save_database_to_sqlite()
            self._save_hnswlib_index()
            self._save_sqlite_database()
            logging.info("Closed FaceRecognition system and saved all data.")
        except Exception as e:
            logging.error(f"Error closing FaceRecognition system: {e}")

        if self.enable_logging and self.frame_count > 0:
            end_time = time.time()
            elapsed_time = end_time - self.start_time if self.start_time else 0
            avg_detection_time = self.total_detection_time / self.frame_count if self.frame_count else 0
            avg_encoding_time = self.total_encoding_time / self.frame_count if self.frame_count else 0
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            logging.info(f"Processed {self.frame_count} frames in {elapsed_time:.2f} seconds.")
            logging.info(f"Average Detection Time: {avg_detection_time * 1000:.2f} ms/frame")
            logging.info(f"Average Encoding Time: {avg_encoding_time * 1000:.2f} ms/frame")
            logging.info(f"Pipeline FPS: {fps:.2f}")


def set_single_core_affinity() -> None:
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([0])
    except (AttributeError, psutil.AccessDenied, NotImplementedError):
        print("Warning: Setting CPU affinity is not supported on this platform or access is denied.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition System with Encryption and Custom Filenames")
    parser.add_argument('--mode', type=str, default='image',
                        choices=['image', 'video', 'webcam'],
                        help='Mode of operation: image, video, or webcam')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image or video file')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the annotated output')
    parser.add_argument('--label', type=str, default=None,
                        help='Label/name for adding a new face (for image mode) or renaming recognized faces (webcam mode)')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging')
    parser.add_argument('--show', action='store_true',
                        help='Enable display of processed frames')
    parser.add_argument('--password', type=str, default=None,
                        help='Password for encrypting/decrypting files')
    parser.add_argument('--detector', type=str, default="mediapipe",
                        choices=["mediapipe", "yunet", "retinaface"],
                        help='Detector type: mediapipe, yunet, retinaface')
    parser.add_argument('--encoder', type=str, default=None,
                        choices=["128", "512"],
                        help='Encoder type: "128" for Facenet 128 and "512" for Facenet 512')
    parser.add_argument('--detection_interval', type=int, default=4,
                        help='Number of frames to skip for face detection, use 1 for images')
    parser.add_argument('--core', type=int, default=0,
                        help='0 use all cores, 1 use 1 core')

    annotate_group = parser.add_mutually_exclusive_group()
    annotate_group.add_argument('--annotate', dest='annotate', action='store_true',
                                help='Enable drawing bounding boxes and labels')
    annotate_group.add_argument('--no-annotate', dest='annotate', action='store_false',
                                help='Disable drawing bounding boxes and labels')
    parser.set_defaults(annotate=True)

    parser.add_argument('--hnsw_index_path', type=str, default=None,
                        help='Custom path for the HNSWlib index file')
    parser.add_argument('--hnsw_labels_path', type=str, default=None,
                        help='Custom path for the HNSWlib labels file')
    parser.add_argument('--hnsw_db_ids_path', type=str, default=None,
                        help='Custom path for the HNSWlib DB IDs file')
    parser.add_argument('--sqlite_db_path', type=str, default=None,
                        help='Custom path for the SQLite database file (unencrypted)')
    parser.add_argument('--sqlite_db_encrypted_path', type=str, default=None,
                        help='Custom path for the encrypted SQLite database file')

    # New argument to set the interested label
    parser.add_argument('--interested_label', type=str, default=None,
                        help='If set, only faces with this label will be recognized/maintained')

    args = parser.parse_args()

    face_recog = FaceRecognition(
        detector_type=args.detector,
        align=True,
        encoder_model_type=args.encoder,
        encoder_mode='cpu_optimized',
        similarity_threshold=0.74,
        enable_logging=args.log,
        show=args.show,
        unknown_trigger_count=1,
        detection_interval=1 if args.mode=="image" else args.detection_interval,
        encryption_password=args.password,
        hnsw_index_path=args.hnsw_index_path,
        hnsw_labels_path=args.hnsw_labels_path,
        hnsw_db_ids_path=args.hnsw_db_ids_path,
        sqlite_db_path=args.sqlite_db_path,
        sqlite_db_encrypted_path=args.sqlite_db_encrypted_path,
        interested_label=args.interested_label
    )

    if args.core:
        print(f"Use only 1 cpu core: {bool(args.core)}")
        set_single_core_affinity()

    if args.mode == 'image':
        if args.input is None:
            logging.error("Please provide the path to the input image using --input")
        else:
            face_recog.process_image(
                image_path=args.input,
                annotate=args.annotate,
                save_path=args.save,
                label=args.label
            )

    elif args.mode == 'video':
        if args.input is None:
            logging.error("Please provide the path to the input video using --input")
        else:
            face_recog.process_video(
                video_path=args.input,
                annotate=args.annotate,
                save_path=args.save
            )

    elif args.mode == 'webcam':
        face_recog.process_webcam(
            annotate=args.annotate,
            save_path=args.save,
            name=args.label
        )

    face_recog.close()
