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
from facenet_gpu import FaceNetClient
from yunet_face_detector import detect_faces as detect_faces_yunet, extract_faces as extract_faces_yunet
from sort_UKF import Sort
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

class FaceRecognition:
    def __init__(self,
                 detector_type: str = 'yunet',
                 align: bool = True,
                 encoder_model_type: str = '128',
                 encoder_mode: str = 'gpu_optimized',
                 similarity_threshold: float = 0.5,
                 unknown_similarity_threshold: float = 0.6,  # New threshold for unknown tracking
                 unknown_trigger_count: int = 3,  # K value
                 enable_logging: bool = True,
                 show: bool = False,
                 detection_interval: int = 7,  # Added detection_interval parameter
                 hnsw_index_path: str = None,       # Modified to allow None
                 hnsw_labels_path: str = None,      # Modified to allow None
                 hnsw_db_ids_path: str = None,      # Modified to allow None
                 hnsw_ef_construction: int = 200,
                 hnsw_m: int = 16,
                 max_recent: int = 200,
                 max_new: int = 250,
                 sqlite_db_path: str = None,        # Added to allow custom SQLite path
                 sqlite_db_encrypted_path: str = None,  # Added to allow custom encrypted SQLite path
                 encryption_password: str = None):  # Added encryption_password parameter
        """
        Initialize the FaceRecognition system with specified detector and encoder.
        """
        self.encoder_model_type = encoder_model_type
        self.detector_type = detector_type.lower()
        self.align = align
        self.similarity_threshold = similarity_threshold
        self.unknown_similarity_threshold = unknown_similarity_threshold
        self.unknown_trigger_count = unknown_trigger_count
        self.enable_logging = enable_logging
        self.show = show
        self.detection_interval = detection_interval  # Added detection_interval
        self.frame_index = 0  # Added frame_index

        # Configure logging based on enable_logging flag
        if self.enable_logging:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize encryption if password is provided
        self.encryption_password = encryption_password
        if self.encryption_password:
            self._initialize_encryption()
            logging.info("Encryption is enabled for file operations.")
        else:
            logging.info("Encryption is disabled.")

        # Initialize the face detector
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

        # Initialize the FaceNet encoder
        self.encoder = FaceNetClient(model_type=encoder_model_type, mode=encoder_mode)
        logging.info(f"Initialized FaceNet-{self.encoder.output_shape} encoder in {encoder_mode} mode.")

        # Dynamically set file paths based on detector and encoder model names if not provided
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
            self.sqlite_db_path = None  # Not used when encrypted
        else:
            if sqlite_db_path is None:
                sqlite_db_path = f"face_embeddings_{self.detector_type}_{self.encoder_model_type}.db"
            self.sqlite_db_encrypted_path = None
            self.sqlite_db_path = sqlite_db_path

        # Initialize SQLite database
        self.sqlite_temp_path = None  # Temporary decrypted database path
        self._initialize_sqlite(sqlite_db_path=sqlite_db_path, sqlite_db_encrypted_path=sqlite_db_encrypted_path)

        # Initialize HNSWlib index
        self.hnsw_index_path = hnsw_index_path
        self.hnsw_labels_path = hnsw_labels_path
        self.hnsw_db_ids_path = hnsw_db_ids_path
        self.embedding_dim = self.encoder.output_shape
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.hnsw_labels = []  # Mapping from HNSWlib index ID to label
        self.hnsw_db_ids = []   # Mapping from HNSWlib index ID to database ID
        self.hnsw_id_counter = 0  # To assign unique IDs in HNSWlib

        # Attempt to load existing HNSWlib index and mappings
        if self._files_exist([self.hnsw_index_path, self.hnsw_labels_path, self.hnsw_db_ids_path]):
            self._load_hnswlib_index()
            logging.info("Loaded existing HNSWlib index and mappings from disk.")
        else:
            # Initialize the HNSWlib index if no existing index is found
            self.hnsw_index.init_index(max_elements=100000, ef_construction=hnsw_ef_construction, M=hnsw_m)
            self.hnsw_index.set_ef(50)  # ef should be > top_k
            logging.info("Initialized new HNSWlib index.")
            # Load embeddings into HNSWlib from SQLite
            self._load_embeddings_into_hnswlib()
            # Save the newly built index and mappings
            self._save_hnswlib_index()

        # Initialize Recent Embeddings for brute-force search
        self.recent_embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
        self.recent_labels = []
        self.max_recent = max_recent

        # Initialize buffer for new embeddings
        self.new_embeddings = []
        self.new_labels = []
        self.max_new = max_new

        # Initialize performance tracking
        self.total_detection_time = 0.0
        self.total_encoding_time = 0.0
        self.frame_count = 0
        self.start_time = None

        # Initialize mapping for "Unknown" faces to unique labels
        self.unknown_faces = {}  # {track_id: {'embeddings': [np.ndarray], 'count': int}}

        # Initialize the SORT tracker
        self.face_tracker = Sort(max_age=60, min_hits=unknown_trigger_count, iou_threshold=0.2)
        self.track_id_to_label = {}  # Mapping from track ID to face label

    def _initialize_encryption(self):
        """
        Initialize encryption components using the provided password.
        """
        # Encryption parameters
        self.backend = default_backend()
        self.iterations = 100000
        self.key_length = 32  # Length for Fernet
        # Note: Salt will be generated per file

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive a symmetric key from the password and salt using PBKDF2HMAC.

        Args:
            password (str): The encryption password.
            salt (bytes): A 16-byte salt.

        Returns:
            bytes: A base64-encoded symmetric key.
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.key_length,
            salt=salt,
            iterations=self.iterations,
            backend=self.backend
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data using Fernet symmetric encryption.

        Args:
            data (bytes): Data to encrypt.

        Returns:
            bytes: Encrypted data with salt prepended.
        """
        salt = os.urandom(16)
        key = self._derive_key(self.encryption_password, salt)
        f = Fernet(key)
        encrypted = f.encrypt(data)
        return salt + encrypted

    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using Fernet symmetric encryption.

        Args:
            encrypted_data (bytes): Encrypted data with salt prepended.

        Returns:
            bytes: Decrypted data.
        """
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        key = self._derive_key(self.encryption_password, salt)
        f = Fernet(key)
        return f.decrypt(encrypted)

    def _files_exist(self, paths: List[str]) -> bool:
        """
        Check if all files in the list exist.

        Args:
            paths (List[str]): List of file paths.

        Returns:
            bool: True if all files exist, False otherwise.
        """
        return all(os.path.exists(path) for path in paths)

    def _encrypt_and_write(self, file_path: str, data: bytes):
        """
        Encrypt data and write it to the specified file.

        Args:
            file_path (str): Path to the file.
            data (bytes): Data to encrypt and write.
        """
        encrypted_data = self._encrypt_data(data)
        with open(file_path, 'wb') as f:
            f.write(encrypted_data)
        logging.info(f"Encrypted and saved data to {file_path}.")

    def _read_and_decrypt(self, file_path: str) -> bytes:
        """
        Read encrypted data from a file and decrypt it.

        Args:
            file_path (str): Path to the encrypted file.

        Returns:
            bytes: Decrypted data.
        """
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = self._decrypt_data(encrypted_data)
        logging.info(f"Decrypted and loaded data from {file_path}.")
        return decrypted_data


    def _initialize_sqlite(self, sqlite_db_path: str, sqlite_db_encrypted_path: str):
        """
        Initialize the SQLite database to store embeddings and labels.
        Handles encryption if enabled.
        """
        if self.encryption_password:
            # Encrypted database path
            if os.path.exists(sqlite_db_encrypted_path):
                # Decrypt to a temporary file
                self.sqlite_temp_fd, self.sqlite_temp_path = tempfile.mkstemp(suffix=".db")
                with os.fdopen(self.sqlite_temp_fd, 'wb') as tmp:
                    decrypted_data = self._decrypt_data(open(sqlite_db_encrypted_path, 'rb').read())
                    tmp.write(decrypted_data)
                # Connect to the temporary decrypted database
                self.conn = sqlite3.connect(self.sqlite_temp_path)
                logging.info("Decrypted and connected to the temporary SQLite database.")
            else:
                # Create a new temporary database
                self.sqlite_temp_fd, self.sqlite_temp_path = tempfile.mkstemp(suffix=".db")
                self.conn = sqlite3.connect(self.sqlite_temp_path)
                logging.info("Initialized a new temporary SQLite database.")
        else:
            # Unencrypted database
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
        """
        Save and encrypt the SQLite database if encryption is enabled.
        """
        if self.encryption_password and self.sqlite_temp_path:
            self.conn.commit()
            self.conn.close()
            # Read the decrypted temporary database
            with open(self.sqlite_temp_path, 'rb') as tmp:
                decrypted_data = tmp.read()
            # Encrypt and save to the encrypted database path
            self._encrypt_and_write(self.sqlite_db_encrypted_path, decrypted_data)
            # Remove the temporary file
            os.remove(self.sqlite_temp_path)
            self.sqlite_temp_path = None
            logging.info("Encrypted and saved the SQLite database.")
        elif not self.encryption_password:
            self.conn.commit()
            self.conn.close()
            logging.info("Saved and closed the unencrypted SQLite database.")

    def _load_hnswlib_index(self):
        """
        Load the HNSWlib index and its mappings from disk with decryption if enabled.
        """
        try:
            if self.encryption_password:
                # Decrypt HNSWlib index
                index_data = self._read_and_decrypt(self.hnsw_index_path)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_index:
                    tmp_index.write(index_data)
                    tmp_index_path = tmp_index.name
                self.hnsw_index.load_index(tmp_index_path, max_elements=100000)
                os.remove(tmp_index_path)

                # Decrypt labels
                labels_data = self._read_and_decrypt(self.hnsw_labels_path)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_labels:
                    tmp_labels.write(labels_data)
                    tmp_labels_path = tmp_labels.name
                with open(tmp_labels_path, 'rb') as f:
                    self.hnsw_labels = pickle.load(f)
                os.remove(tmp_labels_path)

                # Decrypt db_ids
                db_ids_data = self._read_and_decrypt(self.hnsw_db_ids_path)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_db_ids:
                    tmp_db_ids.write(db_ids_data)
                    tmp_db_ids_path = tmp_db_ids.name
                with open(tmp_db_ids_path, 'rb') as f:
                    self.hnsw_db_ids = pickle.load(f)
                os.remove(tmp_db_ids_path)
            else:
                # Load HNSWlib index without encryption
                self.hnsw_index.load_index(self.hnsw_index_path, max_elements=100000)
                with open(self.hnsw_labels_path, 'rb') as f:
                    self.hnsw_labels = pickle.load(f)
                with open(self.hnsw_db_ids_path, 'rb') as f:
                    self.hnsw_db_ids = pickle.load(f)
            self.hnsw_id_counter = len(self.hnsw_labels)
            logging.info("Loaded HNSWlib index and mappings from disk.")
        except Exception as e:
            logging.error(f"Error loading HNSWlib index: {e}")
            # If loading fails, initialize a new index
            self.hnsw_index.init_index(max_elements=100000, ef_construction=200, M=16)
            self.hnsw_index.set_ef(50)
            self.hnsw_labels = []
            self.hnsw_db_ids = []
            self.hnsw_id_counter = 0
            logging.info("Initialized a new HNSWlib index due to loading failure.")

    def _save_hnswlib_index(self):
        """
        Save the HNSWlib index and its mappings to disk with encryption if enabled.
        """
        try:
            # Save HNSWlib index to bytes
            with tempfile.NamedTemporaryFile(delete=False) as tmp_index:
                self.hnsw_index.save_index(tmp_index.name)
                tmp_index_path = tmp_index.name
            with open(tmp_index_path, 'rb') as f:
                index_data = f.read()
            os.remove(tmp_index_path)

            # Encrypt and save HNSWlib index
            if self.encryption_password:
                self._encrypt_and_write(self.hnsw_index_path, index_data)
            else:
                with open(self.hnsw_index_path, 'wb') as f:
                    f.write(index_data)
                logging.info(f"Saved HNSWlib index to {self.hnsw_index_path}.")

            # Serialize and encrypt labels
            labels_data = pickle.dumps(self.hnsw_labels)
            if self.encryption_password:
                self._encrypt_and_write(self.hnsw_labels_path, labels_data)
            else:
                with open(self.hnsw_labels_path, 'wb') as f:
                    f.write(labels_data)
                logging.info(f"Saved HNSWlib labels to {self.hnsw_labels_path}.")

            # Serialize and encrypt db_ids
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
        """
        Load embeddings and labels from the SQLite database into the HNSWlib index.
        """
        try:
            self.cursor.execute('SELECT id, label, embedding FROM faces')
            rows = self.cursor.fetchall()
            for row in rows:
                db_id, label, embedding_blob = row
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                if embedding.shape[0] != self.embedding_dim:
                    logging.warning(f"Embedding size mismatch for label '{label}'. Expected {self.embedding_dim}, got {embedding.shape[0]}. Skipping.")
                    continue
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    logging.warning(f"Zero vector found for label '{label}'. Skipping.")
                    continue
                embedding = embedding / norm
                # Add to HNSWlib index
                self.hnsw_index.add_items(embedding, self.hnsw_id_counter)
                self.hnsw_labels.append(label)
                self.hnsw_db_ids.append(db_id)
                self.hnsw_id_counter += 1
            logging.info("Loaded embeddings into HNSWlib index from SQLite database.")
        except Exception as e:
            logging.error(f"Error loading embeddings into HNSWlib: {e}")

    def add_face(self, image: np.ndarray, label: str) -> bool:
        """
        Add a new face to the database with the given label.

        Args:
            image (numpy.ndarray): Input image in BGR format.
            label (str): Label/name for the face.

        Returns:
            bool: True if the face was added successfully, False otherwise.
        """
        try:
            # Detect and extract faces
            faces = self.extract_faces(image, align=self.align)
            if not faces:
                logging.warning("No faces detected to add.")
                return False

            # For simplicity, take the first detected face
            face_img = faces[0]

            # Preprocess the face image for the encoder
            preprocessed_face = self._preprocess_for_encoder(face_img)

            # Get the embedding
            start_encoding = time.time()
            embedding = self.encoder(preprocessed_face)
            encoding_time = time.time() - start_encoding
            self.total_encoding_time += encoding_time

            # Ensure embedding is a 1D vector
            if len(embedding.shape) > 1:
                embedding = embedding.squeeze()

            # Validate embedding dimensions
            if embedding.shape[0] != self.embedding_dim:
                logging.error(f"Invalid embedding size: expected {self.embedding_dim}, got {embedding.shape[0]}")
                return False

            norm = np.linalg.norm(embedding)
            if norm == 0:
                logging.error("Received zero vector from encoder. Skipping this face.")
                return False
            embedding = embedding / norm  # Normalize the embedding

            # Before adding, check if it is sufficiently different from existing embeddings
            if self.hnsw_index.get_current_count() > 0:
                labels, distances = self.hnsw_index.knn_query(embedding, k=1)
                if labels.size > 0:
                    cosine_similarity = 1 - (distances[0][0] ** 2) / 2
                    if cosine_similarity > self.similarity_threshold:
                        logging.info("Face is too similar to an existing face. Not adding.")
                        return False

            # Add to the new embeddings buffer
            self.new_embeddings.append(embedding)
            self.new_labels.append(label)
            logging.info(f"Added face for label '{label}' to the new embeddings buffer.")

            # If new embeddings exceed the maximum, flush them
            if len(self.new_embeddings) >= self.max_new:
                self._flush_new_embeddings()

            return True
        except Exception as e:
            logging.error(f"Error in add_face: {e}")
            return False

    def _add_to_sqlite(self, label: str, embedding: np.ndarray) -> int:
        """
        Add a face embedding and label to the SQLite database.

        Args:
            label (str): Label/name for the face.
            embedding (numpy.ndarray): Face embedding.

        Returns:
            int: The database ID of the newly added face.
        """
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
        """
        Flush the new embeddings buffer by adding them to the SQLite database and HNSWlib index.
        """
        try:
            for label, embedding in zip(self.new_labels, self.new_embeddings):
                db_id = self._add_to_sqlite(label, embedding)
                if db_id == -1:
                    continue  # Skip adding to HNSWlib if database insertion failed
                if self.hnsw_id_counter < 100000:  # Ensure max_elements is not exceeded
                    self.hnsw_index.add_items(embedding, self.hnsw_id_counter)
                    self.hnsw_labels.append(label)
                    self.hnsw_db_ids.append(db_id)
                    self.hnsw_id_counter += 1
                    logging.info(f"Added '{label}' to HNSWlib index with hnsw_id {self.hnsw_id_counter -1}.")
                else:
                    logging.warning("HNSWlib index has reached its maximum capacity. Cannot add more embeddings.")

            # Clear the new embeddings buffer after flushing
            self.new_embeddings = []
            self.new_labels = []

            # Save the updated index and mappings to disk
            self._save_hnswlib_index()
        except Exception as e:
            logging.error(f"Error flushing new embeddings: {e}")

    def save_database_to_sqlite(self):
        """
        Save the new embeddings and labels to the SQLite database and flush the HNSWlib index.
        """
        try:
            if self.new_embeddings:
                self._flush_new_embeddings()
            logging.info("Saved new embeddings to SQLite and HNSWlib index.")
        except Exception as e:
            logging.error(f"Error in save_database_to_sqlite: {e}")

    def _preprocess_for_encoder(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the face image for the FaceNet encoder.

        Args:
            face_img (numpy.ndarray): Extracted face image in RGB format.

        Returns:
            numpy.ndarray: Preprocessed face image suitable for the encoder.
        """
        # Resize to the encoder's expected input size
        resized_img = cv2.resize(face_img, self.encoder.input_shape, interpolation=cv2.INTER_AREA)

        # Convert to float32 and normalize if required
        img = resized_img.astype(np.float32)
        # Assuming the encoder expects pixel values in [0, 1]
        img /= 255.0

        # Ensure the image is in (H, W, C)
        if img.ndim == 3 and img.shape[2] == 3:
            pass  # Correct shape
        else:
            raise ValueError("Face image has incorrect shape for encoder.")

        # Add batch dimension to make it (1, H, W, C)
        img = np.expand_dims(img, axis=0)

        return img

    def recognize_faces(self, image: np.ndarray, rename_label: str = None) -> List[Dict[str, Any]]:
        """
        Recognize faces in the given image by comparing against the database.
        Uses tracking to reduce the frequency of face detection.

        Args:
            image (numpy.ndarray): Input image in BGR format.
            rename_label (str): If provided, matched embeddings will be renamed to this label.

        Returns:
            list: List of dictionaries containing face labels, confidence scores, and bounding boxes.
        """
        results = []

        # Start timing
        if self.start_time is None:
            self.start_time = time.time()

        self.frame_index += 1

        if ((self.frame_index % self.detection_interval == 0) or (self.frame_index == 1)):
            # Perform face detection
            start_detection = time.time()
            detected_faces = self.detect_faces(image)
            detection_time = time.time() - start_detection
            self.total_detection_time += detection_time

            # Convert detected faces to the format required by the tracker
            formatted_detections = []
            for bbox_dict in detected_faces:
                bbox = bbox_dict.get('bbox', [0, 0, 0, 0])
                confidence = bbox_dict.get('confidence', 1.0)
                formatted_detections.append({'bbox': bbox, 'confidence': confidence})

            # Update the tracker with detections
            tracks = self.face_tracker.update(formatted_detections)
        else:
            # No detections, just update trackers (they will predict their new positions)
            tracks = self.face_tracker.update([])

        # Build a set of active track IDs
        active_track_ids = set()
        for trk in tracks:
            active_track_ids.add(trk['id'])

        # Remove track IDs that are no longer active
        inactive_track_ids = set(self.track_id_to_label.keys()) - active_track_ids
        for track_id in inactive_track_ids:
            del self.track_id_to_label[track_id]
            if track_id in self.unknown_faces:
                del self.unknown_faces[track_id]

        for trk in tracks:
            track_id = trk['id']
            bbox = trk['bbox']
            age = trk['age']

            if track_id in self.track_id_to_label:
                label = self.track_id_to_label[track_id]
                confidence = 1.0  # Since we are tracking, we can set confidence to 1.0
            else:
                # New track, process face
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
                    # Preprocess the face image for the encoder
                    preprocessed_face = self._preprocess_for_encoder(face_img)
                except Exception as e:
                    logging.error(f"Error preprocessing face for track ID {track_id}: {e}")
                    continue

                # Get the embedding
                start_encoding = time.time()
                embedding = self.encoder(preprocessed_face)
                encoding_time = time.time() - start_encoding
                self.total_encoding_time += encoding_time

                # Ensure embedding is a 1D vector
                if len(embedding.shape) > 1:
                    embedding = embedding.squeeze()

                # Validate embedding dimensions
                if embedding.shape[0] != self.embedding_dim:
                    logging.error(f"Invalid embedding size: expected {self.embedding_dim}, got {embedding.shape[0]}")
                    continue

                norm = np.linalg.norm(embedding)
                if norm == 0:
                    logging.error("Received zero vector from encoder. Skipping this face.")
                    continue
                embedding = embedding / norm  # Normalize the embedding

                # Compare with recent embeddings first
                label = "Unknown"
                confidence = 0.0

                if self.recent_embeddings.shape[0] > 0:
                    similarities = np.dot(self.recent_embeddings, embedding.T).flatten()
                    max_similarity = np.max(similarities)
                    max_index = np.argmax(similarities)
                    if max_similarity > self.similarity_threshold:
                        label = self.recent_labels[max_index]
                        confidence = float(max_similarity)

                if label == "Unknown":
                    # Search in HNSWlib
                    if self.hnsw_index.get_current_count() > 0:
                        labels, distances = self.hnsw_index.knn_query(embedding, k=1)
                        if labels.size > 0:
                            cosine_similarity = 1 - (distances[0][0] ** 2) / 2
                            if cosine_similarity > self.similarity_threshold:
                                hnsw_id = labels[0][0]
                                label = self.hnsw_labels[hnsw_id]
                                confidence = float(cosine_similarity)

                                if rename_label:
                                    self.update_label(hnsw_id, rename_label)
                                    label = rename_label

                if label == "Unknown":
                    # Handle unknown embedding
                    label = self._handle_unknown_embedding(track_id, embedding, rename_label)
                    confidence = 1.0  # Or set to max_similarity or another metric

                # Map track ID to label
                self.track_id_to_label[track_id] = label

                # Add to recent embeddings
                self._add_to_recent_embeddings(embedding, label)

            # Append the result with standardized bounding box
            results.append({
                'label': self.track_id_to_label[track_id],
                'confidence': float(confidence),
                'bbox': bbox
            })

        # Increment frame count
        self.frame_count += 1

        return results

    def _handle_unknown_embedding(self, track_id: int, embedding: np.ndarray, rename_label: str = None) -> str:
        """
        Handle an unknown embedding by accumulating embeddings for the unknown face.
        Assigns a unique label after the unknown face has been detected unknown_trigger_count times.

        Args:
            track_id (int): The track ID associated with the embedding.
            embedding (numpy.ndarray): Normalized face embedding.
            rename_label (str): If provided, the embedding will be added with this label.

        Returns:
            str: Label assigned to the embedding.
        """
        if rename_label:
            # Assign the provided label directly
            self.new_embeddings.append(embedding)
            self.new_labels.append(rename_label)
            logging.info(f"Added face with label '{rename_label}' to the new embeddings buffer.")

            # Add to HNSWlib index
            if self.hnsw_id_counter < 100000:
                db_id = self._add_to_sqlite(rename_label, embedding)
                if db_id != -1:
                    self.hnsw_index.add_items(embedding, self.hnsw_id_counter)
                    self.hnsw_labels.append(rename_label)
                    self.hnsw_db_ids.append(db_id)
                    self.hnsw_id_counter += 1
                    logging.info(f"Added '{rename_label}' to HNSWlib index with hnsw_id {self.hnsw_id_counter -1}.")
            else:
                logging.warning("HNSWlib index has reached its maximum capacity. Cannot add more embeddings.")

            # If new embeddings exceed the maximum, flush them
            if len(self.new_embeddings) >= self.max_new:
                self._flush_new_embeddings()

            return rename_label
        else:
            # Accumulate embeddings for unknown faces
            if track_id not in self.unknown_faces:
                self.unknown_faces[track_id] = {'embeddings': [embedding], 'count': 1}
            else:
                self.unknown_faces[track_id]['embeddings'].append(embedding)
                self.unknown_faces[track_id]['count'] += 1

            if self.unknown_faces[track_id]['count'] >= self.unknown_trigger_count:
                # Assign a unique label
                unique_label = self._generate_unique_label()
                # Average the embeddings
                avg_embedding = np.mean(self.unknown_faces[track_id]['embeddings'], axis=0)
                # Before adding, check if it is sufficiently different from existing embeddings
                if self.hnsw_index.get_current_count() > 0:
                    labels, distances = self.hnsw_index.knn_query(avg_embedding, k=1)
                    if labels.size > 0:
                        cosine_similarity = 1 - (distances[0][0] ** 2) / 2
                        if cosine_similarity > self.similarity_threshold:
                            existing_label = self.hnsw_labels[labels[0][0]] if labels[0][0] < len(self.hnsw_labels) else "Unknown"
                            logging.info("Unknown face is too similar to an existing face. Not adding.")
                            return existing_label

                # Add to the new embeddings buffer
                self.new_embeddings.append(avg_embedding)
                self.new_labels.append(unique_label)
                logging.info(f"Added unknown face as '{unique_label}' to the new embeddings buffer.")

                # Add to HNSWlib index
                if self.hnsw_id_counter < 100000:
                    db_id = self._add_to_sqlite(unique_label, avg_embedding)
                    if db_id != -1:
                        self.hnsw_index.add_items(avg_embedding, self.hnsw_id_counter)
                        self.hnsw_labels.append(unique_label)
                        self.hnsw_db_ids.append(db_id)
                        self.hnsw_id_counter += 1
                        logging.info(f"Added '{unique_label}' to HNSWlib index with hnsw_id {self.hnsw_id_counter -1}.")
                else:
                    logging.warning("HNSWlib index has reached its maximum capacity. Cannot add more embeddings.")

                # If new embeddings exceed the maximum, flush them
                if len(self.new_embeddings) >= self.max_new:
                    self._flush_new_embeddings()

                # Remove from unknown_faces
                del self.unknown_faces[track_id]
                return unique_label
            else:
                return "Unknown"

    def _generate_unique_label(self) -> str:
        """
        Generate a unique label using uuid4.

        Returns:
            str: A unique label string.
        """
        unique_id = uuid.uuid4().hex[:8]  # Shorten UUID for readability
        unique_label = f"Unknown_{unique_id}"
        return unique_label

    def _add_to_recent_embeddings(self, embedding: np.ndarray, label: str):
        """
        Add an embedding and label to the recent embeddings buffer.

        Args:
            embedding (numpy.ndarray): Normalized face embedding.
            label (str): Label associated with the embedding.
        """
        self.recent_embeddings = np.vstack([self.recent_embeddings, embedding])
        self.recent_labels.append(label)

        # If recent embeddings exceed the maximum, remove the oldest
        if self.recent_embeddings.shape[0] > self.max_recent:
            self.recent_embeddings = self.recent_embeddings[1:]
            self.recent_labels.pop(0)

    def update_label(self, hnsw_id: int, new_label: str):
        """
        Update the label for a given HNSWlib index ID in both the HNSWlib index and the SQLite database.

        Args:
            hnsw_id (int): The HNSWlib index ID of the embedding.
            new_label (str): The new label to assign.
        """
        try:
            db_id = self.hnsw_db_ids[hnsw_id]
            # Update hnsw_labels
            self.hnsw_labels[hnsw_id] = new_label
            # Update SQLite
            self.cursor.execute('UPDATE faces SET label = ? WHERE id = ?', (new_label, db_id))
            self.conn.commit()
            logging.info(f"Updated label for hnsw_id {hnsw_id} (db_id {db_id}) to '{new_label}'.")
            # Save updated index and mappings
            self._save_hnswlib_index()
        except Exception as e:
            logging.error(f"Error updating label: {e}")

    def process_image(self, image_path: str, annotate: bool = True, save_path: str = None):
        """
        Perform face recognition on a single image.

        Args:
            image_path (str): Path to the input image.
            annotate (bool): Whether to draw bounding boxes and labels on the image.
            save_path (str): Path to save the annotated image. If None, not saved.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Image not found at path: {image_path}")
                return

            # Recognize faces (detection and recognition are now synchronized)
            recognized_faces = self.recognize_faces(image)

            print(recognized_faces)

            # Annotate image
            annotated_image = image.copy()
            if annotate:
                for face in recognized_faces:
                    bbox = face['bbox']
                    label = face['label']
                    confidence = face['confidence']

                    # Draw bounding box
                    x, y, w, h = bbox
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Put label and confidence
                    text = f"{label} ({confidence:.2f})"
                    cv2.putText(annotated_image, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if self.show:
                cv2.imshow('Face Recognition - Image', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_path:
                if self.encryption_password:
                    # Encode image to bytes
                    _, buffer = cv2.imencode('.jpg', annotated_image)
                    image_bytes = buffer.tobytes()
                    # Encrypt and save
                    self._encrypt_and_write(save_path, image_bytes)
                else:
                    cv2.imwrite(save_path, annotated_image)
                    logging.info(f"Annotated image saved to {save_path}")

        except Exception as e:
            logging.error(f"Error in process_image: {e}")

    def process_video(self, video_path: str, annotate: bool = True, save_path: str = None):
        """
        Perform face recognition on a video file.

        Args:
            video_path (str): Path to the input video.
            annotate (bool): Whether to draw bounding boxes and labels on the video frames.
            save_path (str): Path to save the annotated video. If None, not saved.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"Cannot open video file: {video_path}")
                return

            # Prepare video writer if save_path is provided
            if save_path:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30  # Default FPS
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if self.encryption_password:
                    # Write frames to a temporary video file
                    self.video_temp_fd, self.video_temp_path = tempfile.mkstemp(suffix=".avi")
                    out = cv2.VideoWriter(self.video_temp_path, fourcc, fps, (width, height))
                else:
                    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                    logging.info(f"Saving annotated video to {save_path}")
            else:
                out = None

            # Reset performance metrics
            self.total_detection_time = 0.0
            self.total_encoding_time = 0.0
            self.frame_count = 0
            self.start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Recognize faces (detection and recognition are now synchronized)
                recognized_faces = self.recognize_faces(frame)

                # Annotate frame
                annotated_frame = frame.copy()
                if annotate:
                    for face in recognized_faces:
                        bbox = face['bbox']
                        label = face['label']
                        confidence = face['confidence']

                        # Draw bounding box
                        x, y, w, h = bbox
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Put label and confidence
                        text = f"{label} ({confidence:.2f})"
                        cv2.putText(annotated_frame, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if self.show:
                    cv2.imshow('Face Recognition - Video', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if out:
                    if self.encryption_password:
                        # Encode frame to bytes
                        _, buffer = cv2.imencode('.avi', annotated_frame)
                        frame_bytes = buffer.tobytes()
                        # Write bytes to temporary file
                        with open(self.video_temp_path, 'ab') as tmp_video:
                            tmp_video.write(frame_bytes)
                    else:
                        out.write(annotated_frame)

            cap.release()
            if out:
                out.release()
                if self.encryption_password:
                    # Read temporary video file
                    with open(self.video_temp_path, 'rb') as tmp_video:
                        video_bytes = tmp_video.read()
                    # Encrypt and save to desired path
                    self._encrypt_and_write(save_path, video_bytes)
                    # Remove temporary file
                    os.remove(self.video_temp_path)
            if self.show:
                cv2.destroyAllWindows()

            # Calculate performance metrics
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
        """
        Perform face recognition in real-time using the webcam.

        Args:
            annotate (bool): Whether to draw bounding boxes and labels on the video frames.
            save_path (str): Path to save the annotated video. If None, not saved.
            duration (int): Duration in seconds to record. If 0, run indefinitely until 'Ctrl+C' is pressed.
            name (str): Name to assign to faces recognized during the webcam stream.
        """
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("Cannot open webcam.")
                return

            # Prepare video writer if save_path is provided
            if save_path:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30  # Default FPS
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if self.encryption_password:
                    # Write frames to a temporary video file
                    self.video_temp_fd, self.video_temp_path = tempfile.mkstemp(suffix=".avi")
                    out = cv2.VideoWriter(self.video_temp_path, fourcc, fps, (width, height))
                else:
                    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                    logging.info(f"Saving annotated webcam video to {save_path}")
            else:
                out = None

            # Reset performance metrics
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

                    # Recognize faces with the option to rename labels
                    recognized_faces = self.recognize_faces(frame, rename_label=name)

                    # Annotate frame
                    annotated_frame = frame.copy()
                    if annotate:
                        for face in recognized_faces:
                            bbox = face['bbox']
                            label = face['label']
                            confidence = face['confidence']

                            # Draw bounding box
                            x, y, w, h = bbox
                            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            # Put label and confidence
                            text = f"{label} ({confidence:.2f})"
                            cv2.putText(annotated_frame, text, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if self.show:
                        cv2.imshow('Face Recognition - Webcam', annotated_frame)
                        # Press 'q' to quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    if out:
                        if self.encryption_password:
                            # Encode frame to bytes
                            _, buffer = cv2.imencode('.avi', annotated_frame)
                            frame_bytes = buffer.tobytes()
                            # Write bytes to temporary file
                            with open(self.video_temp_path, 'ab') as tmp_video:
                                tmp_video.write(frame_bytes)
                        else:
                            out.write(annotated_frame)

                    # Handle duration
                    if 0 < duration < (time.time() - self.start_time):
                        logging.info(f"Duration of {duration} seconds reached. Stopping webcam.")
                        break

            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt received. Stopping webcam.")
            finally:
                cap.release()
                if out:
                    out.release()
                    if self.encryption_password:
                        # Read temporary video file
                        with open(self.video_temp_path, 'rb') as tmp_video:
                            video_bytes = tmp_video.read()
                        # Encrypt and save to desired path
                        self._encrypt_and_write(save_path, video_bytes)
                        # Remove temporary file
                        os.remove(self.video_temp_path)
                if self.show:
                    cv2.destroyAllWindows()

                # Calculate performance metrics
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
            logging.error(f"Error in process_webcam: {e}")

    def close(self):
        """
        Close the SQLite database connection, save new embeddings, and handle encryption.
        """
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

# Usage Example
if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face Recognition System with Encryption and Custom Filenames")
    parser.add_argument('--mode', type=str, default='image',
                        choices=['image', 'video', 'webcam'],
                        help='Mode of operation: image, video, or webcam')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image or video file')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the annotated output')
    parser.add_argument('--label', type=str, default=None,
                        help='Label/name for adding a new face (only for image mode) or renaming recognized faces in webcam mode')
    parser.add_argument('--log', action='store_true',
                        help='Enable detailed logging')
    parser.add_argument('--show', action='store_true',
                        help='Enable display of processed frames (default is disabled)')
    parser.add_argument('--password', type=str, default=None,
                        help='Password for encrypting/decrypting files')

    # Add mutually exclusive group for annotate
    annotate_group = parser.add_mutually_exclusive_group()
    annotate_group.add_argument('--annotate', dest='annotate', action='store_true',
                                help='Enable drawing of bounding boxes and labels on detected faces')
    annotate_group.add_argument('--no-annotate', dest='annotate', action='store_false',
                                help='Disable drawing of bounding boxes and labels on detected faces')
    parser.set_defaults(annotate=True)  # Default is to annotate

    # Added arguments for custom file paths
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

    args = parser.parse_args()

    # Initialize FaceRecognition
    face_recog = FaceRecognition(
        detector_type='yunet',  # 'mediapipe', 'retinaface', 'yunet', etc.
        align=True,
        encoder_model_type='128',
        encoder_mode='gpu_optimized',
        similarity_threshold=0.85,  # Adjusted threshold
        enable_logging=args.log,
        show=args.show,
        unknown_trigger_count=0,  # Set to 0 for image mode, adjust as needed
        encryption_password=args.password,  # Pass the encryption password
        hnsw_index_path=args.hnsw_index_path,
        hnsw_labels_path=args.hnsw_labels_path,
        hnsw_db_ids_path=args.hnsw_db_ids_path,
        sqlite_db_path=args.sqlite_db_path,
        sqlite_db_encrypted_path=args.sqlite_db_encrypted_path
    )

    if args.mode == 'image':
        if args.input is None:
            logging.error("Please provide the path to the input image using --input")
        elif args.label is not None:
            # Add a new face
            image = cv2.imread(args.input)
            if image is not None:
                success = face_recog.add_face(image, args.label)
                if success:
                    logging.info(f"Successfully added face with label '{args.label}'.")
                else:
                    logging.warning("Failed to add face.")
            else:
                logging.error(f"Failed to load image from {args.input}")
        else:
            # Recognize faces in the image
            face_recog.process_image(args.input, annotate=args.annotate, save_path=args.save)

    elif args.mode == 'video':
        if args.input is None:
            logging.error("Please provide the path to the input video using --input")
        else:
            face_recog.process_video(args.input, annotate=args.annotate, save_path=args.save)

    elif args.mode == 'webcam':
        face_recog.process_webcam(annotate=args.annotate, save_path=args.save, name=args.label)

    # Close the FaceRecognition system
    face_recog.close()
