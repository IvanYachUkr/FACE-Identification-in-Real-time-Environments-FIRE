# --------------------------------------------
# File: modules/database.py
# --------------------------------------------
import sqlite3
import os
import tempfile
import logging
import numpy as np

class DatabaseManager:
    def __init__(self, sqlite_db_path: str, sqlite_db_encrypted_path: str, encryptor, embedding_dim: int, detector_type: str, encoder_model_type: str):
        self.sqlite_db_path = sqlite_db_path
        self.sqlite_db_encrypted_path = sqlite_db_encrypted_path
        self.encryptor = encryptor
        self.embedding_dim = embedding_dim
        self.detector_type = detector_type
        self.encoder_model_type = encoder_model_type
        self.conn = None
        self.cursor = None
        self._initialize_sqlite()

    def _initialize_sqlite(self):
        if self.encryptor and self.sqlite_db_encrypted_path:
            self.conn = sqlite3.connect(':memory:')
            logging.info("Initialized in-memory SQLite database.")
            if os.path.exists(self.sqlite_db_encrypted_path):
                try:
                    with open(self.sqlite_db_encrypted_path, 'rb') as enc_file:
                        decrypted_data = self.encryptor.decrypt_data(enc_file.read())

                    # To load into memory, we must write to a temporary file first, then use backup.
                    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
                    with open(temp_db_path, 'wb') as tmp:
                        tmp.write(decrypted_data)

                    disk_conn = sqlite3.connect(temp_db_path)
                    disk_conn.backup(self.conn)
                    disk_conn.close()

                    os.close(temp_db_fd)
                    os.remove(temp_db_path)
                    logging.info("Decrypted and loaded existing database into memory.")
                except Exception as e:
                    logging.error(f"Failed to load encrypted database: {e}")
                    # Continue with an empty in-memory DB
        else:
            if self.sqlite_db_path is None:
                self.sqlite_db_path = f"face_embeddings_{self.detector_type}_{self.encoder_model_type}.db"
            self.conn = sqlite3.connect(self.sqlite_db_path)

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

    def save(self):
        if self.encryptor and self.sqlite_db_encrypted_path:
            # To save from memory, we must write to a temporary file first, then encrypt.
            temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
            disk_conn = sqlite3.connect(temp_db_path)
            self.conn.backup(disk_conn)
            disk_conn.close()

            with open(temp_db_path, 'rb') as tmp:
                db_data = tmp.read()

            self.encryptor.encrypt_and_write(self.sqlite_db_encrypted_path, db_data)
            self.conn.close()

            os.close(temp_db_fd)
            os.remove(temp_db_path)
            logging.info("Encrypted and saved the in-memory SQLite database.")
        elif not self.encryptor:
            self.conn.commit()
            self.conn.close()
            logging.info("Saved and closed the unencrypted SQLite database.")

    def add_face_embedding(self, label: str, embedding: np.ndarray) -> int:
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

    def load_all_embeddings(self):
        try:
            self.cursor.execute('SELECT id, label, embedding FROM faces')
            rows = self.cursor.fetchall()
            return rows
        except Exception as e:
            logging.error(f"Error loading embeddings from SQLite: {e}")
            return []
