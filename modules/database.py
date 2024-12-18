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
        self.sqlite_temp_path = None
        self._initialize_sqlite()

    def _initialize_sqlite(self):
        if self.encryptor and self.sqlite_db_encrypted_path:
            if os.path.exists(self.sqlite_db_encrypted_path):
                sqlite_temp_fd, self.sqlite_temp_path = tempfile.mkstemp(suffix=".db")
                with open(self.sqlite_db_encrypted_path, 'rb') as enc_file:
                    decrypted_data = self.encryptor.decrypt_data(enc_file.read())
                with open(self.sqlite_temp_path, 'wb') as tmp:
                    tmp.write(decrypted_data)
                self.conn = sqlite3.connect(self.sqlite_temp_path)
                logging.info("Decrypted and connected to the temporary SQLite database.")
            else:
                sqlite_temp_fd, self.sqlite_temp_path = tempfile.mkstemp(suffix=".db")
                self.conn = sqlite3.connect(self.sqlite_temp_path)
                logging.info("Initialized a new temporary SQLite database.")
        else:
            if self.sqlite_db_path is None:
                self.sqlite_db_path = f"face_embeddings_{self.detector_type}_{self.encoder_model_type}.db"
            self.conn = sqlite3.connect(self.sqlite_db_path)
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

    def save(self):
        if self.encryptor and self.sqlite_temp_path:
            self.conn.commit()
            self.conn.close()
            with open(self.sqlite_temp_path, 'rb') as tmp:
                decrypted_data = tmp.read()
            self.encryptor.encrypt_and_write(self.sqlite_db_encrypted_path, decrypted_data)
            os.remove(self.sqlite_temp_path)
            self.sqlite_temp_path = None
            logging.info("Encrypted and saved the SQLite database.")
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
