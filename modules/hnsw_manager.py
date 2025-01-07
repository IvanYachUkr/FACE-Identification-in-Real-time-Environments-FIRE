# --------------------------------------------
# File: modules/hnsw_manager.py
# --------------------------------------------
import logging
import os
import pickle
import tempfile
import hnswlib
import numpy as np

class HNSWManager:
    def __init__(self, embedding_dim: int, hnsw_index_path: str, hnsw_labels_path: str,
                 hnsw_db_ids_path: str, encryptor, hnsw_ef_construction: int = 200, hnsw_m: int = 16):
        self.embedding_dim = embedding_dim
        self.hnsw_index_path = hnsw_index_path
        self.hnsw_labels_path = hnsw_labels_path
        self.hnsw_db_ids_path = hnsw_db_ids_path
        self.encryptor = encryptor

        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.hnsw_labels = []
        self.hnsw_db_ids = []
        self.hnsw_id_counter = 0

        if self._files_exist([self.hnsw_index_path, self.hnsw_labels_path, self.hnsw_db_ids_path]):
            self._load_hnswlib_index()
            logging.info("Loaded existing HNSWlib index and mappings from disk.")
        else:
            self.hnsw_index.init_index(max_elements=100000, ef_construction=hnsw_ef_construction, M=hnsw_m)
            self.hnsw_index.set_ef(200)
            # self.hnsw_index.set_ef(50)
            logging.info("Initialized new HNSWlib index.")

    def _files_exist(self, paths):
        return all(os.path.exists(path) for path in paths)

    def _load_hnswlib_index(self):
        try:
            if self.encryptor:
                index_data = self.encryptor.read_and_decrypt(self.hnsw_index_path)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_index:
                    tmp_index.write(index_data)
                    tmp_index_path = tmp_index.name
                self.hnsw_index.load_index(tmp_index_path, max_elements=100000)
                os.remove(tmp_index_path)

                labels_data = self.encryptor.read_and_decrypt(self.hnsw_labels_path)
                with tempfile.NamedTemporaryFile(delete=False) as tmp_labels:
                    tmp_labels.write(labels_data)
                    tmp_labels_path = tmp_labels.name
                with open(tmp_labels_path, 'rb') as f:
                    self.hnsw_labels = pickle.load(f)
                os.remove(tmp_labels_path)

                db_ids_data = self.encryptor.read_and_decrypt(self.hnsw_db_ids_path)
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

    def save_hnswlib_index(self):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_index:
                self.hnsw_index.save_index(tmp_index.name)
                tmp_index_path = tmp_index.name
            with open(tmp_index_path, 'rb') as f:
                index_data = f.read()
            os.remove(tmp_index_path)

            if self.encryptor:
                self.encryptor.encrypt_and_write(self.hnsw_index_path, index_data)
            else:
                with open(self.hnsw_index_path, 'wb') as f:
                    f.write(index_data)
                logging.info(f"Saved HNSWlib index to {self.hnsw_index_path}.")

            labels_data = pickle.dumps(self.hnsw_labels)
            if self.encryptor:
                self.encryptor.encrypt_and_write(self.hnsw_labels_path, labels_data)
            else:
                with open(self.hnsw_labels_path, 'wb') as f:
                    f.write(labels_data)
                logging.info(f"Saved HNSWlib labels to {self.hnsw_labels_path}.")

            db_ids_data = pickle.dumps(self.hnsw_db_ids)
            if self.encryptor:
                self.encryptor.encrypt_and_write(self.hnsw_db_ids_path, db_ids_data)
            else:
                with open(self.hnsw_db_ids_path, 'wb') as f:
                    f.write(db_ids_data)
                logging.info(f"Saved HNSWlib DB IDs to {self.hnsw_db_ids_path}.")

            logging.info("Saved HNSWlib index and mappings to disk.")
        except Exception as e:
            logging.error(f"Error saving HNSWlib index: {e}")

    def load_embeddings_into_hnswlib(self, rows):
        try:
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

    def add_embedding(self, embedding: np.ndarray, label: str, db_id: int):
        if self.hnsw_id_counter < 100000:
            self.hnsw_index.add_items(embedding, self.hnsw_id_counter)
            self.hnsw_labels.append(label)
            self.hnsw_db_ids.append(db_id)
            logging.info(f"Added '{label}' to HNSWlib index with hnsw_id {self.hnsw_id_counter}.")
            self.hnsw_id_counter += 1
        else:
            logging.warning("HNSWlib index has reached its maximum capacity. Cannot add more embeddings.")

    def query(self, embedding: np.ndarray, k=1):
        if self.hnsw_index.get_current_count() > 0:
            labels, distances = self.hnsw_index.knn_query(embedding, k=k)
            return labels, distances
        return None, None

    # def update_label(self, hnsw_id: int, new_label: str, db_cursor, db_conn):
    #     try:
    #         if hnsw_id < 0 or hnsw_id >= len(self.hnsw_db_ids):
    #             logging.error("Invalid hnsw_id for update_label.")
    #             return
    #         db_id = self.hnsw_db_ids[hnsw_id]
    #         self.hnsw_labels[hnsw_id] = new_label
    #         db_cursor.execute('UPDATE faces SET label = ? WHERE id = ?', (new_label, db_id))
    #         db_conn.commit()
    #         logging.info(f"Updated label for hnsw_id {hnsw_id} (db_id {db_id}) to '{new_label}'.")
    #         self.save_hnswlib_index()
    #     except Exception as e:
    #         logging.error(f"Error updating label: {e}")


    def update_label(self, hnsw_id: int, new_label: str, db_cursor, db_conn, similarity_threshold: float = 0.7):
        """
        Updates the label for the given hnsw_id and also tries to rename all similar embeddings
        above a certain similarity threshold. If conflicting known labels are found, no group
        unification occurs. If only unknown or a single known label is present, unify them under new_label.
        """
        try:
            if hnsw_id < 0 or hnsw_id >= len(self.hnsw_db_ids):
                logging.error("Invalid hnsw_id for update_label.")
                return
            # Get reference embedding
            db_id = self.hnsw_db_ids[hnsw_id]
            reference_embedding = self._get_embedding_from_db_id(db_id, db_cursor)
            if reference_embedding is None:
                # Fallback: Just update current hnsw_id if embedding not found
                self._rename_single_entry(hnsw_id, new_label, db_cursor, db_conn)
                return

            # Find all similar embeddings
            similar_ids = self.find_similar_embeddings(reference_embedding, similarity_threshold, k=50)
            if not similar_ids:
                # If no similar, just rename this one
                self._rename_single_entry(hnsw_id, new_label, db_cursor, db_conn)
                return

            # Check labels of all similar embeddings
            current_labels = [self.hnsw_labels[sid] for sid in similar_ids]
            # Determine if there's a conflict
            known_labels = [lbl for lbl in current_labels if not lbl.lower().startswith("unknown")]
            if len(set(known_labels)) > 1:
                # Conflict: multiple different known labels
                logging.warning("Conflicting known labels found. Not unifying this group.")
                # Rename only the requested one
                self._rename_single_entry(hnsw_id, new_label, db_cursor, db_conn)
                return

            # If no conflict:
            # If there is at least one known label among them and it's different from new_label,
            # we prioritize that known label over the new_label
            if known_labels and known_labels[0] != new_label and not new_label.lower().startswith("unknown"):
                # Attempting to override a known label with a different known label
                # This might be undesired. If user wants to rename anyway, we can force it.
                # For now, let's unify under the chosen new_label since user triggered update_label.
                pass

            # Unify all similar embeddings under new_label
            self.unify_labels(similar_ids, new_label, db_cursor, db_conn)
        except Exception as e:
            logging.error(f"Error updating label: {e}")

    def _rename_single_entry(self, hnsw_id, new_label, db_cursor, db_conn):
        db_id = self.hnsw_db_ids[hnsw_id]
        db_cursor.execute('UPDATE faces SET label = ? WHERE id = ?', (new_label, db_id))
        db_conn.commit()
        self.hnsw_labels[hnsw_id] = new_label
        logging.info(f"Updated label for hnsw_id {hnsw_id} (db_id {db_id}) to '{new_label}'.")
        self.save_hnswlib_index()

    def unify_labels(self, hnsw_ids: list, new_label: str, db_cursor, db_conn):
        """
        Renames all embeddings corresponding to hnsw_ids to the same label in both DB and HNSW.
        """
        try:
            db_ids = [self.hnsw_db_ids[hid] for hid in hnsw_ids]
            # Update DB
            for db_id in db_ids:
                db_cursor.execute('UPDATE faces SET label = ? WHERE id = ?', (new_label, db_id))
            db_conn.commit()
            # Update in-memory labels
            for hid in hnsw_ids:
                self.hnsw_labels[hid] = new_label
            logging.info(f"Unified {len(hnsw_ids)} embeddings under label '{new_label}'.")
            self.save_hnswlib_index()
        except Exception as e:
            logging.error(f"Error unifying labels: {e}")

    def find_similar_embeddings(self, reference_embedding: np.ndarray, similarity_threshold: float, k: int = 50) -> list:
        """
        Given a reference embedding, find all embeddings in HNSW index that exceed the given similarity threshold.
        Similarity is computed as cosine similarity = 1 - distance.
        Returns a list of hnsw_ids that are similar to the reference embedding.
        """
        if self.hnsw_index.get_current_count() == 0:
            return []
        num_embeddings = self.hnsw_index.get_current_count()
        k_search = min(50, num_embeddings)
        labels, distances = self.hnsw_index.knn_query(reference_embedding, k=k_search)
        # distances are cosine distance, similarity = 1 - distance
        similar_ids = []
        for i in range(len(labels[0])):
            sim = 1 - distances[0][i]
            if sim >= similarity_threshold:
                similar_ids.append(labels[0][i])
        return similar_ids

    def _get_embedding_from_db_id(self, db_id: int, db_cursor):
        """
        Retrieve embedding from DB using db_id.
        """
        try:
            db_cursor.execute('SELECT embedding FROM faces WHERE id=?', (db_id,))
            row = db_cursor.fetchone()
            if row:
                embedding_blob = row[0]
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding
        except Exception as e:
            logging.error(f"Error retrieving embedding from DB: {e}")
        return None
