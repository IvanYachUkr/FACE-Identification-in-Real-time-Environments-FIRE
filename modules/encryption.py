# --------------------------------------------
# File: modules/encryption.py
# --------------------------------------------
import os
import base64
import tempfile
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import logging

class Encryptor:
    def __init__(self, encryption_password: str):
        self.encryption_password = encryption_password
        self.backend = default_backend()
        self.iterations = 610000
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

    def encrypt_data(self, data: bytes) -> bytes:
        salt = os.urandom(16)
        key = self._derive_key(self.encryption_password, salt)
        f = Fernet(key)
        encrypted = f.encrypt(data)
        return salt + encrypted

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        salt = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        key = self._derive_key(self.encryption_password, salt)
        f = Fernet(key)
        return f.decrypt(encrypted)

    def encrypt_and_write(self, file_path: str, data: bytes):
        encrypted_data = self.encrypt_data(data)
        with open(file_path, 'wb') as f:
            f.write(encrypted_data)
        logging.info(f"Encrypted and saved data to {file_path}.")

    def read_and_decrypt(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = self.decrypt_data(encrypted_data)
        logging.info(f"Decrypted and loaded data from {file_path}.")
        return decrypted_data
