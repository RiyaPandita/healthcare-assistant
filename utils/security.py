import re
from typing import Dict, Any, List
from cryptography.fernet import Fernet
import hashlib

class SecurityManager:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        self.pii_patterns = {
            'phone': r'\b\d{10}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'name': r'(?:Dr\.|Mr\.|Mrs\.|Ms\.) [A-Z][a-z]+ [A-Z][a-z]+'
        }

    def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data by removing potential security threats"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potential script injections
                value = re.sub(r'<script.*?>.*?</script>', '', value, flags=re.DOTALL)
                # Remove potential SQL injections
                value = re.sub(r'(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|INTO|CREATE)', 
                             lambda m: m.group(1).lower(), value)
            sanitized[key] = value
        return sanitized

    def deidentify_text(self, text: str) -> str:
        """Remove personally identifiable information from text"""
        for pii_type, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f"[REDACTED {pii_type.upper()}]", text)
        return text

    def validate_file_type(self, filename: str, allowed_types: List[str]) -> bool:
        """Validate file type against allowed types"""
        return any(filename.lower().endswith(f".{ext}") for ext in allowed_types)

    def compute_hash(self, data: str) -> str:
        """Compute SHA-256 hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()

    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode())

    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data).decode()