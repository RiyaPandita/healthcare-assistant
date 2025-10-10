# agents/ingestion_agent.py
import os
import re
from typing import Dict, Any, Optional
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from agents.base import BaseAgent, AgentResult
from utils.security import SecurityManager
from utils.logging import HealthcareLogger
from utils.config import Config

class IngestionAgent(BaseAgent):
    name = "ingestion"

    def __init__(self):
        self.security = SecurityManager()
        self.logger = HealthcareLogger()
        self.config = Config()

    def _validate_xray(self, xray_path: str) -> bool:
        """Validate X-ray image file"""
        try:
            if not xray_path or not os.path.exists(xray_path):
                return False
                
            # Check file type
            allowed_types = self.config.settings.get('security', {}).get('allowed_file_types', ['png', 'jpg', 'jpeg'])
            if not self.security.validate_file_type(xray_path, allowed_types):
                self.logger.log_error("XRAY_VALIDATION", f"Invalid file type for {xray_path}")
                return False
                
            # Check file size
            max_size = self.config.settings.get('security', {}).get('max_file_size_mb', 10) * 1024 * 1024
            if os.path.getsize(xray_path) > max_size:
                self.logger.log_error("XRAY_VALIDATION", f"File too large: {xray_path}")
                return False
                
            # Try opening the image to verify it's valid
            with Image.open(xray_path) as img:
                img.verify()
            return True
            
        except Exception as e:
            self.logger.log_error("XRAY_VALIDATION", str(e))
            return False

    def _extract_pdf_text(self, pdf_path: Optional[str]) -> str:
        """Extract text from PDF using OCR and text extraction"""
        if not pdf_path or not os.path.exists(pdf_path):
            return ""
        try:
            # First try direct text extraction
            text = extract_text(pdf_path) or ""
            if not text.strip():
                # If no text found, try OCR
                pages = convert_from_path(pdf_path)
                for page in pages:
                    text += pytesseract.image_to_string(page)
            return self.security.deidentify_text(text[:4000])
        except Exception as e:
            self.logger.log_error("PDF_ERROR", str(e))
            return ""

    def validate_files(self, xray_path: Optional[str], pdf_path: Optional[str]) -> bool:
        """Validate uploaded files"""
        try:
            allowed_types = self.config.settings.get('security', {}).get('allowed_file_types', 
                ['png', 'jpg', 'jpeg', 'pdf'])
            max_size = self.config.settings.get('security', {}).get('max_file_size_mb', 10) * 1024 * 1024
            
            for path in [xray_path, pdf_path]:
                if path and os.path.exists(path):
                    if not self.security.validate_file_type(path, allowed_types):
                        self.logger.log_error("FILE_VALIDATION", f"Invalid file type: {path}")
                        return False
                    if os.path.getsize(path) > max_size:
                        self.logger.log_error("FILE_VALIDATION", f"File too large: {path}")
                        return False
            return True
        except Exception as e:
            self.logger.log_error("FILE_VALIDATION", str(e))
            return False

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        events = []
        try:
            # Debug logging
            self.logger.log_info("INGESTION_START", f"Received payload: {str(payload)}")
            
            # Validate and sanitize input
            sanitized_payload = self.security.sanitize_input(payload)
            xray_path = sanitized_payload.get("xray_path")
            pdf_path = sanitized_payload.get("pdf_path")
            
            # Log paths for debugging
            self.logger.log_info("INGESTION_PATHS", 
                f"X-ray path: {xray_path}, PDF path: {pdf_path}")
            
            # Check if xray file exists
            if xray_path and os.path.exists(xray_path):
                self.logger.log_info("INGESTION_FILE", 
                    f"X-ray file exists: {os.path.getsize(xray_path)} bytes")
            else:
                self.logger.log_error("INGESTION_FILE", 
                    f"X-ray file missing or invalid: {xray_path}")
            
            # Early validation of X-ray
            if not self._validate_xray(xray_path):
                self.logger.log_error("INGESTION_VALIDATION", 
                    f"X-ray validation failed for {xray_path}")
                events.append(self.event("xray_validation_failed", 
                    {"error": "Invalid or missing X-ray file"}))
                return AgentResult({"error": "Invalid X-ray file"}, events)
            
            # Extract text from PDF if present
            pdf_text = self._extract_pdf_text(pdf_path) if pdf_path else ""
            events.append(self.event("pdf_processed", {"chars": len(pdf_text)}))
            
            # Process patient data
            patient_data = sanitized_payload.get("patient", {})
            notes = sanitized_payload.get("notes", "")
            
            # Build output
            output = {
                "xray_path": xray_path,
                "pdf_text": pdf_text,
                "notes": self.security.deidentify_text(notes),
                "patient": {
                    "age": patient_data.get("age"),
                    "allergies": patient_data.get("allergies", [])
                }
            }
            
            events.append(self.event("ingestion_complete", {
                "xray": bool(xray_path),
                "pdf": bool(pdf_path),
                "notes_length": len(notes)
            }))
            
            return AgentResult(output, events)
            
        except Exception as e:
            self.logger.log_error("INGESTION_ERROR", str(e))
            events.append(self.event("error", {"message": str(e)}))
            return AgentResult({"error": "Failed to process input files"}, events)

        output = {
            "patient": {
                "age": patient.get("age"),
                "allergies": patient.get("allergies", [])
            },
            "xray_path": xray_path,
            "notes": notes,
            "pdf_text": pdf_text
        }
        events.append(self.event("deidentify", {"masked": True}))
        return AgentResult(output, events)