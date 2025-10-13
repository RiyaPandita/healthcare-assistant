import logging
import os
from datetime import datetime
import json
from typing import Dict, Any


class HealthcareLogger:
    def __init__(self):
        self.setup_logging()

    def setup_logging(self):
        """Setup logging with cloud-friendly configuration"""
        # Configure for both local and cloud environments
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Always set up console logging first
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )
        
        # Initialize logger
        self.logger = logging.getLogger('HealthcareAssistant')
        
        # Try to set up file logging if possible (for local development)
        if not self.is_cloud_environment():
            try:
                os.makedirs('logs', exist_ok=True)
                file_handler = logging.FileHandler('logs/healthcare.log')
                file_handler.setFormatter(logging.Formatter(log_format))
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"File logging not available: {str(e)}")

    def is_cloud_environment(self) -> bool:
        """Check if running in Streamlit Cloud or similar environment"""
        return (os.environ.get('STREAMLIT_CLOUD') == 'true' or 
                os.environ.get('IS_CLOUD_ENVIRONMENT') == 'true')

    def error(self, message: str):
        """Log an error message"""
        self.logger.error(message)

    def info(self, message: str):
        """Log an info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)

    def debug(self, message: str):
        """Log a debug message"""
        self.logger.debug(message)
    
    def error(self, message: str):
        """Log an error message"""
        self.logger.error(message)

    def info(self, message: str):
        """Log an info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a structured event"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'type': event_type,
            'data': data
        }
        self.logger.info(json.dumps(log_entry))
        return log_entry