import logging
import os
from datetime import datetime
import json
from typing import Dict, Any

class HealthcareLogger:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        try:
            # Ensure logs directory exists
            os.makedirs('logs', exist_ok=True)
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/healthcare.log'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger('HealthcareAssistant')
        except Exception as e:
            print(f"Warning: Could not setup file logging: {str(e)}")
            # Fallback to console only logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('HealthcareAssistant')
        
    def log_event(self, event_type: str, data: Dict[str, Any]):
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'type': event_type,
            'data': data
        }
        self.logger.info(json.dumps(log_entry))
        return log_entry

    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log an error message"""
        self.logger.error(f"{error_type}: {error_message}", extra={'context': context})
        
    def log_info(self, info_type: str, message: str, context: Dict[str, Any] = None):
        """Log an info message"""
        self.logger.info(f"{info_type}: {message}", extra={'context': context})