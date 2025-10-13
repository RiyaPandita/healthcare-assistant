# agents/base.py
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from utils.logging import HealthcareLogger
from utils.config import Config

@dataclass
class AgentResult:
    output: Dict[str, Any]
    events: List[Dict[str, Any]]  # Enhanced event logging

class BaseAgent(ABC):
    name: str = "base_agent"
    
    def __init__(self):
        self.logger = HealthcareLogger()
        self.config = Config()
    
    @abstractmethod
    def run(self, payload: Dict[str, Any]) -> AgentResult:
        """Run the agent's main logic"""
        raise NotImplementedError
    
    def event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized event log entry"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "agent": self.name,
            "data": data
        }
    
    def error(self, message: str):
        """Log an error message"""
        self.logger.error(f"{self.name} - {message}")
        
    def info(self, message: str):
        """Log an info message"""
        self.logger.info(f"{self.name} - {message}")
        
    def warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(f"{self.name} - {message}")