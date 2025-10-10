# agents/imaging_agent.py
from typing import Dict, Any, Tuple
from agents.base import BaseAgent, AgentResult
from models.imaging_stub import ChestXrayAnalyzer
import os
from utils.config import Config
import logging

class ImagingAgent(BaseAgent):
    name = "imaging"
    
    def __init__(self):
        self.config = Config()
        self.analyzer = ChestXrayAnalyzer()
        self.medical_settings = self.config.settings.get('medical', {})
        self.min_confidence = self.medical_settings.get('min_confidence_threshold', 0.3)
        self.severity_thresholds = self.medical_settings.get('severity_thresholds', {
            'mild': 0.3,
            'moderate': 0.6,
            'severe': 0.8
        })
        
    def validate_image(self, path: str) -> bool:
        """Validate the image file against security settings"""
        try:
            if not path or not os.path.exists(path):
                logging.error(f"Image file not found: {path}")
                return False
                
            security_settings = self.config.settings.get('security', {})
            allowed_types = security_settings.get('allowed_file_types', ['png', 'jpg', 'jpeg'])
            max_size = security_settings.get('max_file_size_mb', 10) * 1024 * 1024
            
            ext = os.path.splitext(path)[1].lower()[1:]
            size = os.path.getsize(path)
            
            if ext not in allowed_types:
                logging.error(f"Invalid file type: {ext}")
                return False
                
            if size > max_size:
                logging.error(f"File too large: {size} bytes")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating image: {str(e)}")
            return False
    
    def analyze_severity(self, probabilities: Dict[str, float]) -> Tuple[str, float]:
        """Determine severity level from condition probabilities"""
        max_prob = max(probabilities.values()) if probabilities else 0
        
        if max_prob >= self.severity_thresholds.get('severe', 0.8):
            return 'severe', max_prob
        elif max_prob >= self.severity_thresholds.get('moderate', 0.6):
            return 'moderate', max_prob
        elif max_prob >= self.severity_thresholds.get('mild', 0.3):
            return 'mild', max_prob
        return 'normal', max_prob

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        events = []
        try:
            path = payload.get("xray_path")
            
            # Validate image
            if not self.validate_image(path):
                events.append(self.event("validation_failed", {"path": path}))
                return AgentResult(
                    {"error": "Invalid or missing X-ray image"},
                    events
                )
            
            # Get predictions
            try:
                # Get predictions
                probabilities = self.analyzer.predict(path)
                
                # Determine severity from highest probability
                max_prob = max(probabilities.values())
                if max_prob >= self.severity_thresholds.get('severe', 0.8):
                    severity_level = 'severe'
                elif max_prob >= self.severity_thresholds.get('moderate', 0.6):
                    severity_level = 'moderate'
                else:
                    severity_level = 'mild'
                
                events.append(self.event("analysis_complete", {
                    "probabilities": probabilities,
                    "severity": severity_level,
                    "confidence": max_prob
                }))
                
                # Filter low confidence predictions
                filtered_probs = {
                    k: v for k, v in probabilities.items() 
                    if v >= self.min_confidence
                }
                
                if not filtered_probs:
                    filtered_probs = {"normal": 1.0}
                    severity_level = "normal"
                
                output = {
                    "condition_probs": filtered_probs,
                    "severity_hint": severity_level,
                    "confidence": max_prob
                }
                
                # Add auto-escalation if configured
                if severity_level == 'severe' or max_prob >= self.severity_thresholds.get('severe', 0.8):
                    output["requires_escalation"] = True
                    events.append(self.event("escalation_flagged", {
                        "reason": f"Severity: {severity_level}, Confidence: {max_prob:.2f}"
                    }))
                
                return AgentResult(output, events)
                
            except Exception as e:
                logging.error(f"Error analyzing image: {str(e)}")
                events.append(self.event("analysis_error", {"error": str(e)}))
                return AgentResult(
                    {"error": "Failed to analyze X-ray image"},
                    events
                )
            min_confidence = self.config.settings['medical']['min_confidence_threshold']
            max_prob = max(probs.values())
            
            if max_prob < min_confidence:
                events.append(self.event("low_confidence", {
                    "max_prob": max_prob,
                    "threshold": min_confidence
                }))
            
            return AgentResult(
                {"condition_probs": probs, "severity_hint": severity},
                events
            )
            
        except Exception as e:
            return self.handle_error(e, payload)