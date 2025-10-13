# agents/imaging_agent.py
from typing import Dict, Any, Tuple
from agents.base import BaseAgent, AgentResult
from models.imaging_stub import ChestXrayAnalyzer
from models.xray_processor import (
    extract_pdf_text, parse_report, estimate_lung_involvement_percentages,
    rale_bucket, map_to_simple_scale, generate_impression
)
import os
from PIL import Image
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
        
    def validate_file(self, path: str, allowed_types: list) -> bool:
        """Validate any file against security settings"""
        try:
            if not path or not os.path.exists(path):
                logging.error(f"File not found: {path}")
                return False
                
            security_settings = self.config.settings.get('security', {})
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
            logging.error(f"Error validating file: {str(e)}")
            return False
            
    def validate_image(self, path: str) -> bool:
        """Validate the image file against security settings"""
        return self.validate_file(path, ['png', 'jpg', 'jpeg'])
    
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
            xray_path = payload.get("xray_path")
            xray_report_path = payload.get("xray_report_path")
            prescription_path = payload.get("prescription_path")
            
            # Validate X-ray image
            if not self.validate_image(xray_path):
                events.append(self.event("validation_failed", {"path": xray_path}))
                return AgentResult(
                    {"error": "Invalid or missing X-ray image"},
                    events
                )
                
            # Validate report and prescription if provided
            if xray_report_path and not self.validate_file(xray_report_path, ['pdf']):
                events.append(self.event("validation_warning", {"path": xray_report_path, "type": "report"}))
                xray_report_path = None
                
            if prescription_path and not self.validate_file(prescription_path, ['pdf', 'jpg', 'jpeg', 'png']):
                events.append(self.event("validation_warning", {"path": prescription_path, "type": "prescription"}))
                prescription_path = None
            
            try:
                # Load and process X-ray image
                img = Image.open(xray_path).convert("RGB")
                events.append(self.event("image_loaded", {"path": xray_path}))

                # Process X-ray report if available
                xray_report_text = ""
                if xray_report_path and os.path.exists(xray_report_path):
                    with open(xray_report_path, 'rb') as f:
                        xray_report_text = extract_pdf_text(f.read())
                    events.append(self.event("report_processed", {"path": xray_report_path}))
                        
                # Process prescription if available
                prescription_text = ""
                if prescription_path and os.path.exists(prescription_path):
                    with open(prescription_path, 'rb') as f:
                        prescription_text = extract_pdf_text(f.read())
                    events.append(self.event("prescription_processed", {"path": prescription_path}))

                # Parse radiological findings from X-ray report
                rf = parse_report(xray_report_text) if xray_report_text else {
                    "ground_glass_opacity": False,
                    "consolidation": False,
                    "reticular_thickening": False,
                    "pleural_effusion": False,
                    "pneumothorax": False,
                    "laterality": None,
                    "zones_involved": None,
                    "distribution": None,
                }

                # Analyze image
                left_pct, right_pct = estimate_lung_involvement_percentages(img)
                left_score_img = rale_bucket(left_pct)
                right_score_img = rale_bucket(right_pct)

                # Apply laterality adjustments
                left_score = left_score_img
                right_score = right_score_img
                if rf.get("laterality") == "left":
                    right_score = max(0, right_score // 2)
                elif rf.get("laterality") == "right":
                    left_score = max(0, left_score // 2)

                total = left_score + right_score
                mapped, mapped_label = map_to_simple_scale(total)

                # Generate impression
                impression = generate_impression(rf, left_score, right_score, total, mapped, mapped_label)

                # Create comprehensive output
                output = {
                    "radiographic_findings": rf,
                    "severity_score": {
                        "left_lung": int(left_score),
                        "right_lung": int(right_score),
                        "total": int(total),
                        "mapped_scale": int(mapped),
                        "mapped_label": mapped_label,
                    },
                    "image_estimates": {
                        "left_pct": round(left_pct, 2),
                        "right_pct": round(right_pct, 2),
                        "left_score_image": int(left_score_img),
                        "right_score_image": int(right_score_img),
                    },
                    "impression": impression,
                }

                # Add severity event
                events.append(self.event("analysis_complete", {
                    "severity": mapped_label,
                    "score": total,
                    "findings": rf
                }))

                # Determine if severity warrants auto-escalation
                requires_escalation = mapped >= 4  # Moderate-Severe or worse
                if requires_escalation:
                    output["requires_escalation"] = True
                    events.append(self.event("escalation_flagged", {
                        "reason": f"Severity: {mapped_label}, Score: {total}/8"
                    }))

                return AgentResult(output, events)
                
            except Exception as e:
                logging.error(f"Error analyzing image: {str(e)}")
                events.append(self.event("analysis_error", {"error": str(e)}))
                return AgentResult(
                    {"error": "Failed to analyze X-ray image"},
                    events
                )
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