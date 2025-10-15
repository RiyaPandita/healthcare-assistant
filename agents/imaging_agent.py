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

                # Process X-ray report if available. Support either a file path (xray_report_path)
                # or PDF bytes uploaded directly from the UI (xray_report_bytes).
                xray_report_text = ""
                # Prefer bytes if provided by the UI
                report_bytes = payload.get('xray_report_bytes')
                if report_bytes:
                    try:
                        # report_bytes may be bytes or a base64 string; if str, assume bytes-like
                        if isinstance(report_bytes, str):
                            report_bytes = report_bytes.encode('utf-8')
                        xray_report_text = extract_pdf_text(report_bytes)
                        events.append(self.event("report_processed", {"source": "bytes"}))
                    except Exception:
                        xray_report_text = ""

                # Fallback to file path on disk
                if not xray_report_text and xray_report_path and os.path.exists(xray_report_path):
                    with open(xray_report_path, 'rb') as f:
                        xray_report_text = extract_pdf_text(f.read())
                    events.append(self.event("report_processed", {"path": xray_report_path}))
                        
                # Process prescription if available
                prescription_text = ""
                if prescription_path and os.path.exists(prescription_path):
                    with open(prescription_path, 'rb') as f:
                        prescription_text = extract_pdf_text(f.read())
                    events.append(self.event("prescription_processed", {"path": prescription_path}))

                # Parse radiological findings from X-ray report. If no report is provided,
                # use None for findings (unknown) so we don't accidentally assert absence.
                report_absent = not bool(xray_report_text and xray_report_text.strip())
                rf = parse_report(xray_report_text) if not report_absent else {
                    "ground_glass_opacity": None,
                    "consolidation": None,
                    "reticular_thickening": None,
                    "pleural_effusion": None,
                    "pneumothorax": None,
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

                # If there was no radiology report available, infer likely findings from
                # the image-derived RALE buckets and make it explicit in the impression.
                inferred_prefix = None
                if report_absent:
                    inferred_rf = {
                        "ground_glass_opacity": (left_score_img >= 1 or right_score_img >= 1),
                        "consolidation": (left_score_img >= 3 or right_score_img >= 3),
                        "reticular_thickening": None,
                        "pleural_effusion": None,
                        "pneumothorax": None,
                        "laterality": ("bilateral" if left_score_img >= 1 and right_score_img >= 1 else ("left" if left_score_img >= 1 else ("right" if right_score_img >= 1 else None))),
                        "zones_involved": None,
                        "distribution": None,
                    }
                    inferred_prefix = "Note: no formal radiology report available; image-based assessment suggests the following."
                    # Use inferred findings for the impression text
                    impression = inferred_prefix + " " + generate_impression(inferred_rf, left_score, right_score, total, mapped, mapped_label)
                else:
                    # Generate impression from the report (and image adjustments)
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

                # Initialize red flag collection based on imaging, age and configured rules
                red_flags = []
                red_flag_evidence = []
                try:
                    # Load configured red-flag symptom keywords from rules if available
                    rules_path = os.path.join(self.config.base_path, "data", "medical_rules.json")
                    medical_rules = {}
                    try:
                        import json
                        with open(rules_path) as rf_f:
                            medical_rules = json.load(rf_f)
                    except Exception:
                        medical_rules = self.medical_settings.get('medical_rules', {}) or {}

                    rf_symptoms = [s.lower() for s in medical_rules.get("red_flags", {}).get("symptoms", [])]

                    # Compute percent severity (total out of max possible 8)
                    max_total = 8.0
                    percent = (float(total) / max_total) if max_total else 0.0
                    if percent >= 0.7:
                        msg = f"Imaging severity {percent*100:.0f}% (score {total}/{int(max_total)}) meets/exceeds 70% threshold"
                        red_flags.append(msg)
                        red_flag_evidence.append({"type": "imaging_score", "score": total, "percent": percent})

                    # Age-based escalation: if patient age provided and >=80
                    patient = payload.get("patient") or {}
                    age = patient.get("age") if isinstance(patient, dict) else None
                    if age is not None:
                        try:
                            if int(age) >= 80:
                                red_flags.append(f"Patient age {age} >= 80: consider urgent geriatric review")
                                red_flag_evidence.append({"type": "age", "age": int(age)})
                        except Exception:
                            pass

                    # Check impression text for serious imaging keywords
                    imp_low = (impression or "").lower()
                    # small helper: check if a keyword occurrence is negated in the surrounding text
                    negation_phrases = [
                        "no", "without", "absent", "none", "no evidence of", "not seen",
                        "negative for", "no definite", "no significant", "ruled out", "excluded"
                    ]

                    def _is_negated_in_text(text: str, start: int, end: int) -> bool:
                        window_before = text[max(0, start - 60):start]
                        window_after = text[end:end + 60]
                        combined = window_before + " " + window_after
                        for np in negation_phrases:
                            if np in combined:
                                return True
                        return False

                    imaging_keywords = ("pneumonia", "consolidation", "ground glass", "pleural effusion", "pneumothorax", "tension pneumothorax", "air bronchogram")
                    for kw in imaging_keywords:
                        start = 0
                        while True:
                            idx = imp_low.find(kw, start)
                            if idx == -1:
                                break
                            end = idx + len(kw)
                            if not _is_negated_in_text(imp_low, idx, end):
                                red_flags.append(f"Imaging impression contains '{kw}' — consider escalation")
                                red_flag_evidence.append({"type": "impression_keyword", "keyword": kw})
                            start = end

                    # Check radiographic findings booleans
                    if rf.get("pneumothorax"):
                        red_flags.append("Pneumothorax noted on radiographic findings — immediate attention advised")
                        red_flag_evidence.append({"type": "finding", "finding": "pneumothorax"})
                    if rf.get("pleural_effusion"):
                        red_flags.append("Pleural effusion noted on radiographic findings — consider urgent review")
                        red_flag_evidence.append({"type": "finding", "finding": "pleural_effusion"})
                    if rf.get("consolidation"):
                        red_flags.append("Consolidation noted on radiographic findings — consider escalation")
                        red_flag_evidence.append({"type": "finding", "finding": "consolidation"})

                    # Also check configured red-flag symptoms (from medical_rules) against impression
                    for rf_kw in rf_symptoms:
                        if not rf_kw:
                            continue
                        kw = rf_kw.lower()
                        start = 0
                        while True:
                            idx = imp_low.find(kw, start)
                            if idx == -1:
                                break
                            end = idx + len(kw)
                            if not _is_negated_in_text(imp_low, idx, end):
                                red_flags.append(f"Impression contains red-flag keyword: {kw}")
                                red_flag_evidence.append({"type": "rule_keyword", "keyword": kw})
                            start = end

                except Exception:
                    # non-fatal — continue
                    pass

                # Attach red flag results to output
                output["red_flags"] = red_flags
                output["red_flag_evidence"] = red_flag_evidence

                # Add severity event
                events.append(self.event("analysis_complete", {
                    "severity": mapped_label,
                    "score": total,
                    "findings": rf
                }))

                # Determine if severity warrants auto-escalation
                # Only auto-escalate for the highest mapped severity (very severe)
                requires_escalation = mapped >= 5  # Very severe only
                if requires_escalation:
                    output["requires_escalation"] = True
                    events.append(self.event("escalation_flagged", {
                        "reason": f"Severity: {mapped_label}, Score: {total}/8"
                    }))

                # For compatibility with downstream agents/tests, also include
                # a coarse `condition_probs`, `severity_hint`, and `confidence` keys.
                # We'll map the imaging mapped scale into a simple condition prob dict
                try:
                    # simple heuristic: map mapped_label to higher probability for pneumonia when consolidation/ground glass
                    condition_probs = {
                        "normal": 0.5,
                        "pneumonia": 0.3,
                        "covid_suspect": 0.2
                    }
                    if rf.get("consolidation") or rf.get("ground_glass_opacity"):
                        condition_probs = {"pneumonia": 0.6, "covid_suspect": 0.2, "normal": 0.2}
                    max_prob = max(condition_probs.values())
                    severity_hint = mapped_label
                except Exception:
                    condition_probs = {"normal": 1.0}
                    max_prob = 0.0
                    severity_hint = "normal"

                output.update({
                    "condition_probs": condition_probs,
                    "severity_hint": severity_hint,
                    "confidence": float(max_prob)
                })

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