# agents/doctor_agent.py
import pandas as pd
from typing import Dict, Any, Optional
from agents.base import BaseAgent, AgentResult
import google.generativeai as genai
from utils.config import Config
import logging

class DoctorAgent(BaseAgent):
    name = "doctor_escalation"

    def __init__(self, doctors_path: str, api_key: str = None):
        import os
        self.config = Config()
        
        # Load configuration
        self.medical_settings = self.config.settings.get('medical', {})
        model_settings = self.config.settings.get('model', {})
        
        try:
            # Load doctor roster
            self.doctors = pd.read_csv(doctors_path)
            
            # Configure AI model
            if api_key is None:
                api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            
            model_name = model_settings.get('model_name', 'gemini-2.5-flash')
            self.model = genai.GenerativeModel(model_name)
            
        except Exception as e:
            logging.error(f"Error initializing DoctorAgent: {str(e)}")
            raise

    def _select_doctor(self, specialty: Optional[str] = None, severity: str = "mild") -> Dict[str, Any]:
        """Select appropriate doctor based on specialty and severity"""
        try:
            if self.doctors.empty:
                return {
                    "name": "On-call Doctor",
                    "specialty": "General Medicine",
                    "tele_slot_iso8601": "N/A"
                }
            
            # Filter by specialty if provided
            available_docs = self.doctors
            if specialty:
                specialty_docs = self.doctors[
                    self.doctors["specialty"].str.contains(specialty, case=False, na=False)
                ]
                if not specialty_docs.empty:
                    available_docs = specialty_docs
            
            # Prioritize doctors based on severity
            if severity == "severe":
                # Get most experienced doctors first
                available_docs = available_docs.sort_values("experience_years", ascending=False)
            
            return available_docs.iloc[0].to_dict()
            
        except Exception as e:
            logging.error(f"Error selecting doctor: {str(e)}")
            return {
                "name": "On-call Doctor",
                "specialty": "General Medicine",
                "tele_slot_iso8601": "N/A"
            }

    def _generate_escalation_note(self, 
                                red_flags: list, 
                                severity: str, 
                                conditions: Dict[str, float]) -> str:
        """Generate escalation note using LLM"""
        try:
            top_condition = max(conditions.items(), key=lambda x: x[1])[0] if conditions else "unknown"
            
            prompt = f"""
            Summarize medical escalation recommendation (educational demo only).
            Context:
            - Severity Level: {severity}
            - Primary Concern: {top_condition}
            - Red Flags: {', '.join(red_flags) if red_flags else 'None'}
            - Condition Probabilities: {conditions}

            Keep response:
            1. Brief and clear
            2. Non-diagnostic
            3. Focused on immediate next steps
            4. Including appropriate medical disclaimers
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip() if hasattr(response, "text") else (
                "Medical attention recommended. This is an educational demo only."
            )
            
        except Exception as e:
            logging.error(f"Error generating escalation note: {str(e)}")
            return "Immediate medical consultation recommended. This is a demo only."

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        events = []
        try:
            # Extract inputs
            red_flags = payload.get("red_flags", [])
            severity = payload.get("severity_hint", "mild")
            conditions = payload.get("condition_probs", {})
            
            # Determine specialty based on top condition
            top_condition = max(conditions.items(), key=lambda x: x[1])[0] if conditions else None
            specialty_map = {
                "pneumonia": "Pulmonology",
                "covid_suspect": "Internal Medicine",
                "tuberculosis": "Pulmonology"
            }
            specialty = specialty_map.get(top_condition, "General Medicine")
            
            # Select doctor and generate note
            doctor = self._select_doctor(specialty, severity)
            note = self._generate_escalation_note(red_flags, severity, conditions)
            
            output = {
                "escalate": True,
                "doctor": doctor,
                "note": note,
                "severity": severity,
                "specialty_matched": specialty
            }
            
            events.append(self.event("escalation_initiated", {
                "doctor": doctor["name"],
                "severity": severity,
                "specialty": specialty
            }))
            
            return AgentResult(output, events)
            
        except Exception as e:
            logging.error(f"Error in DoctorAgent: {str(e)}")
            events.append(self.event("error", {"message": str(e)}))
            return AgentResult({
                "error": "Failed to process escalation",
                "severity": "severe"
            }, events)