# agents/therapy_agent.py
import pandas as pd
from typing import Dict, Any, List, Optional
from agents.base import BaseAgent, AgentResult
import google.generativeai as genai
from utils.config import Config
import logging

class TherapyAgent(BaseAgent):
    name = "therapy"

    def __init__(self, meds_path: str, interactions_path: str, api_key: str = None):
        import os
        self.config = Config()
        
        # Load data files
        try:
            self.meds = pd.read_csv(meds_path)
            self.interactions = pd.read_csv(interactions_path)
        except Exception as e:
            logging.error(f"Failed to load therapy data files: {str(e)}")
            raise
            
        # Configure AI model
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Configure with explicit credentials
        genai.configure(
            api_key=api_key,
            transport="rest"  # Force REST API instead of gRPC
        )
        
        model_settings = self.config.settings.get('model', {})
        model_name = model_settings.get('model_name', 'gemini-pro')
        self.model = genai.GenerativeModel(model_name)
        
        # Load medical settings
        self.medical_settings = self.config.settings.get('medical', {})
        self.min_confidence = self.medical_settings.get('min_confidence_threshold', 0.3)
        self.severity_thresholds = self.medical_settings.get('severity_thresholds', {
            'mild': 0.3,
            'moderate': 0.6,
            'severe': 0.8
        })

    def _get_severity_level(self, probability: float) -> str:
        """Determine severity level based on configured thresholds."""
        if probability >= self.severity_thresholds.get('severe', 0.8):
            return 'severe'
        elif probability >= self.severity_thresholds.get('moderate', 0.6):
            return 'moderate'
        elif probability >= self.severity_thresholds.get('mild', 0.3):
            return 'mild'
        return 'normal'

    def _filter_otc(self, indication: str, age: int, allergies: List[str]) -> pd.DataFrame:
        """Filter OTC medications based on indication, age, and allergies."""
        try:
            df = self.meds[self.meds["indication"].str.contains(indication, case=False, na=False)]
            df = df[df["age_min"] <= age]
            
            def check_allergies(row):
                if not allergies:
                    return True
                contra_keywords = str(row.get("contra_allergy_keywords", "")).lower().split(";")
                return not any(allergy.lower() in contra_keywords for allergy in allergies)
                
            return df[df.apply(check_allergies, axis=1)]
        except Exception as e:
            logging.error(f"Error filtering OTC medications: {str(e)}")
            return pd.DataFrame()

    def _interaction_notes(self, chosen_names: List[str]) -> List[str]:
        """Check for drug interactions between chosen medications."""
        notes = []
        try:
            for _, row in self.interactions.iterrows():
                if row["drug_a"] in chosen_names and row["drug_b"] in chosen_names:
                    notes.append(f"{row['drug_a']} + {row['drug_b']}: {row['level']} - {row['note']}")
        except Exception as e:
            logging.error(f"Error checking drug interactions: {str(e)}")
        return notes

    def _llm_advice(self, summary: str, meds: List[Dict[str, Any]], red_flags: List[str], 
                   severity: str) -> str:
        """Generate advice using LLM with enhanced context."""
        prompt = f"""
You are assisting in a non-clinical demo. Provide cautious, non-prescriptive OTC guidance.
Summary: {summary}
Severity Level: {severity}
OTC candidates: {meds}
Red flags: {red_flags}

Constraints: 
- Educational demo, not medical advice
- Avoid prescription-only claims
- If red flags present or severity is high, advise immediate care
- Focus on symptom management and safety
- Include clear disclaimers
Return a concise advice paragraph with appropriate cautions based on severity.
"""
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip() if hasattr(resp, "text") else "Not medical advice. If unsure, seek immediate care."
        except Exception as e:
            logging.error(f"Error generating LLM advice: {str(e)}")
            return "Unable to generate advice. Please consult a healthcare professional."

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        events = []
        try:
            # Extract patient information
            patient = payload.get("patient", {})
            age = patient.get("age", 18)
            allergies = patient.get("allergies", [])
            probs = payload.get("condition_probs", {})
            notes = f"{payload.get('notes', '')} {payload.get('pdf_text', '')}".strip()

            # Determine condition and severity
            top_condition = max(probs, key=probs.get) if probs else "normal"
            top_probability = probs.get(top_condition, 0) if probs else 0
            severity = self._get_severity_level(top_probability)
            events.append(self.event("condition_analysis", {
                "condition": top_condition,
                "severity": severity,
                "confidence": top_probability
            }))

            # Check for red flags
            red_flags = []
            red_flag_rules = self.medical_settings.get('red_flags', [
                ("chest pain", "Immediate medical attention advised"),
                ("shortness of breath", "Immediate medical attention advised"),
                ("SpO2 < 92%", "Immediate medical attention advised")
            ])
            
            for phrase, msg in red_flag_rules:
                if phrase.lower() in notes.lower():
                    red_flags.append(msg)
            events.append(self.event("red_flags", {"flags": red_flags}))

            # Get OTC recommendations
            indication_map = self.medical_settings.get('condition_indications', {
                "pneumonia": "fever",
                "covid_suspect": "fever",
                "normal": "cough" if "cough" in notes.lower() else "fever"
            })
            
            indication = indication_map.get(top_condition, "fever")
            candidates = self._filter_otc(indication, age, allergies)

            # Format OTC options
            otc_options = []
            for _, row in candidates.iterrows():
                option = {
                    "sku": row["sku"],
                    "drug_name": row["drug_name"],
                    "dose": row.get("recommended_dose", "as per label"),
                    "freq": row.get("frequency", "as per label"),
                    "warnings": row.get("warnings", "").split(";") if row.get("warnings") else []
                }
                otc_options.append(option)

            # Check interactions and generate advice
            interaction_notes = self._interaction_notes([o["drug_name"] for o in otc_options])
            advice = self._llm_advice(
                summary=f"Condition={top_condition}, Age={age}, Allergies={allergies}",
                meds=otc_options,
                red_flags=red_flags,
                severity=severity
            )

            # Prepare output
            output = {
                "otc_options": otc_options,
                "red_flags": red_flags,
                "interaction_notes": interaction_notes,
                "advice": advice,
                "severity": severity,
                "confidence": top_probability
            }
            events.append(self.event("therapy_complete", {"options": len(otc_options)}))
            
            return AgentResult(output, events)
            
        except Exception as e:
            logging.error(f"Error in TherapyAgent: {str(e)}")
            events.append(self.event("error", {"message": str(e)}))
            return AgentResult(
                {"error": "Failed to process therapy recommendations", "severity": "severe"},
                events
            )