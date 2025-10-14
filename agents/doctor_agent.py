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
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required")
            
            # Configure with explicit credentials
            genai.configure(
                api_key=api_key,
                transport="rest"  # Force REST API instead of gRPC
            )
            
            model_name = model_settings.get('model_name', 'gemini-pro')
            self.model = genai.GenerativeModel(model_name)
            
        except Exception as e:
            logging.error(f"Error initializing DoctorAgent: {str(e)}")
            raise

    def _select_doctor(self, specialty: Optional[str] = None, severity: str = "mild",
                       preferred_language: Optional[str] = None,
                       patient_age: Optional[int] = None,
                       allergies: Optional[list] = None,
                       symptoms: Optional[list] = None,
                       pdf_text: Optional[str] = None) -> Dict[str, Any]:
        """Select appropriate doctor based on specialty and severity"""
        try:
            candidates = self._get_candidates(
                specialty=specialty,
                severity=severity,
                patient_age=patient_age,
                allergies=allergies,
                symptoms=symptoms,
                pdf_text=pdf_text,
            )

            if candidates is None or candidates.empty:
                return {
                    "name": "On-call Doctor",
                    "specialty": "General Medicine",
                    "tele_slot_iso8601": "N/A"
                }

            best = candidates.iloc[0].to_dict()
            return best

        except Exception as e:
            logging.error(f"Error selecting doctor: {str(e)}")
            return {
                "name": "On-call Doctor",
                "specialty": "General Medicine",
                "tele_slot_iso8601": "N/A"
            }

    def _get_candidates(self, *,
                        specialty: Optional[str] = None,
                        severity: str = "mild",
                        patient_age: Optional[int] = None,
                        allergies: Optional[list] = None,
                        symptoms: Optional[list] = None,
                        pdf_text: Optional[str] = None):
        """Return a DataFrame of filtered and sorted candidate doctors for given inputs."""
        try:
            if self.doctors.empty:
                return pd.DataFrame()

            available_docs = self.doctors.copy()

            # Determine effective specialty before filtering using age, symptoms, and pdf_text
            effective_specialty = specialty

            # Age-based specialty adjustments: prefer pediatric/geriatrics when appropriate
            age_based_specialty = None
            try:
                if isinstance(patient_age, int):
                    if patient_age <= 16:
                        age_based_specialty = "Pediatrics"
                    elif patient_age >= 65:
                        age_based_specialty = "Geriatrics"
            except Exception:
                age_based_specialty = None

            # If age-based specialty found, honor it (priority over keyword inference)
            if age_based_specialty:
                effective_specialty = age_based_specialty
            else:
                # Symptoms and PDF text keyword matching to refine specialty
                keywords = set()
                if symptoms and isinstance(symptoms, (list, tuple)):
                    keywords.update([s.lower() for s in symptoms if isinstance(s, str)])
                if pdf_text and isinstance(pdf_text, str):
                    for token in ["cough", "shortness of breath", "dyspnea", "chest pain", "pleuritic", "fever", "pneumonia", "consolidation", "effusion", "pneumothorax", "angina", "palpitations", "ear pain", "sore throat", "sinus", "headache", "seizure", "rash", "itching"]:
                        if token in pdf_text.lower():
                            keywords.add(token)

                # Specialty refinements based on keywords
                if keywords:
                    kw = keywords
                    if any(k in kw for k in ("cough", "shortness of breath", "dyspnea", "pneumonia", "consolidation", "effusion")):
                        effective_specialty = "Pulmonology"
                    if any(k in kw for k in ("chest pain", "pleuritic", "angina", "palpitations")):
                        effective_specialty = "Cardiology"
                    if any(k in kw for k in ("ear pain", "hearing loss", "tinnitus", "sore throat", "sinus")):
                        effective_specialty = "ENT"
                    if any(k in kw for k in ("headache", "seizure", "weakness", "neurologic", "stroke")):
                        effective_specialty = "Neurology"
                    if any(k in kw for k in ("rash", "itching", "dermatitis", "eczema")):
                        effective_specialty = "Dermatology"
                    if "fever" in kw and any(k in kw for k in ("fever", "malaise", "chills")):
                        effective_specialty = "Infectious Disease"

            # Filter by effective specialty if provided
            if effective_specialty:
                specialty_mask = available_docs["specialty"].str.contains(effective_specialty, case=False, na=False)
                if specialty_mask.any():
                    available_docs = available_docs[specialty_mask]

            # Rule: for severe cases, require emergency_available == true if any doctor is available
            if severity == "severe":
                if "emergency_available" in available_docs.columns:
                    severe_mask = available_docs["emergency_available"].astype(str).str.lower().isin(["true", "1", "yes"])
                    if severe_mask.any():
                        available_docs = available_docs[severe_mask]

            # Add sorting keys: prefer higher rating, more experience, fewer consultations (less busy)
            sort_by = []
            if "rating" in available_docs.columns:
                sort_by.append(("rating", False))
            if "experience_years" in available_docs.columns:
                sort_by.append(("experience_years", False))
            if "consultations" in available_docs.columns:
                sort_by.append(("consultations", True))

            # Build pandas sort parameters
            if sort_by:
                by_cols = [col for col, _asc in sort_by]
                ascending = [asc for _col, asc in sort_by]
                available_docs = available_docs.sort_values(by=by_cols, ascending=ascending, na_position='last')

            # As final tie-breaker, sort by experience descending if not already included
            if "experience_years" in available_docs.columns and (not any(k == "experience_years" for k, _ in sort_by)):
                available_docs = available_docs.sort_values("experience_years", ascending=False, kind='mergesort')

            return available_docs

        except Exception as e:
            logging.error(f"Error building candidate list: {str(e)}")
            return pd.DataFrame()

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

            # Pull through imaging/ingestion outputs if present
            imaging_out = payload.get("imaging") or payload.get("imaging_output") or {}
            ingestion_out = payload.get("ingestion") or payload.get("ingestion_output") or {}

            # Patient info
            patient = payload.get("patient", {}) or ingestion_out.get("patient", {})
            patient_age = None
            allergies = None
            if isinstance(patient, dict):
                # coerce age to int when possible (UI may send as string)
                raw_age = patient.get("age")
                try:
                    if raw_age is not None and raw_age != "":
                        patient_age = int(float(raw_age))
                except Exception:
                    patient_age = None
                allergies = patient.get("allergies")

            # Extract symptoms and pdf text from ingestion payload if present
            symptoms = payload.get("symptoms") or ingestion_out.get("symptoms") or None
            pdf_text = payload.get("pdf_text") or ingestion_out.get("pdf_text") or None
            
            # Determine specialty based on top condition, imaging findings, or impression text
            top_condition = max(conditions.items(), key=lambda x: x[1])[0] if conditions else None
            specialty_map = {
                "pneumonia": "Pulmonology",
                "covid_suspect": "Internal Medicine",
                "tuberculosis": "Pulmonology"
            }
            specialty = specialty_map.get(top_condition, "General Medicine")

            # Use imaging outputs and pdf text to refine specialty and severity if available
            try:
                # imaging_out may contain 'impression' string and 'radiographic_findings' dict
                impression = (imaging_out.get("impression") or "").lower() if isinstance(imaging_out, dict) else ""
                rf = imaging_out.get("radiographic_findings") if isinstance(imaging_out, dict) else None

                # If imaging flags consolidation or ground glass -> Pulmonology
                if rf and isinstance(rf, dict):
                    if rf.get("consolidation") or rf.get("ground_glass_opacity"):
                        specialty = "Pulmonology"
                    if rf.get("pleural_effusion") or rf.get("pneumothorax"):
                        specialty = "Emergency Medicine"

                # Check impression text for keywords
                if impression:
                    if "pneumonia" in impression or "consolidation" in impression or "ground glass" in impression:
                        specialty = "Pulmonology"
                    if "pleural effusion" in impression or "pneumothorax" in impression or "tension pneumothorax" in impression:
                        specialty = "Emergency Medicine"

                # If imaging explicitly requests escalation, treat as severe
                if isinstance(imaging_out, dict) and imaging_out.get("requires_escalation"):
                    severity = "severe"
            except Exception:
                # Non-fatal: proceed with earlier specialty/severity
                pass
            
            # Select doctor and generate note (pass through patient info, symptoms and pdf text)
            # Also get candidate list for debug
            candidates = self._get_candidates(
                specialty=specialty,
                severity=severity,
                patient_age=patient_age,
                allergies=allergies,
                symptoms=symptoms,
                pdf_text=pdf_text,
            )
            doctor = self._select_doctor(
                specialty,
                severity,
                preferred_language=None,  # language not used in new logic
                patient_age=patient_age,
                allergies=allergies,
                symptoms=symptoms,
                pdf_text=pdf_text
            )
            note = self._generate_escalation_note(red_flags, severity, conditions)
            
            output = {
                "escalate": True,
                "doctor": doctor,
                "note": note,
                "severity": severity,
                "specialty_matched": specialty
            }
            
            # Emit debug info about candidate list for UI troubleshooting
            try:
                candidate_names = candidates["name"].tolist() if candidates is not None and not candidates.empty else []
            except Exception:
                candidate_names = []

            events.append(self.event("selection_debug", {
                "effective_specialty": specialty,
                "candidates": candidate_names
            }))

            events.append(self.event("escalation_initiated", {
                "doctor": doctor["name"],
                "severity": severity,
                "specialty": specialty,
                "patient_age": patient_age
            }))
            
            return AgentResult(output, events)
            
        except Exception as e:
            logging.error(f"Error in DoctorAgent: {str(e)}")
            events.append(self.event("error", {"message": str(e)}))
            return AgentResult({
                "error": "Failed to process escalation",
                "severity": "severe"
            }, events)