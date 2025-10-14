# agents/therapy_agent.py
import os
import re
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from agents.base import BaseAgent, AgentResult
import google.generativeai as genai
from utils.config import Config
import logging

class TherapyAgent(BaseAgent):
    name = "therapy"

    def __init__(self, meds_path: str, interactions_path: str, api_key: str = None):
        super().__init__()  # Initialize the BaseAgent including logger
        import os
        
        # Load data files
        try:
            self.meds = pd.read_csv(meds_path)
            self.interactions = pd.read_csv(interactions_path)
        except Exception as e:
            self.logger.error(f"Failed to load therapy data files: {str(e)}")
            raise
            
        # Configure AI model
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Configure with explicit credentials
        genai.configure(
            api_key=api_key,
            transport="rest"  # Force REST API instead of gRPC
        )
        
        # Load model settings
        model_settings = self.config.settings.get('model', {})
        model_name = model_settings.get('model_name', 'gemini-pro')
        self.model = genai.GenerativeModel(model_name)
        
        # Load medical rules
        try:
            self.medical_rules = self._load_medication_rules()
            if not self.medical_rules:
                raise ValueError("Medical rules are empty")
        except Exception as e:
            self.error(f"INIT_ERROR: Failed to load medical rules: {str(e)}")
            raise
        
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

    def _load_medication_rules(self) -> Dict:
        """Load medication rules from JSON config"""
        rules_path = os.path.join(self.config.base_path, "data", "medical_rules.json")
        try:
            with open(rules_path) as f:
                return json.load(f)
        except Exception as e:
            error_msg = f"Failed to load rules from {rules_path}: {str(e)}"
            self.error(f"RULES_LOAD_ERROR: {error_msg}")
            return {}

    def _check_covid_symptoms(self, symptoms: List[str]) -> Tuple[bool, str]:
        """Validate if symptoms match COVID-19 patterns"""
        primary_symptoms = set(self.medical_rules.get("covid_primary_symptoms", []))
        secondary_symptoms = set(self.medical_rules.get("covid_secondary_symptoms", []))
        non_covid_symptoms = set(self.medical_rules.get("non_covid_symptoms", []))
        validation_rules = self.medical_rules.get("symptom_validation", {})
        
        # Convert symptoms to set for easier operations
        symptom_set = set(symptoms)
        
        # Check for non-COVID symptoms
        if symptom_set.intersection(non_covid_symptoms):
            return False, "Your symptoms suggest a condition other than COVID-19. Please consult a healthcare provider for proper diagnosis."
        
        # Count primary and secondary symptoms
        primary_matches = len(symptom_set.intersection(primary_symptoms))
        secondary_matches = len(symptom_set.intersection(secondary_symptoms))
        total_matches = primary_matches + secondary_matches
        
        # Validate against rules
        min_primary = validation_rules.get("min_primary_symptoms", 1)
        min_total = validation_rules.get("min_total_symptoms", 2)
        
        if primary_matches < min_primary:
            return False, "Your symptoms don't match typical COVID-19 patterns. Please consult a healthcare provider for proper diagnosis."
        
        if total_matches < min_total:
            return False, "Not enough symptoms to suggest COVID-19. Please consult a healthcare provider to determine the cause."
            
        return True, ""

    def _check_red_flags(self, symptoms: List[str], medical_data: Dict) -> List[str]:
        """Check for red flag symptoms and measurements"""
        red_flags = []
        red_flag_symptoms = self.medical_rules.get("red_flags", {}).get("symptoms", [])
        red_flag_measurements = self.medical_rules.get("red_flags", {}).get("measurements", {})

        # Check symptoms
        for symptom in symptoms:
            if symptom in red_flag_symptoms:
                red_flags.append(f"Warning: {symptom} requires immediate medical attention")

        # Check measurements if available
        measurements = medical_data.get("measurements", {})
        for measure, threshold in red_flag_measurements.items():
            if measure in measurements:
                value = float(measurements[measure])
                threshold_value = float(threshold.replace("<", "").replace(">", ""))
                if (">" in threshold and value > threshold_value) or ("<" in threshold and value < threshold_value):
                    red_flags.append(f"Warning: {measure} reading requires medical attention")

        return red_flags

    def _determine_age_group(self, age: int) -> str:
        """Determine patient age group and applicable restrictions"""
        age_groups = self.medical_rules.get("age_groups", {})
        for group, data in age_groups.items():
            range_str = data.get("range", "")
            if "-" in range_str:
                min_age, max_age = map(int, range_str.split("-"))
                if min_age <= age <= max_age:
                    return group
            elif ">=" in range_str:
                min_age = int(range_str.replace(">=", ""))
                if age >= min_age:
                    return group
        return "adult"

    def _filter_otc(self, symptoms: List[str], age: int, allergies: List[str], medical_data: Dict) -> List[Dict]:
        """Enhanced OTC medication filter with comprehensive rules"""
        try:
            # Validate COVID-19 symptoms when relevant. Don't abort OTC flow entirely
            # if COVID patterns are not met; surface validation messages as warnings instead.
            is_covid, validation_message = self._check_covid_symptoms(symptoms)
            warnings = []
            if not is_covid and validation_message:
                warnings.append(validation_message)
            
            # Check for red flags
            red_flags = self._check_red_flags(symptoms, medical_data)
            if red_flags:
                return [], red_flags
            
            # Determine age group and restrictions
            age_group = self._determine_age_group(age)
            age_restrictions = self.medical_rules["age_groups"][age_group].get("restrictions", [])
            
            # Process allergies
            allergy_map = self.medical_rules.get("allergy_keywords", {})
            expanded_allergies = set()  # Use a set to avoid duplicates
            
            # First, get all allergy keywords that match patient's allergies
            for allergy in allergies:
                allergy_lower = allergy.lower()
                for category, keywords in allergy_map.items():
                    if any(kw in allergy_lower for kw in keywords):
                        expanded_allergies.update(keywords)
                        expanded_allergies.add(category)  # Add the category name too
            
            # Get medication recommendations
            recommendations = {}  # Use dict to track unique medications by SKU
            warnings = []
            processed_drugs = set()  # Track processed drugs to avoid duplicates

            # Collect safe options per symptom
            symptom_safe_options: Dict[str, List[str]] = {}
            for symptom in symptoms:
                symptom_safe_options[symptom] = []
                if symptom in self.medical_rules["symptoms"]:
                    symptom_data = self.medical_rules["symptoms"][symptom]
                    # Filter medications based on contraindications and age restrictions
                    for med in symptom_data["otc_options"]:
                        # Check if med is restricted by age
                        if med in age_restrictions:
                            continue
                        # Check if med's name or category matches any expanded allergies
                        med_lower = med.lower()
                        is_safe = True
                        for allergy in expanded_allergies:
                            if allergy.lower() in med_lower or med_lower in allergy.lower():
                                is_safe = False
                                break
                        if is_safe:
                            symptom_safe_options[symptom].append(med)

            # If multiple symptoms detected, prefer meds that appear in ALL symptom lists (intersection)
            meds_to_add = set()
            non_empty_lists = [opts for opts in symptom_safe_options.values() if opts]
            if len(non_empty_lists) > 1:
                # Compute intersection across symptom-safe lists
                intersect = set(non_empty_lists[0]).intersection(*non_empty_lists[1:])
                if intersect:
                    meds_to_add = intersect
                else:
                    # Fall back to union with a warning
                    meds_to_add = set().union(*non_empty_lists)
                    warnings.append("Recommendations cover mixed symptoms; consider consulting if symptoms persist or are severe.")
            else:
                # Single symptom or none - use the single list or empty
                meds_to_add = set(non_empty_lists[0]) if non_empty_lists else set()

            # Add meds_to_add to recommendations
            for med in meds_to_add:
                if med.lower() in processed_drugs:
                    continue
                med_data = self.meds[self.meds["drug_name"].str.contains(med, case=False)]
                if not med_data.empty:
                    row = med_data.iloc[0]
                    sku = row["sku"]
                    inventory_sku = None
                    try:
                        inv_path = os.path.join(self.config.base_path, "data", "inventory.csv")
                        inv_df = pd.read_csv(inv_path)
                        if 'drug_name' in inv_df.columns and 'sku' in inv_df.columns:
                            match = inv_df[inv_df['drug_name'].str.contains(med, case=False, na=False)]
                            if not match.empty:
                                inventory_sku = match.iloc[0]['sku']
                    except Exception:
                        inventory_sku = None
                    returned_sku = inventory_sku if inventory_sku else sku
                    if returned_sku not in recommendations:
                        recommendations[returned_sku] = {
                            "drug_name": med,
                            "sku": returned_sku,
                            "dose": row.get("recommended_dose", "as per label"),
                            "freq": row.get("frequency", "as per label"),
                            "warnings": row.get("warnings", "").split(";") if row.get("warnings") else []
                        }
                        processed_drugs.add(med.lower())
                        
            # Convert recommendations dict to list and ensure it's never None
            return list(recommendations.values()) or [], warnings
            
        except Exception as e:
            error_msg = f"Error filtering OTC medications: {str(e)}"
            self.error(f"OTC_FILTER_ERROR: {error_msg}")
            return [], [error_msg]

    def _check_interaction(self, drug1: str, drug2: str) -> Optional[str]:
        """Check for interaction between two specific drugs."""
        try:
            # Create a unique interaction key by sorting drug names
            drug1_lower = drug1.lower()
            drug2_lower = drug2.lower()
            interaction_key = tuple(sorted([drug1_lower, drug2_lower]))
            
            # Look for interaction
            interaction = self.interactions[
                ((self.interactions["drug_a"].str.lower() == interaction_key[0]) & 
                 (self.interactions["drug_b"].str.lower() == interaction_key[1])) |
                ((self.interactions["drug_a"].str.lower() == interaction_key[1]) & 
                 (self.interactions["drug_b"].str.lower() == interaction_key[0]))
            ]
            
            if not interaction.empty:
                row = interaction.iloc[0]
                # Use original drug names but in sorted order
                drugs_sorted = sorted([drug1, drug2])
                return f"Interaction Warning: {drugs_sorted[0]} + {drugs_sorted[1]}: {row['level']} - {row['note']}"
            return None
        except Exception as e:
            self.error(f"Failed to check interaction between {drug1} and {drug2}: {str(e)}")
            return None

    def _interaction_notes(self, chosen_names: List[str]) -> List[str]:
        """Check for drug interactions between chosen medications."""
        notes = set()  # Using a set to prevent duplicates
        try:
            # Sort drug names to ensure consistent ordering
            sorted_names = sorted(chosen_names)
            for i, drug1 in enumerate(sorted_names):
                for drug2 in sorted_names[i+1:]:
                    interaction = self.interactions[
                        ((self.interactions["drug_a"].str.contains(drug1, case=False)) & 
                         (self.interactions["drug_b"].str.contains(drug2, case=False))) |
                        ((self.interactions["drug_a"].str.contains(drug2, case=False)) & 
                         (self.interactions["drug_b"].str.contains(drug1, case=False)))
                    ]
                    if not interaction.empty:
                        row = interaction.iloc[0]
                        notes.add(f"{row['drug_a']} + {row['drug_b']}: {row['level']} - {row['note']}")
        except Exception as e:
            self.error(f"Error checking drug interactions: {str(e)}")
        return list(notes)

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

            # Check for red flags (phrases + imaging + confidence + measurements)
            red_flags = []
            red_flag_rules = self.medical_settings.get('red_flags', [
                ("chest pain", "Immediate medical attention advised"),
                ("shortness of breath", "Immediate medical attention advised"),
                ("SpO2 < 92%", "Immediate medical attention advised")
            ])

            # Phrase-based rules from notes/pdf
            for phrase, msg in red_flag_rules:
                if phrase.lower() in notes.lower():
                    red_flags.append(msg)

            # Pull through imaging outputs if present (xray/impression)
            imaging_out = payload.get("imaging") or payload.get("imaging_output") or {}
            try:
                impression = (imaging_out.get("impression") or "").lower() if isinstance(imaging_out, dict) else ""
                rf = imaging_out.get("radiographic_findings") if isinstance(imaging_out, dict) else None

                # Radiographic findings indicating escalation
                if rf and isinstance(rf, dict):
                    if rf.get("consolidation"):
                        red_flags.append("Imaging: consolidation noted — consider urgent review")
                    if rf.get("ground_glass_opacity"):
                        red_flags.append("Imaging: ground-glass opacities noted — consider urgent review")
                    if rf.get("pleural_effusion"):
                        red_flags.append("Imaging: pleural effusion noted — consider urgent review")
                    if rf.get("pneumothorax"):
                        red_flags.append("Imaging: pneumothorax noted — immediate attention advised")

                # Impression text keywords
                if impression:
                    if any(k in impression for k in ("pneumonia", "consolidation", "ground glass", "air bronchogram")):
                        red_flags.append("Imaging impression suggests pneumonia/consolidation — consider escalation")
                    if any(k in impression for k in ("pleural effusion", "pneumothorax", "tension pneumothorax")):
                        red_flags.append("Imaging impression suggests pleural effusion/pneumothorax — immediate attention advised")

                # Explicit imaging escalation flag
                if isinstance(imaging_out, dict) and imaging_out.get("requires_escalation"):
                    red_flags.append("Imaging indicates escalation is recommended")
            except Exception:
                # Non-fatal: continue with other checks
                pass

            # Confidence-based rule: if initial analysis confidence > 60% treat as red-flag for review
            try:
                if top_probability and float(top_probability) > 0.6:
                    red_flags.append(f"High analysis confidence ({float(top_probability)*100:.0f}%) for {top_condition} — consider escalation")
            except Exception:
                pass

            # Extract symptoms from notes
            symptom_keywords = {
                "fever": ["fever", "temperature", "hot"],
                "cough": ["cough", "coughing"],
                "sore_throat": ["sore throat", "throat pain"],
                "headache": ["headache", "head pain"],
                "body_ache": ["body ache", "muscle pain", "myalgia"],
                "fatigue": ["fatigue", "tired", "exhaustion"],
                "nasal_congestion": ["congestion", "stuffy nose", "blocked nose"]
            }
            
            detected_symptoms = []
            for symptom, keywords in symptom_keywords.items():
                if any(keyword in notes.lower() for keyword in keywords):
                    detected_symptoms.append(symptom)
            
            # If no symptoms detected from notes, we do NOT fallback to
            # condition-derived symptoms. OTC recommendations should only be
            # considered when explicit symptom keywords, allergies, or medical
            # measurements (e.g., temperature, SpO2) are present.
            # Keep detected_symptoms empty to avoid suggesting OTCs when there
            # is no explicit evidence.
            # (No action required here; detected_symptoms remains as computed.)
            
            # Get medical data from notes and PDF
            medical_data = {
                "measurements": {},
                "conditions": []
            }
            
            # Extract measurements from notes (SpO2, temperature, etc.)
            spo2_match = re.search(r"SpO2[:\s]*(\d+)", notes)
            if spo2_match:
                medical_data["measurements"]["spo2"] = spo2_match.group(1)
                
            temp_match = re.search(r"temp[erature]*[:\s]*(\d+\.?\d*)", notes, re.IGNORECASE)
            if temp_match:
                medical_data["measurements"]["temperature"] = temp_match.group(1)
            
            # Emit red_flags event now that we have collected phrase/imaging/confidence flags
            events.append(self.event("red_flags", {"flags": red_flags}))

            # If red flags exist, skip OTC recommendations and surface warnings
            if red_flags:
                otc_options = []
                warnings = ["Escalation required based on red-flag findings; no OTC recommendations provided."]
            else:
                # Get OTC recommendations with enhanced logic
                otc_options, warnings = self._filter_otc(detected_symptoms, age, allergies, medical_data)
                otc_options = otc_options if otc_options is not None else []
                warnings = warnings if warnings is not None else []

            # Process drug interactions once and store them
            drug_names = [o["drug_name"] for o in otc_options]
            processed_interactions = set()  # Track processed interactions to avoid duplicates
            
            for i, drug1 in enumerate(drug_names):
                for drug2 in drug_names[i+1:]:
                    interaction = self._check_interaction(drug1, drug2)
                    if interaction:
                        processed_interactions.add(interaction)
            
            interaction_notes = list(processed_interactions)
            
            # Generate advice with processed interactions
            advice = self._llm_advice(
                summary=f"Condition={top_condition}, Age={age}, Allergies={allergies}",
                meds=otc_options,
                red_flags=red_flags,
                severity=severity
            )

            # Prepare output with defaults for all fields
            output = {
                "otc_options": otc_options or [],  # Ensure it's never None
                "red_flags": red_flags or [],
                "interaction_notes": interaction_notes or [],
                "advice": advice or "Please consult a healthcare professional.",
                "severity": severity or "normal",
                "confidence": top_probability or 0,
                "warnings": warnings if 'warnings' in locals() else []  # Include any validation warnings
            }
            events.append(self.event("therapy_complete", {"options": len(otc_options)}))
            
            return AgentResult(output, events)
            
        except Exception as e:
            error_msg = f"Error in TherapyAgent: {str(e)}"
            self.error(f"THERAPY_ERROR: {error_msg}")
            events.append(self.event("error", {"message": str(e)}))
            return AgentResult(
                {
                    "error": "Failed to process therapy recommendations",
                    "severity": "severe",
                    "red_flags": [],  # Ensure red_flags exists even in error state
                    "otc_options": [],
                    "warnings": ["System error occurred. Please try again or consult a healthcare provider."]
                },
                events
            )