# agents/coordinator.py
from typing import Dict, Any
from agents.base import AgentResult
from agents.ingestion_agent import IngestionAgent
from agents.imaging_agent import ImagingAgent
from agents.therapy_agent import TherapyAgent
from agents.pharmacy_agent import PharmacyAgent
from agents.doctor_agent import DoctorAgent

CONFIDENCE_THRESHOLD = 0.5

class Coordinator:
    def __init__(self, ingestion, imaging, therapy, pharmacy, doctor):
        self.ingestion = ingestion
        self.imaging = imaging
        self.therapy = therapy
        self.pharmacy = pharmacy
        self.doctor = doctor

    def run(self, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        all_events = []

        ing = self.ingestion.run(input_payload)
        all_events += ing.events
        if "error" in ing.output:
            return {"error": ing.output["error"], "events": all_events}

        # Ensure the imaging agent receives both ingestion output and the original
        # input payload (which may contain keys like 'xray_report_path' or
        # 'xray_report_bytes' provided by the UI). Merge them so nothing is lost.
        img_input = {**ing.output, **input_payload}
        img = self.imaging.run(img_input)
        all_events += img.events

        # Collect imaging-sourced red flags so we can surface them and use them
        img_red_flags = img.output.get("red_flags", []) if isinstance(img.output, dict) else []

        ther_input = {**ing.output, **img.output}
        ther = self.therapy.run(ther_input)
        all_events += ther.events

        # choose items to order (mock: choose up to configured number of OTC options)
        items = []
        max_suggestions = self.pharmacy.config.settings.get('pharmacy', {}).get('max_otc_suggestions',
                                                                    self.therapy.config.settings.get('therapy', {}).get('max_otc_suggestions', 3))
        if ther.output.get("otc_options") and len(ther.output["otc_options"]) > 0:
            # take up to max_suggestions OTC options
            for opt in ther.output["otc_options"][:max_suggestions]:
                # send both sku and drug_name to pharmacy so the pharmacy agent can
                # resolve inventory by either SKU or drug name (helps when SKU
                # conventions differ between meds.csv and inventory.csv)
                items.append({
                    "sku": opt.get("sku"),
                    "drug_name": opt.get("drug_name"),
                    "qty": 1
                })

        # Combine red flags from therapy and imaging for downstream agents
        ther_reds = ther.output.get("red_flags", []) if isinstance(ther.output, dict) else []
        combined_red_flags = list({*ther_reds, *img_red_flags})

        ph_input = {
            "pincode": input_payload.get("pincode"), 
            "items": items, 
            "red_flags": combined_red_flags
        }
        ph = self.pharmacy.run(ph_input)
        all_events += ph.events

        # check escalation
        top_prob = max(img.output.get("condition_probs", {}).values()) if img.output.get("condition_probs") else 0
        # Consider imaging-originated red flags as escalation triggers as well
        need_escalation = (top_prob < CONFIDENCE_THRESHOLD) or bool(ther_reds) or bool(img_red_flags)
        escalation = {}
        if need_escalation:
            # include ingestion output (patient info) and imaging output when calling doctor
            # this ensures patient.age, allergies, symptoms, and pdf_text from ingestion are passed
            # Include combined red flags when calling the doctor
            doc_in = {**ing.output, **img.output, "red_flags": combined_red_flags}
            esc = self.doctor.run(doc_in)
            all_events += esc.events
            escalation = esc.output

        else:
            # Even when no escalation is required, provide a best-match doctor for UI display.
            # Use the doctor's lightweight selector to avoid calling LLMs unnecessarily.
            try:
                patient = ing.output.get('patient', {})
                patient_age = None
                if isinstance(patient, dict):
                    raw_age = patient.get('age')
                    try:
                        if raw_age is not None and raw_age != "":
                            patient_age = int(float(raw_age))
                    except Exception:
                        patient_age = None

                selected = self.doctor._select_doctor(
                    specialty=None,
                    severity=img.output.get('severity_hint', 'mild'),
                    patient_age=patient_age,
                    allergies=patient.get('allergies') if isinstance(patient, dict) else None,
                    symptoms=ther.output.get('detected_symptoms') or ther.output.get('symptoms'),
                    pdf_text=ing.output.get('pdf_text')
                )
                escalation = {"doctor": selected, "escalate": False}
            except Exception:
                escalation = {}

        final = {
            "ingestion": ing.output,
            "imaging": img.output,
            "therapy": ther.output,
            "pharmacy": ph.output,
            "escalation": escalation,
            "combined_red_flags": combined_red_flags,
            "order": {
                "pharmacy_id": ph.output.get("pharmacy_id"),
                "items": ph.output.get("items", []),
                "eta_min": ph.output.get("eta_min"),
                "delivery_fee": ph.output.get("delivery_fee"),
                "confirmation_id": "ORD-" + str(abs(hash(str(ph.output))))[:8]
            },
            "events": all_events,
            "disclaimer": "Educational demo, not medical advice."
        }
        return final