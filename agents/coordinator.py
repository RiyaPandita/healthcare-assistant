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

        img = self.imaging.run(ing.output)
        all_events += img.events

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
        
        ph_input = {
            "pincode": input_payload.get("pincode"), 
            "items": items, 
            "red_flags": ther.output.get("red_flags", [])
        }
        ph = self.pharmacy.run(ph_input)
        all_events += ph.events

        # check escalation
        top_prob = max(img.output.get("condition_probs", {}).values()) if img.output.get("condition_probs") else 0
        red_flags = ther.output.get("red_flags", [])
        need_escalation = (top_prob < CONFIDENCE_THRESHOLD) or bool(red_flags)
        escalation = {}
        if need_escalation:
            doc_in = {**img.output, "red_flags": red_flags}
            esc = self.doctor.run(doc_in)
            all_events += esc.events
            escalation = esc.output

        final = {
            "ingestion": ing.output,
            "imaging": img.output,
            "therapy": ther.output,
            "pharmacy": ph.output,
            "escalation": escalation,
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