from agents.ingestion_agent import IngestionAgent
from agents.imaging_agent import ImagingAgent
from agents.therapy_agent import TherapyAgent
from agents.pharmacy_agent import PharmacyAgent
from agents.doctor_agent import DoctorAgent

class AgentRouter:
    def __init__(self, config):
        self.agents = {
            "ingestion": IngestionAgent(),
            "imaging": ImagingAgent(),
            "therapy": TherapyAgent(
                config["meds_path"],
                config["interactions_path"],
                config["gemini_api_key"]
            ),
            "pharmacy": PharmacyAgent(
                config["pharmacies_path"],
                config["inventory_path"],
                config["zipcodes_path"]
            ),
            "doctor": DoctorAgent(
                config["doctors_path"],
                config["gemini_api_key"]
            )
        }

    def run(self, agent_name: str, payload: dict):
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        return self.agents[agent_name].run(payload)