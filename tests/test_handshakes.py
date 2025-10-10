# tests/test_handshakes.py
from agents.ingestion_agent import IngestionAgent
from agents.imaging_agent import ImagingAgent
from agents.therapy_agent import TherapyAgent
from agents.pharmacy_agent import PharmacyAgent
from agents.doctor_agent import DoctorAgent
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def test_handshake_chain(tmp_path, monkeypatch):
    # create GEMINI_API_KEY image
    from PIL import Image
    img_path = tmp_path / "x.png"
    Image.new("L", (64,64), color=128).save(img_path)

    ingestion = IngestionAgent()
    img_agent = ImagingAgent()
    therapy = TherapyAgent("data/meds.csv", "data/interactions.csv", api_key=GEMINI_API_KEY)
    pharmacy = PharmacyAgent("data/pharmacies.json", "data/inventory.csv", "data/zipcodes.csv")
    doctor = DoctorAgent("data/doctors.csv", api_key=GEMINI_API_KEY)

    ing = ingestion.run({"xray_path": str(img_path), "patient": {"age": 30, "allergies":[]}, "notes": "cough"})
    assert "xray_path" in ing.output

    img = img_agent.run(ing.output)
    assert "condition_probs" in img.output

    ther = therapy.run({**ing.output, **img.output})
    assert "otc_options" in ther.output

    ph = pharmacy.run({"pincode": "400053", "items": [{"sku": ther.output["otc_options"][0]["sku"], "qty":1}]})
    assert "pharmacy_id" in ph.output