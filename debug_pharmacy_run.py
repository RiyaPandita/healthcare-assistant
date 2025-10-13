from utils.config import Config
from agents.ingestion_agent import IngestionAgent
from agents.imaging_agent import ImagingAgent
from agents.therapy_agent import TherapyAgent
from agents.pharmacy_agent import PharmacyAgent
from agents.doctor_agent import DoctorAgent
from agents.coordinator import Coordinator
import os

# Setup
config = Config()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
# Create agents (therapy requires API key; if not available we can skip LLM calls, but therapy init requires it)
try:
    ingestion = IngestionAgent()
    imaging = ImagingAgent()
    therapy = TherapyAgent(
        meds_path=config.settings['paths']['data']['meds'],
        interactions_path=config.settings['paths']['data']['interactions'],
        api_key=GEMINI_API_KEY
    )
except Exception as e:
    print('Therapy init or other agent init failed:', e)
    # Try to continue with therapy disabled
    therapy = None

pharmacy = PharmacyAgent(
    pharmacies_path=config.settings['paths']['data']['pharmacies'],
    inventory_path=config.settings['paths']['data']['inventory'],
    zipcodes_path=config.settings['paths']['data']['zipcodes']
)

doctor = None
coord = Coordinator(ingestion, imaging, therapy if therapy else None, pharmacy, doctor)

payload = {
    "patient": {
        "age": 45,
        "allergies": ["none"]
    },
    "notes": "fever,headache,cough",
    "pincode": "400053"
}

# If therapy is None, we'll still run pharmacy by mimicking therapy output
if therapy is None:
    # Create a minimal therapy output that suggests OTC paracetamol and ibuprofen
    otc_options = [
        {"drug_name": "Paracetamol", "sku": "OTC001"},
        {"drug_name": "Ibuprofen", "sku": "OTC009"}
    ]
    ph_input = {
        'pincode': payload['pincode'],
        'items': [{'sku': otc_options[0]['sku'], 'qty': 1}, {'sku': otc_options[1]['sku'], 'qty': 1}],
        'red_flags': []
    }
    ph = pharmacy.run(ph_input)
    print('Simulated therapy OTC options:', otc_options)
    print('Pharmacy output:')
    print(ph.output)
else:
    # Run the full coordinator pipeline
    result = coord.run(payload)
    print('\nTherapy output:')
    print(result.get('therapy'))
    print('\nPharmacy output:')
    print(result.get('pharmacy'))
    print('\nEvents (pharmacy related):')
    for ev in result.get('events', []):
        if ev.get('agent') == 'pharmacy':
            print(ev)
