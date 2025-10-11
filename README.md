# Healthcare-Assistant

Transforming clinical artifacts into safe, non-prescriptive care suggestions and coordinated pharmacy fulfilment through a modular multi-agent orchestration.

---

## Table of Contents
- [Overview](#overview)
- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [Agents and responsibilities](#agents-and-responsibilities)
- [Data mocks and schemas](#data-mocks-and-schema-contracts)
- [Safety, privacy, and limitations](#safety-privacy-and-limitations)
- [Deployment](#deployment-notes)
- [Running locally](#quickstart)
- [Tests](#tests)
- [Sample output](#sample-output)
- [Observability](#observability)
- [Development notes and recommended enhancements](#development-notes--recommended-enhancements)
- [License and acknowledgements](#license--acknowledgements)

---

## Overview

**Healthcare-Assistant** is a proof-of-concept framework that demonstrates how a small network of autonomous agents can process clinical artifacts (images and documents), perform lightweight triage, suggest non-prescriptive over-the-counter (OTC) therapies, match available inventory at partner pharmacies, and simulate order placement with optional mock clinician escalation.  
This repository is intended for demonstration, evaluation, and learning only and is **not medical software**.

### Key Highlights
- Modular Python agents communicating via structured JSON events  
- Deterministic mock datasets for reproducible demos  
- Safety-first controls: PII masking, red-flag escalation, and enforced disclaimers  

**Deployed demo:** [https://covid-healthcare-assist.streamlit.app/](https://covid-healthcare-assist.streamlit.app/)  
**Repository:** [https://github.com/RiyaPandita/healthcare-assistant/tree/main](https://github.com/RiyaPandita/healthcare-assistant/tree/main)

---

## Quickstart

### Prerequisites
- Python 3.10+ (3.11 recommended)
- pip
- Tesseract OCR installed on host (if you plan to run OCR locally)
- (Optional) Docker

### Install and Run (pip)
```bash
git clone https://github.com/RiyaPandita/healthcare-assistant.git
cd healthcare-assistant
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

### Run with Docker
```bash
docker build -t healthcare-assistant .
docker run -p 8501:8501 healthcare-assistant
```

Open [http://localhost:8501](http://localhost:8501) or use the deployed demo link.

---

## Project Structure (High Level)

```
agents/          — Agent implementations: ingestion, imaging, therapy, pharmacy, doctor, coordinator
orchestrator/    — Event model, router, and execution graph for routing and logging
models/          — Lightweight stubs and prompt templates (imaging_stub, prompts)
utils/           — Configuration, logging, and PII/security helpers
data/            — Deterministic mock data: pharmacies.json, inventory.csv, meds.csv, interactions.csv, doctors.csv, zipcodes.csv
app.py           — Streamlit UI entrypoint
tests/           — Unit tests validating core handoffs and logic
Dockerfile, requirements.txt, README.md
```

---

## Agents and Responsibilities

| Agent | Responsibility | Key Behaviors |
|--------|----------------|----------------|
| **Ingestion Agent** | Accept uploads, validate files, extract text, de-identify PII | File validation, image/PDF OCR, PII masking, emits sanitized artifact event |
| **Imaging Agent** | Lightweight chest X-ray triage | Deterministic rule-based stub returning condition probabilities and severity_hint |
| **Therapy Agent** | Map conditions → OTC recommendations | Lookup meds.csv, check age/allergy constraints, screen interactions, produce human-readable advice via templating with enforced safety disclaimer |
| **Pharmacy Match Agent** | Find nearest partner pharmacy with stock and simulate reservation | Query pharmacies.json + inventory.csv, compute distance from zipcodes.csv, return ETA, delivery fee and reservation status |
| **Doctor Agent** | Mock escalation / tele-consult target | Select on-call doctor from doctors.csv and prepare sanitized escalation payload |
| **Coordinator / Orchestrator** | Route events, aggregate results, logging | Event routing, fallback handling, threshold checks for escalation, timestamped event log for traceability |

---

## Tech Stack and Notable Libraries

- **Language:** Python 3.10+  
- **UI:** Streamlit  
- **OCR & File Handling:** Pillow, pytesseract, pdfminer.six  
- **Data Processing:** pandas  
- **Geospatial:** haversine or geopy  
- **LLM & Prompts:** pluggable LLM client wrapper (Gemini client or equivalent)  
- **Packaging & Deployment:** Docker, requirements.txt  
- **Testing:** pytest  
- **Utilities:** PyYAML (config), python-logging or custom utils.logging  

---

## Data Mocks and Schema Contracts

All mock data lives under `/data/` to ensure deterministic runs and privacy.

**Files:**
- `pharmacies.json` — partner pharmacies with lat/lon, services, delivery radius  
- `inventory.csv` — pharmacy_id, sku, drug_name, form, strength, price, qty  
- `meds.csv` — sku, drug_name, indication, age_min, contra_allergy_keywords  
- `interactions.csv` — drug_a, drug_b, level, note  
- `doctors.csv` — doctor_id, name, specialty, tele_slot_iso8601[]  
- `zipcodes.csv` — pincode, lat, lon  

### Example Event Shapes

**Ingestion output:**
```json
{
  "patient": {"age": 45, "allergies": ["ibuprofen"]},
  "xray_path": "./uploads/x1.png",
  "notes": "cough, low-grade fever"
}
```

**Imaging output:**
```json
{"condition_probs": {"pneumonia": 0.42, "normal": 0.38, "covid_suspect": 0.20}, "severity_hint": "mild"}
```

**Pharmacy match output:**
```json
{"pharmacy_id":"ph001","items":[{"sku":"OTC001","qty":10}],"eta_min":45,"delivery_fee":25}
```

---

## Safety, Privacy, and Limitations

### Safety Posture
- Prominent educational disclaimer: *This is not medical advice.*  
- No prescriptions are issued; agents only suggest OTC options.  
- Programmatic enforcement of non-prescriptive language in therapy outputs.  

### Privacy & PII
- Ingestion agent masks PII before downstream processing.  
- Uploaded artifacts are treated as ephemeral. Avoid storing real PHI without secure storage and explicit consent.  

### Known Limitations
- Imaging uses deterministic stub; not diagnostic.  
- Inventory/dispatch simulated; no real payments.  
- LLM outputs may vary; post-processed for safety.  

---

## Observability

- Orchestrator emits timestamped event logs for every major decision and handoff.  
- Logs capture: event type, agent name, payload summary, timestamp.  

**Recommended Enhancement:**  
Add collapsible event trace panel in the Streamlit UI for easier review.

---

## Tests

Run unit tests:
```bash
pytest -q
```

### Current Test Coverage
- Imaging stub behavior  
- Drug interaction checks  
- Agent handshake / router tests  

---

## Sample Output

```json
{
  "order_id": "ORD-3291-001",
  "patient_age": 45,
  "items": [{"sku": "OTC001", "name": "Cough Syrup", "qty": 1, "price": 199}],
  "pharmacy_id": "ph001",
  "eta_min": 17,
  "delivery_fee": 79,
  "status": "Processing",
  "events": [
    {"ts": "2025-10-11T15:02:20Z", "agent": "ingestion", "event": "artifact_sanitized"},
    {"ts": "2025-10-11T15:02:24Z", "agent": "imaging", "event": "triage_complete"},
    {"ts": "2025-10-11T15:02:28Z", "agent": "therapy", "event": "otc_suggestions_returned"},
    {"ts": "2025-10-11T15:02:32Z", "agent": "pharmacy", "event": "reservation_confirmed"}
  ]
}
```

---

## Deployment Notes

Deployed publicly on **Streamlit Community Cloud:**  
[https://covid-healthcare-assist.streamlit.app/](https://covid-healthcare-assist.streamlit.app/)

### For Production Readiness:
- Secure LLM keys in secrets manager  
- Use encrypted ephemeral uploads  
- Store audit logs securely  
- Add upload rate and size limits  

---

## Development Notes & Recommended Enhancements

### Immediate (Low-effort)
- Add compact event timeline panel  
- Expand unit tests for PII masking and concurrency  
- Add sample order JSONs and screenshots  

### Medium-term (1–4 weeks)
- Harden LLM output gating  
- Add GitHub Actions CI for tests and lint  
- Add structured JSON logging  

### Longer-term (Research / Sandbox)
- Integrate validated lightweight imaging model  
- Simulate delivery race conditions via mock queue  
- Add role-based access and encrypted storage  

---

## How You Can Contribute

- Fork, improve, and open a PR with tests  
- Suggested issues:  
  - Improved PII regexes  
  - LLM post-processing filter  
  - In-UI trace component  
  - Edge-case test coverage  

---

## License & Acknowledgements

- Add appropriate LICENSE file if missing  
- Built with **Streamlit**, **pandas**, **Pillow**, **pytesseract**, **pdfminer.six**, and standard Python tooling.  

> ⚠️ This project is for **educational and demonstration purposes only.**  
> Do **not** use for real clinical decision-making.
