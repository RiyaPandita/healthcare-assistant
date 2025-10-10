# Multi-Agent Healthcare Assistant (Demo)

**Disclaimer:** Educational demo only. Not medical advice.

## Overview
This project is a proof-of-concept multi-agent system simulating a healthcare assistant.  
It demonstrates ingestion of clinical artifacts, lightweight triage, OTC therapy suggestions, pharmacy matching, and optional doctor escalation.

## Features
- **Ingestion Agent**: Validates uploads, extracts text, masks PII.
- **Imaging Agent**: Dummy classifier for chest X-rays (rule-based stub).
- **Therapy Agent**: Maps conditions to OTC options, checks interactions, generates advice with Gemini LLM.
- **Pharmacy Agent**: Matches to nearest pharmacy with stock from dummy data.
- **Doctor Agent**: Escalates to mock doctor roster when red flags or low confidence.
- **Coordinator**: Orchestrates the workflow and logs events.

## Tech Stack
- Python 3.11
- Streamlit (UI)
- LangGraph (agent orchestration)
- Gemini (LLM reasoning)
- Pandas, Pillow, pdfminer, pytesseract
- Docker for deployment

## Data Mocks
Located in `/data`:
- `pharmacies.json`
- `inventory.csv`
- `doctors.csv`
- `meds.csv`
- `interactions.csv`
- `zipcodes.csv`

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py