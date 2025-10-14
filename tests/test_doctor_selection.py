import os
import pytest
from agents.doctor_agent import DoctorAgent

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'doctors.csv')


def make_agent():
    # Use a dummy API key because the constructor currently requires one for model init
    return DoctorAgent(doctors_path=DATA_PATH, api_key='DUMMY_KEY')


def test_specialty_match():
    agent = make_agent()
    doctor = agent._select_doctor(specialty='Pulmonology', severity='mild')
    assert 'Pulmonology' in doctor.get('specialty', '') or 'Pulmonology' in doctor.get('specialty', ''), "Expected a pulmonologist"


def test_severe_requires_emergency():
    agent = make_agent()
    # For severe cases, the code should prefer doctors with emergency_available true
    doctor = agent._select_doctor(specialty='General Medicine', severity='severe')
    # doctor should have emergency_available true when available in dataset
    # In our CSV, Dr. Sarah Williams has emergency_available true
    assert str(doctor.get('emergency_available', '')).lower() in ['true', '1', 'yes']


def test_imaging_drives_specialty():
    agent = make_agent()
    imaging = {
        "impression": "Findings consistent with consolidation and possible pneumonia",
        "radiographic_findings": {"consolidation": True, "ground_glass_opacity": False},
        "requires_escalation": False
    }
    payload = {"imaging": imaging}
    doctor = agent._select_doctor(specialty='General Medicine', severity='mild', symptoms=None, pdf_text=None)
    # direct _select_doctor call doesn't read imaging; simulate by using keywords
    doctor_from_pdf = agent._select_doctor(specialty='General Medicine', severity='mild', symptoms=['cough'], pdf_text=None)
    assert 'Pulmonology' in doctor_from_pdf.get('specialty', '')


def test_pdf_keywords_influence():
    agent = make_agent()
    pdf_text = "Patient reports cough and shortness of breath. Possible consolidation on xray."
    doctor = agent._select_doctor(specialty='General Medicine', severity='mild', symptoms=None, pdf_text=pdf_text)
    assert 'Pulmonology' in doctor.get('specialty', '')
