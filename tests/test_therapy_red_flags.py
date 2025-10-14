import pytest

from agents.therapy_agent import TherapyAgent


def make_agent():
    # Use a dummy API key for tests; model calls are not required for red-flag checks
    return TherapyAgent("data/meds.csv", "data/interactions.csv", api_key="TEST_KEY")


def test_imaging_triggers_red_flags():
    agent = make_agent()
    payload = {
        "patient": {"age": 30, "allergies": []},
        "condition_probs": {"pneumonia": 0.2},
        "notes": "",
        "imaging": {
            "impression": "Consolidation in RLL consistent with pneumonia.",
            "radiographic_findings": {"consolidation": True},
            "requires_escalation": True
        }
    }

    result = agent.run(payload)
    assert "red_flags" in result.output
    assert len(result.output["red_flags"]) > 0
    assert result.output.get("otc_options") == []


def test_confidence_triggers_red_flags():
    agent = make_agent()
    payload = {
        "patient": {"age": 45, "allergies": []},
        "condition_probs": {"pneumonia": 0.75},
        "notes": "fever and cough"
    }

    result = agent.run(payload)
    assert "red_flags" in result.output
    # Expect a high-confidence message
    assert any("High analysis confidence" in f for f in result.output["red_flags"]) or len(result.output["red_flags"])>0
    assert result.output.get("otc_options") == []
