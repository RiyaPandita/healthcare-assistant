import numpy as np
from PIL import Image
from agents.imaging_agent import ImagingAgent


def test_imaging_red_flags_age_and_score(tmp_path, monkeypatch):
    # Create a dummy image
    path = tmp_path / "img.png"
    arr = (np.ones((256, 256)) * 128).astype('uint8')
    Image.fromarray(arr).save(path)

    # Force the xray processor to return percentages that map to RALE buckets of 3 and 3
    # which will give total = 6/8 -> 75% severity
    import models.xray_processor as xp

    monkeypatch.setattr(xp, "estimate_lung_involvement_percentages", lambda img: (60.0, 60.0))

    agent = ImagingAgent()

    payload = {
        "xray_path": str(path),
        "xray_report_path": None,
        "prescription_path": None,
        "patient": {"age": 85}
    }

    result = agent.run(payload)

    # Ensure red flags and evidence are present
    assert isinstance(result.output, dict)
    assert "red_flags" in result.output
    assert len(result.output["red_flags"]) > 0
    # Expect evidence entries for imaging score and age
    evidence = result.output.get("red_flag_evidence", [])
    types = {e.get("type") for e in evidence if isinstance(e, dict)}
    assert "imaging_score" in types or any("severity" in f.lower() for f in result.output["red_flags"])
    assert "age" in types or any("age" in f.lower() for f in result.output["red_flags"])
