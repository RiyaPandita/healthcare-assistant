import pytest
from models import xray_processor

SAMPLE_REPORT = '''Patient ID: CXR-2025-001 Name: [Redacted] Age/Sex: 54 years / Male Clinical History: 
RT-PCR positive for SARS-CoV-2. Fever and cough for 5 days. Shortness of breath on exertion. 
Examination: Chest X-ray (PA view) 
Findings: 
Bilateral patchy ground-glass opacities predominantly involving the lower and middle lung zones. 
Distribution is peripheral with relative sparing of the central regions. 
No significant consolidation. 
No pleural effusion or pneumothorax. 
Cardiac silhouette and mediastinum are within normal limits. 
Severity Score (RALE-like): 
Left lung: 2 (25–50% involvement) 
Right lung: 2 (25–50% involvement) 
Total: 4/8 -> Mapped Severity: 2 (Mild-Moderate)
'''

NEGATED_GGO_REPORT = 'Findings: No ground-glass opacities identified. No consolidation.'

INDETERMINATE_REPORT = 'Findings: Mild ill-defined airspace opacity; cannot exclude early infection.'


def test_sample_report_detection_and_impression():
    rf = xray_processor.parse_report(SAMPLE_REPORT)
    # expectations from the sample
    assert rf['ground_glass_opacity'] is True
    assert rf['consolidation'] is False
    assert rf['pleural_effusion'] is False
    assert rf['pneumothorax'] is False
    assert rf['laterality'] == 'bilateral'
    assert 'lower' in (rf['zones_involved'] or [])
    # generate impression
    mapped, label = xray_processor.map_to_simple_scale(4)
    imp = xray_processor.generate_impression(rf, 2, 2, 4, mapped, label)
    assert 'ground-glass opacities' in imp
    assert 'No pleural effusion' in imp or 'No pleural effusion.' in imp
    assert 'No pneumothorax' in imp or 'No pneumothorax.' in imp


def test_negated_ggo_handling():
    rf = xray_processor.parse_report(NEGATED_GGO_REPORT)
    assert rf['ground_glass_opacity'] is False
    assert rf['consolidation'] is False


def test_indeterminate_phrasing():
    rf = xray_processor.parse_report(INDETERMINATE_REPORT)
    # reticular_thickening/consolidation/ggo may be None or True; ensure impression uses cautious phrasing
    mapped, label = xray_processor.map_to_simple_scale(0)
    imp = xray_processor.generate_impression(rf, 0, 0, 0, mapped, label)
    assert 'does not clearly state' in imp or 'no definite acute airspace abnormality' in imp
