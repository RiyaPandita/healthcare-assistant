import io
import re
import logging
import numpy as np
from PIL import Image
import cv2
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)

# Try different PDF parsers in order of preference
def get_pdf_reader():
    """Try different PDF readers in order of preference"""
    readers = [
        ('pypdf', 'pypdf'),
        ('PyPDF2', 'PyPDF2'),
        ('pdfminer.six', 'pdfminer.six')
    ]
    
    for module_name, import_name in readers:
        if importlib.util.find_spec(module_name):
            try:
                if module_name == 'pdfminer.six':
                    from pdfminer.high_level import extract_text
                    return lambda x: type('PdfReader', (), {
                        'pages': [type('Page', (), {'extract_text': lambda: extract_text(x)})()]
                    })()
                else:
                    module = importlib.import_module(import_name)
                    return module.PdfReader
            except ImportError:
                continue
                
    logging.warning("No PDF parser available. PDF report analysis will be limited.")
    return None

SEVERITY_MAP = {
    0: {"label": "None", "mapped": 1},      # Mapping 0 -> 1 (None)
    1: {"label": "Mild", "mapped": 2},
    2: {"label": "Mild", "mapped": 2},
    3: {"label": "Mild-Moderate", "mapped": 3},
    4: {"label": "Mild-Moderate", "mapped": 3},
    5: {"label": "Moderate", "mapped": 4},
    6: {"label": "Moderate-Severe", "mapped": 4},
    7: {"label": "Severe", "mapped": 5},
    8: {"label": "Very Severe", "mapped": 5},
}

# PDF analysis patterns
FINDINGS_PATTERNS = {
    "ground_glass_opacity": r"\b(ground[- ]glass|ggo)\b",
    "consolidation": r"\bconsolidation(s)?\b",
    "reticular_thickening": r"\breticular|\binterstitial thickening\b",
    "pleural_effusion": r"\bpleural effusion\b",
    "pneumothorax": r"\bpneumothorax\b",
}

DISTRIBUTION_PATTERNS = {
    "peripheral": r"\bperipheral\b",
    "central": r"\bcentral|perihilar\b",
    "diffuse": r"\bdiffuse\b",
}

LATERALITY_PATTERNS = {
    "bilateral": r"\bbilateral(ly)?\b",
    "unilateral_right": r"\bright (lung|side)\b",
    "unilateral_left": r"\bleft (lung|side)\b",
}

ZONES_PATTERNS = {
    "upper": r"\bupper\b",
    "middle": r"\bmiddle\b",
    "lower": r"\blower\b",
}

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF using available parser"""
    text = []
    
    # Get PDF reader implementation
    PdfReader = get_pdf_reader()
    if not PdfReader:
        logging.error("No PDF parser available")
        return ""
        
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            except Exception as e:
                logging.warning(f"Error extracting text from page: {str(e)}")
                continue
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
    
    return "\n".join(text).lower()

def parse_report(text: str):
    rf = {
        "ground_glass_opacity": False,
        "consolidation": False,
        "reticular_thickening": False,
        "pleural_effusion": False,
        "pneumothorax": False,
        "laterality": None,
        "zones_involved": [],
        "distribution": None,
    }

    for key, pat in FINDINGS_PATTERNS.items():
        rf[key] = bool(re.search(pat, text))

    # laterality
    if re.search(LATERALITY_PATTERNS["bilateral"], text):
        rf["laterality"] = "bilateral"
    elif re.search(LATERALITY_PATTERNS["unilateral_right"], text):
        rf["laterality"] = "right"
    elif re.search(LATERALITY_PATTERNS["unilateral_left"], text):
        rf["laterality"] = "left"

    # zones
    zones = []
    for z, pat in ZONES_PATTERNS.items():
        if re.search(pat, text):
            zones.append(z)
    rf["zones_involved"] = zones or None

    # distribution
    dist = None
    for d, pat in DISTRIBUTION_PATTERNS.items():
        if re.search(pat, text):
            dist = d
            break
    rf["distribution"] = dist

    return rf

def estimate_lung_involvement_percentages(img: Image.Image):
    """
    Analyzes chest X-ray image to estimate lung involvement percentages
    """
    arr = np.array(img.convert("L"))
    h, w = arr.shape

    # Crop borders
    pad_h = int(0.05 * h)
    pad_w = int(0.05 * w)
    arr = arr[pad_h:h - pad_h, pad_w:w - pad_w]
    h, w = arr.shape

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)

    # Split into left/right ROIs
    mid = w // 2
    shrink = int(0.1 * w)
    left_roi = arr[:, 0:mid - shrink]
    right_roi = arr[:, mid + shrink:w]

    def percent_opacity(roi):
        thresh_val, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        opacity_pixels = np.sum(thresh == 255)
        total_pixels = roi.size
        pct = (opacity_pixels / max(total_pixels, 1)) * 100.0
        return float(np.clip(pct, 0.0, 100.0))

    left_pct = percent_opacity(left_roi)
    right_pct = percent_opacity(right_roi)
    return left_pct, right_pct

def rale_bucket(pct):
    """Maps percentage to RALE score (0-4)"""
    if pct <= 0.0:
        return 0
    elif pct <= 25.0:
        return 1
    elif pct <= 50.0:
        return 2
    elif pct <= 75.0:
        return 3
    else:
        return 4

def map_to_simple_scale(total_score):
    """Maps total score to simplified 1-5 scale"""
    total_score = int(np.clip(total_score, 0, 8))
    mapped = SEVERITY_MAP[total_score]["mapped"]
    label = SEVERITY_MAP[total_score]["label"]
    return mapped, label

def generate_impression(rf, left_score, right_score, total, mapped, mapped_label):
    """Generates a natural language impression from the analysis results"""
    parts = []

    # Findings summary
    locs = []
    if rf.get("laterality") == "bilateral":
        locs.append("bilateral")
    elif rf.get("laterality") in ("left", "right"):
        locs.append(f"{rf['laterality']}-sided")
    if rf.get("zones_involved"):
        locs.append(f"{', '.join(rf['zones_involved'])} zones")
    if rf.get("distribution"):
        locs.append(f"{rf['distribution']} distribution")

    features = []
    if rf.get("ground_glass_opacity"):
        features.append("ground-glass opacities")
    if rf.get("consolidation"):
        features.append("consolidation")
    if rf.get("reticular_thickening"):
        features.append("reticular interstitial thickening")

    if features:
        parts.append(f"Chest radiograph shows {', '.join(features)}" + 
                    (f" with {', '.join(locs)}." if locs else "."))
    else:
        parts.append("Chest radiograph shows no definite acute airspace abnormality by report.")

    if rf.get("pleural_effusion"):
        parts.append("Small pleural effusion noted.")
    if rf.get("pneumothorax"):
        parts.append("No pneumothorax.")

    parts.append(f"Severity estimate (rule-based): Left {left_score}/4, Right {right_score}/4; Total {total}/8.")
    parts.append(f"Mapped severity: {mapped} ({mapped_label}).")
    parts.append("Recommend correlation with clinical status and laboratory markers.")

    return " ".join(parts)