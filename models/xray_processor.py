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

import io
import logging
import importlib.util
# pdf2image and pytesseract are optional and require system packages
# (poppler, tesseract). Import lazily inside extract_pdf_text to avoid
# import-time failures on platforms like Streamlit Cloud where system
# packages may be missing.

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF file using the best available backend.
    Automatically falls back to OCR if text extraction fails.
    """

    text = []

    # --- 1️⃣ Preferred pure-text parsers ---
    try:
        # Try pypdf
        if importlib.util.find_spec("pypdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            logging.info("PDF text extracted using pypdf.")
    except Exception as e:
        logging.warning(f"pypdf extraction failed: {e}")

    if not text:
        try:
            # Try PyPDF2
            if importlib.util.find_spec("PyPDF2"):
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(pdf_bytes))
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                logging.info("PDF text extracted using PyPDF2.")
        except Exception as e:
            logging.warning(f"PyPDF2 extraction failed: {e}")

    if not text:
        try:
            # Try pdfminer.six
            if importlib.util.find_spec("pdfminer.high_level"):
                from pdfminer.high_level import extract_text
                page_text = extract_text(io.BytesIO(pdf_bytes))
                if page_text:
                    text.append(page_text)
                logging.info("PDF text extracted using pdfminer.six.")
        except Exception as e:
            logging.warning(f"pdfminer.six extraction failed: {e}")

    # --- 2️⃣ OCR fallback for scanned PDFs ---
    if not text:
        try:
            try:
                from pdf2image import convert_from_bytes
            except Exception:
                logging.warning("pdf2image unavailable; skipping OCR fallback (requires poppler).")
                raise

            try:
                import pytesseract
            except Exception:
                logging.warning("pytesseract unavailable; skipping OCR fallback (requires tesseract).")
                raise

            images = convert_from_bytes(pdf_bytes)
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    text.append(ocr_text)
            logging.info("PDF text extracted using OCR fallback.")
        except Exception as e:
            logging.error(f"OCR fallback failed: {e}")

    # --- 3️⃣ Final cleanup ---
    final_text_raw = "\n".join(text)

    # Normalize common unicode punctuation that commonly appears in reports
    # (various hyphen/dash characters, minus sign, non-breaking hyphen, arrows)
    final_text_norm = re.sub(r'[\u2010-\u2015\u2212\u2011]', '-', final_text_raw)
    final_text_norm = final_text_norm.replace('\u2192', '->').replace('→', '->')

    final_text = final_text_norm.lower().strip()
    if not final_text.strip():
        logging.warning("⚠️ No text could be extracted from the PDF.")
    else:
        logging.info(f"✅ Successfully extracted ~{len(final_text.split())} words from PDF.")

    return final_text


def extract_and_parse_pdf(pdf_bytes: bytes):
    """
    Convenience helper for UI: extract text from PDF bytes and return structured
    parsed results including radiographic findings, RALE-like scores (if present),
    mapped severity and a generated impression.

    Returns a dict with keys: text, parsed_findings, left, right, total, mapped, mapped_label, impression
    """
    text = extract_pdf_text(pdf_bytes)
    rf = parse_report(text)

    # Attempt to find explicit RALE-like numeric scores in the text
    left = right = 0
    m_left = re.search(r'left\s*lung\s*[:\-]?\s*(\d)', text, flags=re.IGNORECASE)
    if m_left:
        left = int(m_left.group(1))
    m_right = re.search(r'right\s*lung\s*[:\-]?\s*(\d)', text, flags=re.IGNORECASE)
    if m_right:
        right = int(m_right.group(1))

    # fallback patterns
    if not (left or right):
        m = re.search(r'left lung:.*?(\d)', text, flags=re.IGNORECASE)
        if m:
            left = int(m.group(1))
        m2 = re.search(r'right lung:.*?(\d)', text, flags=re.IGNORECASE)
        if m2:
            right = int(m2.group(1))

    total = left + right
    mapped, mapped_label = map_to_simple_scale(total)
    impression = generate_impression(rf, left, right, total, mapped, mapped_label)

    return {
        'text': text,
        'parsed_findings': rf,
        'left': left,
        'right': right,
        'total': total,
        'mapped': mapped,
        'mapped_label': mapped_label,
        'impression': impression,
    }

def parse_report(text: str):
    rf = {
        "ground_glass_opacity": None,
        "consolidation": None,
        "reticular_thickening": None,
        "pleural_effusion": None,
        "pneumothorax": None,
        "laterality": None,
        "zones_involved": [],
        "distribution": None,
    }

    # Normalize text: lowercase and normalize common unicode hyphen/minus characters
    normalized_text = text.lower() if isinstance(text, str) else ''
    # replace various hyphen/minus characters with ASCII hyphen
    normalized_text = re.sub(r'[\u2010-\u2015\u2212\u2011]', '-', normalized_text)

    # Helper to detect negation around a regex match
    negation_words = ["no", "without", "absent", "none", "no evidence of", "not seen", "negative for", "no definite", "no significant"]

    def _is_match_negated(m):
        # check preceding text window and following window for negation cues
        start, end = m.start(), m.end()
        window_before = normalized_text[max(0, start - 60):start]
        window_after = normalized_text[end:end + 60]
        combined = window_before + " " + window_after
        for w in negation_words:
            if w in combined:
                return True
        return False

    for key, pat in FINDINGS_PATTERNS.items():
        found_any = False
        found_positive = False
        found_negated = False
        for m in re.finditer(pat, normalized_text, flags=re.IGNORECASE):
            found_any = True
            if _is_match_negated(m):
                found_negated = True
            else:
                found_positive = True

        if found_positive:
            rf[key] = True
        elif found_negated:
            rf[key] = False
        else:
            rf[key] = None

    # laterality
    # laterality (use normalized_text)
    if re.search(LATERALITY_PATTERNS["bilateral"], normalized_text, flags=re.IGNORECASE):
        rf["laterality"] = "bilateral"
    elif re.search(LATERALITY_PATTERNS["unilateral_right"], normalized_text, flags=re.IGNORECASE):
        rf["laterality"] = "right"
    elif re.search(LATERALITY_PATTERNS["unilateral_left"], normalized_text, flags=re.IGNORECASE):
        rf["laterality"] = "left"

    # zones
    zones = []
    for z, pat in ZONES_PATTERNS.items():
        if re.search(pat, normalized_text, flags=re.IGNORECASE):
            zones.append(z)
    rf["zones_involved"] = zones or None

    # distribution
    dist = None
    for d, pat in DISTRIBUTION_PATTERNS.items():
        if re.search(pat, normalized_text, flags=re.IGNORECASE):
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
    if rf.get("ground_glass_opacity") is True:
        features.append("ground-glass opacities")
    if rf.get("consolidation") is True:
        features.append("consolidation")
    if rf.get("reticular_thickening") is True:
        features.append("reticular interstitial thickening")
    # Construct findings sentence
    if features:
        parts.append(f"Chest radiograph shows {', '.join(features)}" + (f" with {', '.join(locs)}." if locs else "."))
    else:
        # If all major features are explicitly False, report no acute airspace abnormality
        major_flags = [rf.get("ground_glass_opacity"), rf.get("consolidation"), rf.get("reticular_thickening")]
        if all(flag is False for flag in major_flags):
            parts.append("Chest radiograph shows no definite acute airspace abnormality by report.")
        else:
            # indeterminate if None or mixed -> use cautious phrasing
            parts.append("Chest radiograph report does not clearly state acute airspace abnormality.")

    # Pleural effusion messaging
    if rf.get("pleural_effusion") is True:
        parts.append("Pleural effusion noted.")
    elif rf.get("pleural_effusion") is False:
        parts.append("No pleural effusion.")

    # Pneumothorax messaging
    if rf.get("pneumothorax") is True:
        parts.append("Pneumothorax present.")
    elif rf.get("pneumothorax") is False:
        parts.append("No pneumothorax.")

    parts.append(f"Severity estimate (rule-based): Left {left_score}/4, Right {right_score}/4; Total {total}/8.")
    parts.append(f"Mapped severity: {mapped} ({mapped_label}).")
    parts.append("Recommend correlation with clinical status and laboratory markers.")

    return " ".join(parts)