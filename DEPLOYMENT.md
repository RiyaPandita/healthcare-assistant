Streamlit Cloud deployment notes

This project uses `pdf2image` and `pytesseract` as optional fallbacks for PDF OCR and scanned document processing. These libraries require system packages that are not installed by default on Streamlit Cloud.

System dependencies (must be installed on the host):
- poppler (provides `pdftoppm`) — needed by `pdf2image`
- tesseract-ocr — needed by `pytesseract`

On Debian/Ubuntu the packages are:

```bash
sudo apt-get update && sudo apt-get install -y poppler-utils tesseract-ocr
```

On Streamlit Cloud you cannot install system packages via apt during build. Options:
- Use pre-built Docker deployment on a platform where you control system packages.
- Avoid OCR fallback by ensuring uploaded PDFs contain selectable text (the code already tries pure-text extraction first).
- Add a small server-side utility to perform OCR in an environment with poppler/tesseract and call it from the app.

Runtime behavior in this repo:
- `models/xray_processor.extract_pdf_text` will attempt pure-text extraction first (pypdf / PyPDF2 / pdfminer).
- If those fail it attempts OCR fallback, but the imports for `pdf2image` and `pytesseract` are lazy and will be skipped with a logged warning if the system packages are not present. This avoids import-time failures on Streamlit Cloud.

Recommendation:
- If you plan to deploy on Streamlit Cloud and need OCR, host a small OCR microservice (e.g., on a VPS) or use a container-based deployment where you control system packages.
