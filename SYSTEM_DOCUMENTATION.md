# Healthcare Assistant System Documentation

## System Overview

The Healthcare Assistant is a multi-agent system designed to provide preliminary medical assessment and recommendations using AI and data integration. The system demonstrates how AI could assist in healthcare workflows while maintaining strict safety boundaries.

## Core Components

### 1. Main Application (app.py)
- **UI Framework**: Built with Streamlit
- **Layout**: Two-tab interface (New Consultation and About)
- **Safety Features**: Prominent red disclaimers and medical warnings

### 2. Agent Architecture

#### 2.1 Ingestion Agent
- Handles document processing
- Processes X-ray images and PDF documents
- Prepares data for other agents

#### 2.2 Imaging Agent
- Analyzes chest X-rays
- Provides condition probabilities
- Determines severity levels
- Uses AI models for image analysis
- Returns structured analysis with:
  * Condition probabilities
  * Severity assessment
  * Key observations

#### 2.3 Therapy Agent
Function:
- Processes medical analysis
- Suggests OTC medications
- Checks drug interactions
- Identifies red flags

Logic:
- Uses `meds.csv` for medication database
- Checks `interactions.csv` for drug conflicts
- Considers:
  * Patient age
  * Allergies
  * Symptoms
  * Condition severity
  * Drug interactions
- Uses Gemini AI for advanced reasoning

#### 2.4 Pharmacy Agent
Function:
- Locates nearby pharmacies
- Checks medication availability
- Calculates delivery estimates

Logic:
- Uses `pharmacies.json` for store database
- Checks `inventory.csv` for stock
- Uses `zipcodes.csv` for location mapping
- Calculates ETA based on:
  * Distance from pharmacy
  * Traffic conditions
  * Time of day
- Determines delivery fees based on:
  * Distance
  * Order value
  * Time of delivery

#### 2.5 Doctor Agent
Function:
- Manages doctor recommendations
- Handles escalation cases
- Provides telemedicine options

Logic:
- Uses `doctors.csv` for physician database
- Recommends doctors based on:
  * Specialty matching condition
  * Availability
  * Location proximity
- Manages scheduling slots

### 3. Coordinator
- Orchestrates all agent interactions
- Manages workflow sequence
- Handles error cases
- Ensures system coherence

## Workflow Logic

1. **Initial Assessment**
   - User inputs:
     * Age
     * Allergies
     * Symptoms
     * X-ray image
     * Additional documents
     * Location (pincode)

2. **Processing Pipeline**
   ```
   User Input → Ingestion → Imaging Analysis → Therapy Assessment → Pharmacy Matching → Doctor Recommendation (if needed)
   ```

3. **Decision Making**
   a. **Condition Assessment**
      - AI analysis of X-rays
      - Symptom evaluation
      - Risk factor analysis
   
   b. **Medication Recommendations**
      - Based on:
        * Condition probability
        * Symptom severity
        * Patient allergies
        * Drug interactions
        * Age considerations
      - Limited to OTC medications only
   
   c. **Pharmacy Selection**
      - Criteria:
        * Stock availability
        * Distance from patient
        * Operating hours
        * Delivery capability
   
   d. **Doctor Escalation**
      - Triggered by:
        * High-risk conditions
        * Severe symptoms
        * Complex medical history
        * Multiple drug interactions

4. **Safety Mechanisms**
   - Red flag detection
   - Drug interaction checking
   - Severity assessment
   - Automatic escalation
   - OTC-only limitations

## Data Sources

1. **Medication Database** (meds.csv)
   - Drug names
   - Categories
   - Dosage information
   - OTC status
   - Contraindications

2. **Pharmacy Data** (pharmacies.json)
   - Store locations
   - Operating hours
   - Delivery zones
   - Contact information

3. **Inventory Management** (inventory.csv)
   - Stock levels
   - SKU information
   - Price data
   - Availability status

4. **Doctor Directory** (doctors.csv)
   - Physician profiles
   - Specialties
   - Availability
   - Telemedicine capability

5. **Geographic Data** (zipcodes.csv)
   - Location mapping
   - Delivery zones
   - Distance calculations

## API Integration

1. **Gemini AI**
   - Used for:
     * Advanced medical reasoning
     * Natural language processing
     * Decision support
     * Risk assessment

## Safety and Privacy Features

1. **Data Handling**
   - No PHI storage
   - Anonymous processing
   - Temporary file handling
   - Secure API communications

2. **Medical Safety**
   - OTC medication limits
   - Drug interaction checks
   - Automatic escalation
   - Clear disclaimers
   - Professional referral system

## Limitations and Boundaries

1. **Medical Scope**
   - Educational demonstration only
   - No diagnostic capabilities
   - Limited to preliminary assessment
   - OTC medications only

2. **Technical Boundaries**
   - Basic image analysis
   - Limited condition coverage
   - Simplified pharmacy matching
   - Basic telemedicine integration

## Future Enhancement Areas

1. **Medical Capabilities**
   - Enhanced image analysis
   - Broader condition coverage
   - More sophisticated drug interaction checks
   - Advanced symptom analysis

2. **Technical Features**
   - Real-time pharmacy inventory
   - Advanced delivery tracking
   - Integrated telemedicine platform
   - Electronic health record integration

3. **User Experience**
   - Multiple language support
   - Accessibility features
   - Mobile optimization
   - Offline capabilities

## Best Practices for Use

1. **Input Quality**
   - Clear X-ray images
   - Complete symptom information
   - Accurate medical history
   - Updated contact details

2. **Safety Guidelines**
   - Verify all recommendations
   - Follow up with professionals
   - Report any issues
   - Monitor for updates

## System Requirements

1. **Technical Requirements**
   - Python 3.7+
   - Required libraries (streamlit, pandas, etc.)
   - Internet connectivity
   - Sufficient storage for uploads

2. **API Requirements**
   - Valid Gemini API key
   - Stable internet connection
   - Adequate API quota

## Example Trace Flow

Below is a detailed trace of the system's execution flow based on a sample input:

### Input Data
```python
{
    "age": 85,
    "allergies": ["sore throat", "runny nose"],
    "symptoms": "high fever, cough, throat pain",
    "xray_file": "chest-xray-2.jpg",  # 4.3KB
    "additional_docs": "prescep.pdf",  # 239.3KB
    "pincode": "500081"
}
```

### Complete Flow Trace

1. **Initial Form Submission** (`app.py`)
```python
payload = {
    "patient": {
        "age": int(age),  # 85
        "allergies": [a.strip() for a in allergies.split(",")]
    },
    "xray_path": xray_path,
    "pdf_path": pdf_path,
    "notes": notes,
    "pincode": pincode
}
result = coord.run(payload)  # Main entry point
```

2. **Document Processing** (`agents/ingestion_agent.py`)
```python
class IngestionAgent:
    def process(self, xray_path, pdf_path):
        # Process X-ray image
        xray_data = self._process_image(xray_path)
        # Process optional PDF
        pdf_data = self._process_pdf(pdf_path) if pdf_path else None
        return {"xray": xray_data, "pdf": pdf_data}
```

3. **Image Analysis** (`agents/imaging_agent.py`)
```python
class ImagingAgent:
    def analyze(self, xray_data):
        # AI model analysis
        results = self._run_analysis(xray_data)
        severity = self._assess_severity(results)
        return {
            "condition_probs": {
                "pneumonia": 0.85,
                "bronchitis": 0.65
            },
            "severity_hint": "moderate"
        }
```

4. **Therapy Assessment** (`agents/therapy_agent.py`)
```python
class TherapyAgent:
    def recommend(self, analysis, patient_data):
        # Check age-related risks (age 85 is high-risk)
        risks = self._assess_risks(patient_data["age"])
        
        # Evaluate symptoms
        symptoms = self._analyze_symptoms([
            "high fever", "cough", "throat pain"
        ])
        
        # Generate recommendations
        return {
            "red_flags": [
                "Age over 65 with high fever",
                "Multiple respiratory symptoms"
            ],
            "otc_options": [
                {
                    "drug_name": "Acetaminophen",
                    "dose": "500mg",
                    "freq": "Every 6 hours",
                    "warnings": ["Do not exceed 3000mg/day"]
                }
            ]
        }
```

5. **Pharmacy Matching** (`agents/pharmacy_agent.py`)
```python
class PharmacyAgent:
    def find_pharmacy(self, pincode, medications):
        # Find nearest pharmacy in 500081 area
        nearby = self._query_pharmacies(pincode)
        # Check medication availability
        available = self._check_inventory(nearby[0], medications)
        
        return {
            "pharmacy_id": "PH123",
            "eta_min": 30,
            "delivery_fee": 50
        }
```

6. **Doctor Escalation** (`agents/doctor_agent.py`)
```python
class DoctorAgent:
    def evaluate_escalation(self, analysis, patient_data):
        # Age 85 + severe symptoms triggers escalation
        if self._needs_escalation(analysis, patient_data):
            return {
                "doctor": {
                    "name": "Dr. Smith",
                    "specialty": "Pulmonology",
                    "tele_slot_iso8601": "2025-10-10T15:30:00Z"
                }
            }
```

### Result Processing

The coordinator (`agents/coordinator.py`) aggregates all agent responses:

```python
class Coordinator:
    def run(self, payload):
        # Sequential processing
        ingestion_result = self.ingestion.process(...)
        imaging_result = self.imaging.analyze(...)
        therapy_result = self.therapy.recommend(...)
        pharmacy_result = self.pharmacy.find_pharmacy(...)
        doctor_result = self.doctor.evaluate_escalation(...)
        
        return {
            "imaging": imaging_result,
            "therapy": therapy_result,
            "pharmacy": pharmacy_result,
            "escalation": doctor_result
        }
```

### Final Output Display (`app.py`)

1. Shows condition probabilities with progress bars
2. Displays OTC medication recommendations
3. Shows pharmacy delivery details
4. Presents doctor escalation recommendation
5. All wrapped in safety disclaimers

### Key Decision Points

1. **High-Risk Factors Identified**:
   - Age (85 years)
   - High fever
   - Multiple respiratory symptoms

2. **Safety Measures Triggered**:
   - Automatic doctor escalation
   - Limited OTC recommendations
   - Clear warning displays

3. **Location-Based Services**:
   - Pharmacy in 500081 area
   - Delivery time estimation
   - Fee calculation

This example demonstrates how the system processes a complex medical case while maintaining safety boundaries and providing appropriate escalation paths.

This documentation provides a comprehensive overview of the Healthcare Assistant system, its components, logic, and safety measures. It serves as a reference for understanding the system's capabilities and limitations while emphasizing its educational and demonstrative nature.