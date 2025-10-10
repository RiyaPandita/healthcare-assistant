# app.py
import streamlit as st
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from agents.ingestion_agent import IngestionAgent
from agents.imaging_agent import ImagingAgent
from agents.therapy_agent import TherapyAgent
from agents.pharmacy_agent import PharmacyAgent
from agents.doctor_agent import DoctorAgent
from agents.coordinator import Coordinator
from utils.config import Config

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is not set. Please set it in your environment variables or Streamlit secrets.")
    st.stop()


st.set_page_config(page_title="Healthcare Assistant (Demo)", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: transparent !important;
    }
    .block-container {
        padding-top: 1rem !important;
        background-color: transparent !important;
    }
    .stMarkdown {
        background-color: transparent !important;
    }
    .disclaimer {
        padding: 1rem;
        background-color: rgba(248, 249, 250, 0.8);
        border-radius: 5px;
        margin-bottom: 2rem;
        border: 1px solid rgba(0,0,0,0.1);
    }
    .timeline-event {
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 6px;
        background-color: #f8f9fa;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .timeline-event:hover {
        transform: translateX(4px);
    }
    .timeline-event small {
        color: #6c757d;
        font-size: 0.85em;
    }
    .timeline-event strong {
        color: #2c3e50;
        display: block;
        margin: 4px 0;
    }
    .timeline-event.success {
        border-left: 4px solid #28a745;
        background-color: rgba(39, 174, 96, 0.1);
    }
    .timeline-event.warning {
        border-left-color: #F1C40F;
        background-color: rgba(241, 196, 15, 0.1);
    }
    .timeline-event.error {
        border-left-color: #E74C3C;
        background-color: rgba(231, 76, 60, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header section with prominent disclaimer
st.title("🏥 Healthcare Assistant")
with st.container():
    st.markdown("""
        <div class='disclaimer'>
        <h3 style="color: red;">⚠️ Important Disclaimer</h3>
        <p style="color: red;">This is an <b>educational demonstration only</b>. Not intended for real medical use.
        Always consult qualified healthcare professionals for medical advice.</p>
        </div>
    """, unsafe_allow_html=True)

# Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY",""))

# Main content in tabs
tab1, tab2 = st.tabs(["📋 New Consultation", "ℹ️ About"])

# About tab content
with tab2:
    st.markdown("""
    ## About This Healthcare Assistant

    This is an educational demonstration of a multi-agent healthcare system designed to showcase 
    the potential of AI-assisted healthcare workflows.

    ### 🔑 Key Features
    - **Initial Assessment**: Upload and analysis of chest X-rays and medical documents
    - **Smart Triage**: Automated analysis of potential conditions
    - **Safe Recommendations**: Non-prescription (OTC) medication suggestions only
    - **Pharmacy Integration**: Real-time matching with nearby pharmacies
    - **Doctor Connect**: Optional telemedicine consultation routing
    
    ### 🛡️ Safety & Privacy
    - This is a **demonstration only** - not for real medical use
    - No personal health information (PHI) is stored
    - All uploads are treated as anonymous
    - Strict focus on over-the-counter medications only
    - Immediate escalation for serious symptoms
    
    ### 📝 How It Works
    1. Upload your chest X-ray and any supporting documents
    2. Provide basic information (age, allergies, symptoms)
    3. Receive an automated initial assessment
    4. Get matched with nearby pharmacies for OTC medications
    5. Option to connect with healthcare professionals if needed
    
    ### ⚠️ Limitations
    - Demonstration purposes only
    - No diagnostic claims
    - Limited to basic triage and OTC suggestions
    - Always consult healthcare professionals for medical advice
    """)

# Main consultation tab
with tab1:
    # Upload section with better organization
    with st.form("upload_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            allergies = st.text_input("Known Allergies (comma-separated)", value="ibuprofen", 
                                    help="List any known allergies, separated by commas")
            notes = st.text_area("Symptoms & Notes", value="cough, low-grade fever",
                               help="Describe current symptoms and relevant medical history")
            
        with col2:
            st.subheader("Documents & Location")
            xray_file = st.file_uploader("Upload Chest X-ray", type=["png","jpg","jpeg"],
                                       help="Upload a clear chest X-ray image")
            pdf_file = st.file_uploader("Additional Medical Documents (Optional)", type=["pdf"],
                                      help="Upload any relevant medical reports or documents")
            pincode = st.text_input("Delivery Pincode", value="400053",
                                  help="Enter delivery location pincode for pharmacy matching")
        
        st.markdown("---")
        col1, col2, col3 = st.columns([2,1,2])
        with col2:
            submitted = st.form_submit_button("🔄 Start Assessment", use_container_width=True)

if submitted:
    if not xray_file:
        st.error("Please upload an X-ray image.")
        st.stop()

    os.makedirs("uploads", exist_ok=True)
    xray_path = os.path.join("uploads", xray_file.name)
    with open(xray_path, "wb") as f: f.write(xray_file.getbuffer())

    pdf_path = None
    if pdf_file:
        pdf_path = os.path.join("uploads", pdf_file.name)
        with open(pdf_path, "wb") as f: f.write(pdf_file.getbuffer())

    # Load configuration
    config = Config()
    
    # Initialize agents with configuration-based paths
    ingestion = IngestionAgent()
    imaging = ImagingAgent()
    therapy = TherapyAgent(
        meds_path=config.settings['paths']['data']['meds'],
        interactions_path=config.settings['paths']['data']['interactions'],
        api_key=GEMINI_API_KEY
    )
    pharmacy = PharmacyAgent(
        pharmacies_path=config.settings['paths']['data']['pharmacies'],
        inventory_path=config.settings['paths']['data']['inventory'],
        zipcodes_path=config.settings['paths']['data']['zipcodes']
    )
    doctor = DoctorAgent(
        doctors_path=config.settings['paths']['data']['doctors'],
        api_key=GEMINI_API_KEY
    )
    coord = Coordinator(ingestion, imaging, therapy, pharmacy, doctor)

    payload = {
        "patient": {"age": int(age), "allergies": [a.strip() for a in allergies.split(",") if a.strip()]},
        "xray_path": xray_path,
        "pdf_path": pdf_path,
        "notes": notes,
        "pincode": pincode
    }

    with st.spinner("Processing..."):
        result = coord.run(payload)

    # Display results in an organized layout
    st.markdown("### 📊 Assessment Results")
    
    # Create three columns for the main results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Initial Analysis")
        imaging_result = result.get("imaging", {})
        if "condition_probs" in imaging_result:
            probs = imaging_result["condition_probs"]
            formatted_probs = {k: round(float(v), 2) for k, v in probs.items()}
            
            # Create a more visual representation of probabilities
            st.markdown("**Condition Analysis:**")
            for condition, prob in formatted_probs.items():
                st.progress(prob)
                st.markdown(f"*{condition.replace('_', ' ').title()}*: {prob*100:.1f}%")
            
            st.markdown(f"**Severity Assessment:** {imaging_result.get('severity_hint', 'N/A').title()}")

    with col2:
        st.markdown("#### Recommended Care")
        therapy_result = result.get("therapy", {})
        
        # Display red flags prominently if present
        red_flags = therapy_result.get("red_flags", [])
        if red_flags:
            st.error("⚠️ **Important Medical Alerts:**\n" + "\n".join([f"- {flag}" for flag in red_flags]))
        
        # Display OTC recommendations in a clean format
        if "otc_options" in therapy_result:
            st.markdown("**Suggested Over-the-Counter Options:**")
            for option in therapy_result["otc_options"]:
                with st.expander(f"💊 {option['drug_name']}"):
                    st.markdown(f"""
                    - **Dosage:** {option.get('dose', 'As per label')}
                    - **Frequency:** {option.get('freq', 'As per label')}
                    - **Product Code:** {option.get('sku', 'N/A')}
                    """)
                    if option.get('warnings'):
                        st.warning("⚠️ " + ", ".join(option['warnings']))

    # Display pharmacy and delivery information
    st.markdown("### 🏪 Pharmacy & Delivery")
    col1, col2 = st.columns(2)
    
    with col1:
        pharmacy_result = result.get("pharmacy", {})
        if pharmacy_result:
            st.markdown("""
            **Selected Pharmacy:**  
            📍 MedQuick Pharmacy  
            🕒 Estimated Delivery: {} minutes  
            💰 Delivery Fee: ₹{}
            """.format(
                int(pharmacy_result.get("eta_min", 0)),
                int(pharmacy_result.get("delivery_fee", 0))
            ))

    with col2:
        order_result = result.get("order", {})
        if order_result:
            # Generate a deterministic order ID from timestamp and pharmacy details
            from datetime import datetime
            timestamp = int(datetime.now().timestamp())
            pharmacy_id = pharmacy_result.get("pharmacy_id", "0")
            confirmation_id = f"ORD-{timestamp % 10000:04d}-{pharmacy_id[-3:]}"
            st.success(f"""
            ✅ **Order Confirmed!**  
            🔖 Reference ID: {confirmation_id}  
            📦 Status: Processing
            """)

    # Doctor escalation if needed
    if result.get("escalation", {}):
        st.warning("""
        👨‍⚕️ **Medical Consultation Recommended**
        
        Based on the assessment, we recommend speaking with a healthcare professional.
        A telemedicine consultation can be arranged with our partner doctors.
        """)
        
        doctor_info = result["escalation"].get("doctor", {})
        if doctor_info:
            st.info(f"""
            **Available Doctor:**  
            Dr. {doctor_info.get('name', 'N/A')}  
            Specialty: {doctor_info.get('specialty', 'General Medicine')}  
            Next Available Slot: {doctor_info.get('tele_slot_iso8601', 'Contact for scheduling')}
            """)

    # # Technical Details (Hidden by default)
    # with st.expander("� Technical Details", expanded=False):
    #     st.caption("System events and processing timeline for technical reference")
        
    #     # Filter and group events by status
    #     events = result.get("events", [])
    #     if events:
    #         # Group events by agent
    #         agent_events = {}
    #         for event in events:
    #             try:
    #                 agent = event.get('agent', 'system')
    #                 if agent not in agent_events:
    #                     agent_events[agent] = []
    #                 agent_events[agent].append(event)
    #             except Exception:
    #                 continue

    #         # Display events by agent in tabs
    #         if agent_events:
    #             agent_tabs = st.tabs([f"📊 {agent.title()}" for agent in agent_events.keys()])
    #             for tab, (agent, events) in zip(agent_tabs, agent_events.items()):
    #                 with tab:
    #                     for event in events:
    #                         try:
    #                             ts = datetime.fromisoformat(event.get('timestamp', '')).strftime('%H:%M:%S')
    #                             event_type = event.get('type', 'unknown')
    #                             data = event.get('data', {})
                                
    #                             # Determine event status for styling
    #                             status = "success"
    #                             if "error" in event_type.lower() or "fail" in event_type.lower():
    #                                 status = "error"
    #                             elif "warning" in event_type.lower() or "alert" in event_type.lower():
    #                                 status = "warning"
                                
    #                             # Format event data
    #                             if isinstance(data, dict):
    #                                 data_summary = ', '.join(f"**{k}**: {v}" for k, v in data.items() 
    #                                                     if not isinstance(v, (dict, list)))
    #                             else:
    #                                 data_summary = str(data)
                                
    #                             # Display styled event
    #                             st.markdown(f"""
    #                                 <div class="timeline-event {status}">
    #                                     <small>{ts}</small><br>
    #                                     <strong>{event_type}</strong><br>
    #                                     {data_summary}
    #                                 </div>
    #                             """, unsafe_allow_html=True)
    #                         except Exception:
    #                             continue

    # Final disclaimer
    st.markdown("---")
    st.info("🔔 This is an educational demonstration. All recommendations should be verified with qualified healthcare professionals.")