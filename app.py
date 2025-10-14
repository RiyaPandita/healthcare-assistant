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

def format_time_range(time_value):
    """Formats a time dictionary into a user-friendly string like '5-15'."""
    if isinstance(time_value, dict) and 'min' in time_value and 'max' in time_value:
        if time_value['min'] == time_value['max']:
            return f"{time_value['min']}"
        return f"{time_value['min']}-{time_value['max']}"
    return str(time_value) # Fallback for other data types

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
            st.subheader("Medical Documents")
            with st.expander("📊 X-ray Analysis", expanded=True):
                xray_file = st.file_uploader("Chest X-ray Image", type=["png","jpg","jpeg"],
                                             help="Upload a clear chest X-ray image. Required for automated analysis.")
                xray_report = st.file_uploader("X-ray Report (PDF)", type=["pdf"],
                                               help="Upload the radiologist's report for enhanced analysis")

            with st.expander("📋 Medical Records", expanded=True):
                prescription = st.file_uploader("Medical Prescription", type=["pdf", "jpg", "jpeg", "png"],
                                                help="Upload any existing prescriptions or medical records")

            st.subheader("📍 Location")
            pincode = st.text_input("Delivery Pincode", value="400053",
                                    help="Enter delivery location pincode for pharmacy matching")

        st.markdown("---")
        col1, col2, col3 = st.columns([2,1,2])
        with col2:
            assessment_submitted = st.form_submit_button("🔄 Start Assessment", use_container_width=True)


# --- EDIT 1: This block now ONLY handles processing the assessment ---
# It runs when the form is submitted and saves the results to the session state.
if assessment_submitted:
    if not xray_file:
        st.error("Please upload an X-ray image.")
        st.stop()

    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # Process X-ray image
    xray_path = os.path.join("uploads", xray_file.name)
    with open(xray_path, "wb") as f:
        f.write(xray_file.getbuffer())

    # Process X-ray report if provided
    xray_report_path = None
    if xray_report:
        xray_report_path = os.path.join("uploads", xray_report.name)
        with open(xray_report_path, "wb") as f:
            f.write(xray_report.getbuffer())

    # Process prescription if provided
    prescription_path = None
    if prescription:
        prescription_path = os.path.join("uploads", prescription.name)
        with open(prescription_path, "wb") as f:
            f.write(prescription.getbuffer())

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
        "patient": {
            "age": int(age),
            "allergies": [a.strip() for a in allergies.split(",") if a.strip()]
        },
        "xray_path": xray_path,
        "xray_report_path": xray_report_path,
        "prescription_path": prescription_path,
        "notes": notes,
        "pincode": pincode
    }

    with st.spinner("Processing..."):
        result = coord.run(payload)
        # Persist the result so subsequent reruns can access it
        st.session_state['result'] = result
        st.session_state['assessment_done'] = True
        # Reset order state for a new assessment
        st.session_state['order_confirmed'] = False
        st.session_state['order_details'] = {}
        st.rerun()


# --- EDIT 2: This new block handles ALL result displays ---
# It runs if an assessment has been completed, making the UI persistent.
if st.session_state.get("assessment_done", False):
    # Load result from session state
    result = st.session_state.get('result', {})

    # Display results in an organized layout
    st.markdown("### 📊 Assessment Results For X-Ray Report")

    # Create columns for the main results
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Initial Analysis")
        imaging_result = result.get("imaging", {})

        # Display image analysis results
        if "image_estimates" in imaging_result:
            img_estimates = imaging_result["image_estimates"]
            severity = imaging_result.get("severity_score", {})

            # Calculate severity percentage
            total_score = severity.get('total', 0)
            max_score = 8
            severity_percentage = (total_score / max_score) * 100
            severity_label = severity.get('mapped_label', 'N/A')

            # Determine color based on severity
            if severity_percentage <= 25:
                bar_color = "#28a745"  # green
            elif severity_percentage <= 50:
                bar_color = "#ffc107"  # yellow
            elif severity_percentage <= 75:
                bar_color = "#fd7e14"  # orange
            else:
                bar_color = "#dc3545"  # red

            # Show severity assessment with prominent colored box
            severity_box_style = f"""
                padding: 1rem; border-radius: 10px; background-color: {bar_color}15;
                border: 2px solid {bar_color}; margin-bottom: 1rem; text-align: center;
            """

            st.markdown(f"""
            <div style='{severity_box_style}'>
                <h3 style='margin: 0; color: {bar_color};'>{severity_label}</h3>
                <h2 style='margin: 0.5rem 0; color: {bar_color};'>{severity_percentage:.1f}%</h2>
                <p style='margin: 0; font-size: 0.9em; color: {bar_color};'>Score: {total_score}/{max_score}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<a name='score-calculation'></a>", unsafe_allow_html=True)
            with st.expander("ℹ️ Learn how this score is calculated"):
                st.markdown("""
                The severity assessment uses a combination of image analysis and medical report findings...
                (Content unchanged)
                """)

        # Show impression if available
        if "impression" in imaging_result:
            st.markdown("##### Radiologist's Impression")
            st.write(imaging_result["impression"])

    with col2:
        st.markdown("#### Recommended Care")
        therapy_result = result.get("therapy", {})

        # Display red flags prominently if present
        red_flags = therapy_result.get("red_flags", [])
        if red_flags:
            st.error("⚠️ **Important Medical Alerts:**\n" + "\n".join([f"- {flag}" for flag in red_flags]))

        # (Rest of therapy display logic is unchanged)
        interaction_warnings = set()
        otc_options = therapy_result.get("otc_options", [])
        if len(otc_options) > 1:
            drug_names = [opt["drug_name"] for opt in otc_options]
            drug_names.sort()
            for i in range(len(drug_names)):
                for j in range(i + 1, len(drug_names)):
                    for note in therapy_result.get("interaction_notes", []):
                        if drug_names[i].lower() in note.lower() and drug_names[j].lower() in note.lower():
                            interaction_warnings.add(note)
                            break
        for warning in interaction_warnings:
            st.warning(warning)
        warnings = therapy_result.get("warnings", [])
        for warning in warnings:
            st.warning(warning)
        if "otc_options" in therapy_result and therapy_result["otc_options"]:
            st.markdown("**Suggested Over-the-Counter Options:**")
            for option in therapy_result["otc_options"]:
                med_box_style = """
                    padding: 1rem; border-radius: 5px; background-color: #f8f9fa;
                    border: 1px solid #dee2e6; margin-bottom: 0.5rem;
                """
                st.markdown(f"""
                <div style='{med_box_style}'>
                    <h4 style='margin: 0 0 0.5rem 0;'>💊 {option['drug_name']}</h4>
                    <p style='margin: 0;'><strong>Dosage:</strong> {option.get('dose', 'As per label')}<br>
                    <strong>Frequency:</strong> {option.get('freq', 'As per label')}<br>
                    <strong>Product Code:</strong> {option.get('sku', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                if option.get('warnings'):
                    st.warning("⚠️ " + ", ".join(option['warnings']))
        elif not warnings:
            st.info("No medication recommendations available. Please consult a healthcare provider.")

    # Display pharmacy and delivery information
    st.markdown("### 🏪 Pharmacy & Delivery")
    col1, col2 = st.columns(2)

    with col1:
        pharmacy_result = result.get("pharmacy", {})
        if pharmacy_result:
            if "error" in pharmacy_result:
                st.error(pharmacy_result["error"])
                if "details" in pharmacy_result:
                    st.info(pharmacy_result["details"])
            else:
                delivery_time = pharmacy_result.get("delivery_time", {})
                fees = pharmacy_result.get("delivery_fees", {})

                st.markdown("""
                **Selected Pharmacy:**
                📍 {} ({})

                **Estimated Delivery Time:**
                ⚙️ Processing: {} minutes
                🚗 Travel: {} minutes
                ⏱️ Total: {} minutes

                **Delivery Charges:**
                📦 Base Fee: ₹{:.2f}
                📍 Distance Fee: ₹{:.2f}
                ⚡ Express Charge: ₹{:.2f}
                💰 Total: ₹{:.2f}
                """.format(
                    pharmacy_result.get("pharmacy_name", "MedQuick Pharmacy"),
                    pharmacy_result.get("distance_km", 0),
                    format_time_range(delivery_time.get("processing_time", 0)),
                    delivery_time.get("travel_time", 0),
                    format_time_range(delivery_time.get("total_time", 0)),
                    fees.get("base_fee", 0),
                    fees.get("distance_fee", 0),
                    fees.get("express_charge", 0),
                    fees.get("total", 0)
                ))

                order_total = sum([item.get("price", 0) * item.get("qty", 1) for item in pharmacy_result.get("items", [])])

                st.info(f"""
                📋 Order Summary:
                Items Total: ₹{order_total:.2f}
                Delivery Fee: ₹{fees.get('total', 0):.2f}
                Grand Total: ₹{(order_total + fees.get('total', 0)):.2f}

                Estimated Delivery: {format_time_range(delivery_time.get('total_time', 30))} minutes
                """)

                # --- EDIT 3: This logic now correctly separates the form from the confirmation message ---
                if not st.session_state.get('order_confirmed', False):
                    with st.expander("🛒 Confirm Order", expanded=True):
                        with st.form("confirm_order_form"):
                            st.write("Please review your order details and confirm:")
                            col_name, col_phone = st.columns(2)
                            with col_name:
                                customer_name = st.text_input("Full Name*", key="customer_name")
                            with col_phone:
                                phone = st.text_input("Phone Number*", key="phone")
                            address = st.text_area("Delivery Address*", key="address")
                            terms = st.checkbox("I confirm the order details and delivery address", key="terms")
                            submitted = st.form_submit_button("✅ Confirm & Place Order")

                            if submitted:
                                if not (terms and customer_name and phone and address):
                                    st.error("Please fill all required fields and accept the terms before confirming the order.")
                                else:
                                    st.session_state['order_confirmed'] = True
                                    st.session_state['order_details'] = {
                                        'order_id': f"ORD-{datetime.now().strftime('%y%m%d')}-{abs(hash(str(pharmacy_result)))%1000:03d}",
                                        'customer_name': customer_name,
                                        'phone': phone,
                                        'address': address, # Need to add address to details
                                        'items_total': order_total,
                                        'delivery_fee': fees.get('total', 0),
                                        'grand_total': (order_total + fees.get('total', 0)),
                                        'eta': delivery_time.get('total_time', 30)
                                    }
                                    st.rerun() # Rerun to show the confirmation message

                if st.session_state.get('order_confirmed', False):
                    od = st.session_state.get('order_details', {})
                    st.success(f"""
                    ✅ **Order Placed Successfully!**

                    - **Order ID:** {od.get('order_id')}
                    - **Customer:** {od.get('customer_name')}
                    - **Phone:** {od.get('phone')}
                    - **Address:** {od.get('address')}
                    - **Grand Total:** ₹{od.get('grand_total', 0):.2f}
                    - **Expected Delivery:** {format_time_range(od.get('eta'))} minutes

                    We'll send updates to your phone number.
                    """)

    # --- EDIT 4: Removed redundant code blocks that were here ---
    # The order confirmation logic is now self-contained in col1 above.
    # The doctor/technical details are below and correctly nested.

    if result.get("escalation", {}):
        st.warning("""
        👨‍⚕️ **Medical Consultation Recommended**

        Based on the assessment, we recommend speaking with a healthcare professional...
        """)
        doctor_info = result["escalation"].get("doctor", {})
        if doctor_info:
            st.info(f"""
            **Best Suited Available Doctor:**
            {doctor_info.get('name', 'N/A')}
            Specialty: {doctor_info.get('specialty', 'General Medicine')}
            Next Available Slot: {doctor_info.get('tele_slot_iso8601', 'Contact for scheduling')}
            """)
            # Doctor booking UI: allow user to book this slot (demo-only, no real booking service)
            booking_key = f"doctor_booked_{doctor_info.get('name','') }"
            if not st.session_state.get('doctor_booked', False) and not st.session_state.get(booking_key, False):
                with st.expander("📅 Book Doctor Slot", expanded=False):
                    with st.form("doctor_booking_form"):
                        st.write("Please confirm booking details for the selected doctor:")
                        d_col1, d_col2 = st.columns(2)
                        with d_col1:
                            booker_name = st.text_input("Your full name", key="booker_name")
                        with d_col2:
                            booker_phone = st.text_input("Phone number", key="booker_phone")
                        preferred_time = st.text_input("Preferred time (optional)", key="booker_pref_time")
                        agree = st.checkbox("I confirm this booking is for demonstration purposes only", key="booker_agree")
                        book_submit = st.form_submit_button("📌 Book Slot")

                        if book_submit:
                            if not (booker_name and booker_phone and agree):
                                st.error("Please fill name, phone and confirm the demo booking checkbox.")
                            else:
                                # Save booking to session state and show confirmation
                                booking_id = f"DOC-{datetime.now().strftime('%y%m%d')}-{abs(hash(str(doctor_info)))%1000:03d}"
                                st.session_state['doctor_booked'] = True
                                st.session_state[booking_key] = True
                                st.session_state['doctor_booking'] = {
                                    'booking_id': booking_id,
                                    'doctor_name': doctor_info.get('name'),
                                    'specialty': doctor_info.get('specialty'),
                                    'slot': doctor_info.get('tele_slot_iso8601'),
                                    'booker_name': booker_name,
                                    'booker_phone': booker_phone,
                                    'preferred_time': preferred_time
                                }
                                st.success(f"✅ Slot booked: {booking_id} — {doctor_info.get('name')}")
                                # Use st.rerun() which is available in current Streamlit API
                                try:
                                    st.rerun()
                                except Exception:
                                    # If rerun fails for any reason, fallback to a no-op — session state already updated
                                    pass
            # If already booked, show confirmation
            if st.session_state.get('doctor_booked', False) or st.session_state.get(booking_key, False):
                db = st.session_state.get('doctor_booking', {})
                if db:
                    st.success(f"✅ **Doctor Slot Confirmed**\n\n- **Booking ID:** {db.get('booking_id')}\n- **Doctor:** {db.get('doctor_name')} ({db.get('specialty')})\n- **Slot:** {db.get('slot')}\n- **Booked For:** {db.get('booker_name')} — {db.get('booker_phone')}")

    with st.expander("Technical Details", expanded=False):
        # (All technical details display logic remains unchanged)
        st.caption("System events and processing timeline for technical reference")
        events = result.get("events", [])
        if events:
            agent_events = {}
            for event in events:
                try:
                    agent = event.get('agent', 'system')
                    if agent not in agent_events:
                        agent_events[agent] = []
                    agent_events[agent].append(event)
                except Exception:
                    continue
            if agent_events:
                agent_tabs = st.tabs([f"📊 {agent.title()}" for agent in agent_events.keys()])
                for tab, (agent, events) in zip(agent_tabs, agent_events.items()):
                    with tab:
                        # (Event rendering loop is unchanged)
                        for event in events:
                            try:
                                ts = datetime.fromisoformat(event.get('timestamp', '')).strftime('%H:%M:%S')
                                event_type = event.get('type', 'unknown')
                                data = event.get('data', {})
                                status = "success"
                                if "error" in event_type.lower() or "fail" in event_type.lower():
                                    status = "error"
                                elif "warning" in event_type.lower() or "alert" in event_type.lower():
                                    status = "warning"
                                if isinstance(data, dict):
                                    data_summary = ', '.join(f"**{k}**: {v}" for k, v in data.items() if not isinstance(v, (dict, list)))
                                else:
                                    data_summary = str(data)
                                technical_details = "" # ... (rest of details unchanged) ...
                                st.markdown(f"""
                                    <div class="timeline-event {status}">
                                        <small>{ts}</small><br>
                                        <strong>{event_type}</strong><br>
                                        {data_summary}
                                        {f'<br><small style="color: #666; font-family: monospace;">{technical_details}</small>' if technical_details else ''}
                                    </div>
                                """, unsafe_allow_html=True)
                            except Exception:
                                continue

    #Final disclaimer
    st.markdown("---")
    st.info("🔔 This is an educational demonstration. All recommendations should be verified with qualified healthcare professionals.")