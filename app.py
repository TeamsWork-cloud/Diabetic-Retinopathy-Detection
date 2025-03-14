import streamlit as st
from PIL import Image
import torch
import datetime

# Import custom modules
from model import load_model, predict

# Set device
device = torch.device("cpu")

# Page configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Classification",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for the download button margin
st.markdown("""
<style>
    .download-button-container {
        margin-top: 1rem;
    }
    .sidebar-info {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model once and cache
@st.cache_resource
def get_model():
    model_path = r".\model\final_dr_model.pt"
    model, _ = load_model(model_path, device=device)  # Unpack correctly
    return model

# Sidebar for terminology explanation
with st.sidebar:
    st.header("Understanding the Terms")
    
    # Explanation of DR classifications
    st.subheader("Diabetic Retinopathy Stages")
    classifications = {
        "No DR": "No visible signs of diabetic retinopathy",
        "Mild DR": "Small areas of balloon-like swelling in the retina's blood vessels",
        "Moderate DR": "More extensive damage to blood vessels, affecting blood supply to retina",
        "Severe DR": "Many blocked blood vessels, signaling the retina to grow new blood vessels",
        "Proliferative DR": "Advanced stage with abnormal new blood vessels growing on the retina"
    }
    
    for term, explanation in classifications.items():
        with st.expander(term):
            st.write(explanation)
    
    # What is DR?
    with st.expander("What is Diabetic Retinopathy?"):
        st.write("Diabetic retinopathy is an eye condition that can cause vision loss and blindness in people with diabetes. It affects blood vessels in the retina (the light-sensitive layer at the back of the eye).")

# Load the model
try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Main content
st.title("Diabetic Retinopathy Classification 🩺")
st.write("Upload an image of a retina to classify the severity of diabetic retinopathy.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed deprecated parameter
        
        # Perform prediction
        CLASSES = ["No DR ✅", "Mild DR 🟡", "Moderate DR 🟠", "Severe DR 🔴", "Proliferative DR ⚠️"]
        predicted_class, confidence, probabilities = predict(model, image, transform=None, 
                                                           classes=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
        
        # Display results
        st.subheader("Prediction Results")
        st.success(f"**Predicted Class:** {CLASSES[predicted_class]} | **Confidence:** {confidence:.2f}%")
        
        # Severity-Based Recommendations with color coding
        recommendations = {
            0: ("No Diabetic Retinopathy ✅", "Your retina appears healthy. Maintain regular eye checkups."),
            1: ("Mild Diabetic Retinopathy 🟡", "Early-stage detected. Consult an ophthalmologist."),
            2: ("Moderate Diabetic Retinopathy 🟠", "Schedule an appointment and manage blood sugar strictly."),
            3: ("Severe Diabetic Retinopathy 🔴", "Seek urgent medical attention and consider treatment."),
            4: ("Proliferative Diabetic Retinopathy ⚠️", "Urgent action required—consult an ophthalmologist immediately.")
        }
        
        # Color codes for different severity levels
        severity_colors = {
            0: "#28a745",  # Green for No DR
            1: "#ffc107",  # Yellow for Mild DR
            2: "#fd7e14",  # Orange for Moderate DR
            3: "#dc3545",  # Red for Severe DR
            4: "#6c1c1c"   # Dark red for Proliferative DR
        }
        
        # Display recommendation with color based on severity
        st.markdown(
            f"""
            <div style="background-color: {severity_colors[predicted_class]}20; 
                       padding: 1rem; 
                       border-radius: 0.5rem; 
                       border-left: 5px solid {severity_colors[predicted_class]};">
                <h3 style="color: {severity_colors[predicted_class]};">{recommendations[predicted_class][0]}</h3>
                <p>{recommendations[predicted_class][1]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create downloadable report
        report_text = f"""
        DIABETIC RETINOPATHY ASSESSMENT REPORT
        Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
        
        RESULT: {CLASSES[predicted_class]}
        Confidence: {confidence:.2f}%
        
        RECOMMENDATION:
        {recommendations[predicted_class][0]}
        {recommendations[predicted_class][1]}
        
        DISCLAIMER:
        This result is generated by an AI system and is for informational purposes only.
        Please consult a healthcare professional for proper diagnosis and treatment.
        """
        
        # Download button with margin
        st.markdown('<div class="download-button-container">', unsafe_allow_html=True)
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"DR_Assessment_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.write("---")
        st.write("**Disclaimer:** This tool is for informational purposes only. Consult a healthcare professional for diagnosis.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
