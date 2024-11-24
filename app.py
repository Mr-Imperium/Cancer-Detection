import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Cancer Image Detection",
    page_icon="🔬",
    layout="wide"
)

import os
import sys

# Initialize session state for package installation
if 'packages_installed' not in st.session_state:
    st.session_state.packages_installed = False

# Install required packages if not already installed
if not st.session_state.packages_installed:
    st.write("Installing required packages...")
    os.system(f"{sys.executable} -m pip install torch==2.2.0 torchvision==0.17.0")
    st.session_state.packages_installed = True

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io

# Model setup
@st.cache_resource
def load_model():
    model = models.resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        model.load_state_dict(torch.load('cancer_detection_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'cancer_detection_model.pth' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Main app
def main():
    st.title("Cancer Image Detection")
    st.write("Upload a medical image for cancer detection")
    
    # System info
    st.sidebar.title("System Information")
    st.sidebar.info(f"""
    - Python Version: {sys.version.split()[0]}
    - PyTorch Version: {torch.__version__}
    - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}
    """)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["tif"])
    
    if uploaded_file is not None:
        try:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    # Preprocess image
                    input_tensor = preprocess_image(image)
                    
                    # Get prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        predicted_class = torch.argmax(probabilities).item()
                        confidence = probabilities[predicted_class].item() * 100
                    
                    # Display results
                    result = "Cancer Detected" if predicted_class == 1 else "No Cancer Detected"
                    st.header(f"Result: {result}")
                    st.subheader(f"Confidence: {confidence:.2f}%")
                    
                    # Display probability bar chart
                    st.bar_chart({
                        "No Cancer": probabilities[0].item() * 100,
                        "Cancer": probabilities[1].item() * 100
                    })
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error("Stack trace:", exc_info=True)

if __name__ == "__main__":
    main()
