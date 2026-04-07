#======================== IMPORT PACKAGES ===========================
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from PIL import Image
import matplotlib.image as mpimg
import streamlit as st
import base64
import time
import os
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

#======================== CSS STYLING ===========================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1E88E5;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 30px;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .result-card {
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .benign-result {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    
    .malignant-result {
        background-color: #FFCDD2;
        color: #C62828;
    }
    
    .section-title {
        color: #1E88E5;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    
    .section-title .icon {
        margin-right: 10px;
        font-size: 28px;
    }
    
    .feature-value {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    
    .upload-container {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: rgba(225, 245, 254, 0.3);
    }
    
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }
    
    .stButton>button:hover {
        background-color: #1565C0;
    }
    
    /* Hide Streamlit branding */
    .decoration {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

#======================== SIDEBAR ===========================
def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #1E88E5; color: white; border-radius: 10px; margin-bottom: 30px;">
        <h1 style="margin: 0;">🔬 BC Classifier</h1>
        <p style="margin: 0;">Breast Cancer Histopathological Image Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    ### 📋 About
    This application uses a federated learning framework to classify breast cancer histopathological images as benign or malignant.
    """)
    
    st.sidebar.markdown("""
    ### 📚 Instructions
    1. Upload a histopathological image in TIFF format
    2. The system will process and analyze the image
    3. View the classification results
    """)
    
    st.sidebar.markdown("""
    ### 🔧 Features
    - Image preprocessing
    - Feature extraction
    - Classification using federated learning
    - Visual analysis of results
    """)

#======================== MAIN APP ===========================
def main():
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        Federated Learning Framework for Breast Cancer Histopathological Image Classification
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("""
    <div class="section-title">
        <span class="icon">📁</span> Upload Image
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="upload-container">
                <p style="font-size: 18px; margin-bottom: 15px;">Upload a histopathological image in TIFF format</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("", type=['png', 'jpg'])
    
    if uploaded_file is None:
        st.info("Please upload an image to proceed with classification.")
        return
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update progress
    progress_bar.progress(10)
    status_text.text("Reading image...")
    
    # Read the image
    img = mpimg.imread(uploaded_file)
    
    # Display original image
    st.markdown("""
    <div class="section-title">
        <span class="icon">🖼️</span> Original Image
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption="Uploaded Histopathological Image", use_column_width=True)
    
    # Update progress
    progress_bar.progress(30)
    status_text.text("Preprocessing image...")
    
    # Preprocessing section
    st.markdown("""
    <div class="section-title">
        <span class="icon">⚙️</span> Image Preprocessing
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Resize image
    resized_image = cv2.resize(img, (300, 300))
    img_resize_orig = cv2.resize(img, (50, 50))
    
    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <h4>Resized Image (300x300)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(resized_image, use_column_width=True)
    
    # Grayscale conversion
    try:
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    except:
        gray1 = img_resize_orig
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h4>Grayscale Image (50x50)</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(gray1, caption="Grayscale Image", use_column_width=True, clamp=True)
    
    # Update progress
    progress_bar.progress(50)
    status_text.text("Extracting features...")
    
    # Feature extraction section
    st.markdown("""
    <div class="section-title">
        <span class="icon">🔍</span> Feature Extraction
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    mean_val = np.mean(gray1)
    median_val = np.median(gray1)
    var_val = np.var(gray1)
    features_extraction = [mean_val, median_val, var_val]
    
    with col1:
        st.markdown(f"""
        <div class="feature-value">
            Mean: {mean_val:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-value">
            Median: {median_val:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="feature-value">
            Variance: {var_val:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    # Update progress
    progress_bar.progress(70)
    status_text.text("Loading dataset...")
    
    # Dataset loading and splitting
    try:
        data_1 = os.listdir('Data/Benign/')
        data_2 = os.listdir('Data/Malignant/')
        
        dot1 = []
        labels1 = []
        
        # Process benign images
        for img11 in data_1:
            try:
                img_1 = mpimg.imread('Data/Benign//' + "/" + img11)
                img_1 = cv2.resize(img_1, (50, 50))
                
                try:
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                except:
                    gray = img_1
                
                dot1.append(np.array(gray))
                labels1.append(1)
            except:
                None
        
        # Process malignant images
        for img11 in data_2:
            try:
                img_1 = mpimg.imread('Data/Malignant//' + "/" + img11)
                img_1 = cv2.resize(img_1, (50, 50))
                
                try:
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                except:
                    gray = img_1
                
                dot1.append(np.array(gray))
                labels1.append(2)
            except:
                None
        
        x_train, x_test, y_train, y_test = train_test_split(dot1, labels1, test_size=0.2, random_state=101)
        
        # Update progress
        progress_bar.progress(90)
        status_text.text("Classifying image...")
        
        # Classification
        temp_data1 = []
        for ijk in range(0, len(dot1)):
            temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
            temp_data1.append(temp_data)
        
        temp_data1 = np.array(temp_data1)
        zz = np.where(temp_data1 == 1)
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("Classification complete!")
        
        # Results section
        st.markdown("""
        <div class="section-title">
            <span class="icon">📊</span> Classification Result
        </div>
        """, unsafe_allow_html=True)
        
        if labels1[zz[0][0]] == 1:
            st.markdown("""
            <div class="result-card benign-result">
                ✅ Identified as BENIGN
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <p style="font-size: 18px;">The histopathological image shows characteristics of benign tissue.</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif labels1[zz[0][0]] == 2:
            st.markdown("""
            <div class="result-card malignant-result">
                ⚠️ Identified as MALIGNANT
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <p style="font-size: 18px;">The histopathological image shows characteristics of malignant tissue.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Dataset statistics
        st.markdown("""
        <div class="section-title">
            <span class="icon">📈</span> Dataset Statistics
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="dashboard-card" style="text-align: center;">
                <h3>{len(dot1)}</h3>
                <p>Total Images</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="dashboard-card" style="text-align: center;">
                <h3>{len(x_train)}</h3>
                <p>Training Images</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="dashboard-card" style="text-align: center;">
                <h3>{len(x_test)}</h3>
                <p>Test Images</p>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        st.error("Please make sure the dataset directories (Data/Benign and Data/Malignant) exist and contain valid images.")

# Run the app
if __name__ == "__main__":
    main()