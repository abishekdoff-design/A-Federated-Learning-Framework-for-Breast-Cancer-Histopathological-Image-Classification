# Federated Learning Framework for Breast Cancer Histopathological Image Classification

## Project Overview
This project presents a secure and scalable framework for the automated diagnosis of breast cancer from histopathological slides. By utilizing a **Federated Learning** approach, the system ensures medical data privacy by allowing models to be trained locally at different "nodes" (hospitals) without moving the original sensitive patient data to a central server.

## Technical Features
- **User Authentication:** Secure Registration and Login system using SQLite3.
- **Image Preprocessing:** Grayscale conversion and morphological resizing using OpenCV.
- **Feature Extraction:** Hybrid approach using **GLCM** (Gray-Level Co-occurrence Matrix) for texture analysis and **PCA** for dimensionality reduction.
- **AI Models:** Comparative analysis between:
  - Custom 2D Convolutional Neural Networks (CNN).
  - VGG19 (Transfer Learning).
  - Hybrid VGG-19 Architectures.
- **Web Interface:** Fully functional dashboard built with Streamlit.

## System Requirements
To run this project, you need Python 3.8+ installed along with the following libraries:
```bash
pip install streamlit opencv-python tensorflow scikit-learn scikit-image numpy matplotlib seaborn
