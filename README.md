# Diabetic-Retinopathy-Detection
For support or inquiries, please contact us at [Zainab Fatima](mailto:zenebb.19@gmail.com) or [Laiba Masood](mailto:laibamasood24@gmail.com).

## Overview
This project aims to detect diabetic retinopathy (DR) using deep learning techniques, specifically **EfficientNet-B2**, to classify fundus images into five severity levels. The system includes a user-friendly Streamlit web interface for medical professionals to upload and analyze retinal images.

## Features
* **Preprocessing**: Image resizing, normalization, and augmentation techniques to improve model performance
* **Model**: Transfer learning with EfficientNet-B2 pre-trained architecture
* **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler to fine-tune learning rates
* **Evaluation**: F1-score, Precision, Recall, and Grad-CAM visualization for explainability
* **Web Interface**: Interactive Streamlit application for easy image uploading and analysis
* **Deployment**: Fully deployed solution accessible via web browser

## Installation
Run the following command to install dependencies:
```
pip install -r requirements.txt
```

## Running the Application
Launch the Streamlit interface locally with:
```
streamlit run app.py
```

## Deployment
The application is currently deployed and accessible at (https://diabetic-retinopathy-detection-5nz6go26hhgypbvx3hsq8g.streamlit.app/). Medical professionals can access the system through any modern web browser without additional setup.

## Usage
1. Access the deployed application or run it locally
2. Upload a fundus image through the web interface
3. The system will automatically analyze the image and classify it into one of five DR severity levels
4. Review the confidence scores and recommendations based on the classification

## Diabetic Retinopathy Classification Levels
* **No DR**: No visible signs of diabetic retinopathy
* **Mild DR**: Microaneurysms only
* **Moderate DR**: More microaneurysms, dot and blot hemorrhages, cotton wool spots
* **Severe DR**: More than 20 intraretinal hemorrhages, venous beading, intraretinal microvascular abnormalities
* **Proliferative DR**: Neovascularization, vitreous hemorrhage, or retinal detachment

## Technical Implementation
The core of the system leverages a fine-tuned EfficientNet-B2 model, which was trained on a balanced dataset of retinal images. The Streamlit interface provides an intuitive way for medical professionals to interact with the model, upload images, and receive rapid classification results with appropriate medical recommendations.
