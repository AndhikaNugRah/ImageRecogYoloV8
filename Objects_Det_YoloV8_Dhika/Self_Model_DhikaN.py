#1.Import Necessary Libraries

import streamlit as st  # Framework for building web apps
import PIL  # Python Imaging Library for image processing
from PIL import Image  # Import Image class specifically
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting images

from pathlib import Path  # For handling file paths
import torch  # PyTorch library for deep learning
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights  # Object detection model
from torchvision.utils import draw_bounding_boxes  # To draw bounding boxes on images

#2. Load Local Modules:
#These modules contain settings and helper functions
import settings
import helper

#3. Create Containers for Content:
main_container = st.container()  # Container for main app content
container = st.sidebar.container()  # Container for sidebar elements
container.empty()  # Clear the sidebar container

#4. Title and Image Upload:
with main_container:
    st.title("Objects Detector by Andhika Nugraha :student:")
    upload = st.file_uploader(label="Upload Your Image & Detect Here :", type=["png", "jpg", "jpeg"])

#5.Sidebar Elements
st.sidebar.header("Image Config")

# Confidence slider
numb_confidence = st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40, format="%d") / 100

# Task selection
selected_task = st.sidebar.radio(
    "Select Task",
    ['Detection', 'Segmentation'],
)

#6.Load Pre Trained Model
# Select model path based on task
if selected_task == 'Detection':
    path_model = Path(settings.DETECTION_MODEL)
elif selected_task == 'Segmentation':
    path_model = Path(settings.SEGMENTATION_MODEL)

# Load the model with error handling
try:
    model = helper.load_model(path_model)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {path_model}")
    st.error(ex)

#7.Image Processing Logic
if upload and upload.type.endswith(('jpg', 'png', 'jpeg')):
    img = Image.open(upload).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        try:
            if img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = img
                st.image(img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if upload and upload.type.endswith(('jpg', 'png', 'jpeg')):
                uploaded_image = img
                res = model.predict(uploaded_image, conf=numb_confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

else:
    st.error("Please select a valid source type!")
