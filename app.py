import streamlit as st
import numpy as np
import cv2
import os
import uuid
from plate_detection import load_models, preprocess_image, detect_license_plate, crop_license_plate, highlight_license_plate, encode_image
from crnn_recognition import load_char_to_id_mapping, preprocess_plate, predict_and_decode_plate
from PIL import Image, ImageFont, ImageDraw
import io

# Create directories if they don't exist
UPLOAD_FOLDER = "uploads"
CROPPED_FOLDER = "cropped"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

# Load models (do this only once)
@st.cache_resource
def load_models_cached():
    yolo_model, crnn_model = load_models('models/yolo11s_28feb.pt', 'models/crnn_model_pred_20250225_102044.keras')
    char_to_id_mapping = load_char_to_id_mapping('models/char_to_id_mapping.csv')
    return yolo_model, crnn_model, char_to_id_mapping

yolo_model, crnn_model, char_to_id_mapping = load_models_cached()

def process_image(image_path):
    # Load the image from the file path
    image = cv2.imread(image_path)

    if image is None:
        st.error("Error loading image. Please ensure it's a valid image file.")
        return None, None, None

    # Preprocess the image
    image = preprocess_image(image)

    # Detect license plate
    max_box = detect_license_plate(image, yolo_model)

    # Crop the license plate
    cropped_plate = crop_license_plate(image, max_box)

    plate_number = None  # Initialize plate_number
    cropped_image_path = None
    if cropped_plate is not None:
        # Save the cropped plate to the "cropped" folder
        cropped_filename = os.path.join(CROPPED_FOLDER, f"cropped_{uuid.uuid4()}.jpg")
        cv2.imwrite(cropped_filename, cropped_plate)
        cropped_image_path = cropped_filename

        preprocessed_plate = preprocess_plate(cropped_plate)
        if preprocessed_plate is not None:
            prediction_result = predict_and_decode_plate(preprocessed_plate, crnn_model, char_to_id_mapping)
            plate_number = prediction_result['predicted_seq']
        else:
            plate_number = None
    else:
        plate_number = None

    # Highlight the license plate
    highlighted_image = highlight_license_plate(image, max_box)
    
    # Encode the highlighted image
    highlighted_image_base64 = encode_image(highlighted_image)

    return plate_number, highlighted_image_base64, highlighted_image, cropped_image_path

def display_predicted_plate(plate_number):
    """Displays the predicted plate number with each character in a separate box."""
    if not plate_number:
        st.write("No license plate detected.")
        return

    num_chars = len(plate_number)

    # Load Vazirmatn font
    st.markdown(
        """
        <style>
        @font-face {
            font-family: 'Vazirmatn';
            src: url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@29.1.0/dist/Vazirmatn.woff2') format('woff2');
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use columns to create the boxes
    cols = st.columns(num_chars)  # Create n columns

    for i, char in enumerate(plate_number):
        with cols[i]:
            st.markdown(
                f"<div style='border: 2px solid black; "
                f"text-align: center; "
                f"background-color: #007bff; "  # Change background color color: #007bff
                f"font-family: Vazirmatn; "  # Change font to Vazirmatn
                f"font-size: 2em; "
                f"padding: 5px; "
                f"width: 100%; "  # Ensure box fills the column
                f"box-sizing: border-box;'>"  # Include border in width calculation
                f"{char}</div>",
                unsafe_allow_html=True,
            )

# Streamlit UI
st.title("License Plate Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to the "uploads" folder
    image = Image.open(uploaded_file)
    image_filename = os.path.join(UPLOAD_FOLDER, f"uploaded_{uuid.uuid4()}.jpg")
    image.save(image_filename)

    # Process the image
    plate_number, highlighted_image_base64, highlighted_image, cropped_image_path = process_image(image_filename)

    # Display the predicted plate
    # st.subheader("Predicted License Plate")
    display_predicted_plate(plate_number)

    # Display the highlighted image
    # st.subheader("Highlighted License Plate")
    st.image(highlighted_image, use_container_width=True, channels="BGR")
