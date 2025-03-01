from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
import os
import uuid  # Import the uuid module
from plate_detection import load_models, preprocess_image, detect_license_plate, crop_license_plate, highlight_license_plate, encode_image
from crnn_recognition import load_char_to_id_mapping, preprocess_plate, predict_and_decode_plate
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

# Load models
yolo_model, crnn_model = load_models('models/yolo11s_28feb.pt', 'models/crnn_model_pred_20250225_102044.keras')
char_to_id_mapping = load_char_to_id_mapping('models/char_to_id_mapping.csv')

def process_image(image_path, filename=""):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    image = preprocess_image(image)

    # Detect license plate
    max_box = detect_license_plate(image, yolo_model)

    # Crop the license plate
    cropped_plate = crop_license_plate(image, max_box)

    # Decode the plate
    if cropped_plate is not None:
        preprocessed_plate = preprocess_plate(cropped_plate)
        if preprocessed_plate is not None:
            prediction_result = predict_and_decode_plate(preprocessed_plate, crnn_model, char_to_id_mapping)
            print(f"Prediction result: {prediction_result}")
            plate_number = prediction_result['predicted_seq']
        else:
            plate_number = None
    else:
        plate_number = None

    # Highlight the license plate
    highlighted_image = highlight_license_plate(image, max_box)
    
    # Encode the highlighted image
    highlighted_image_base64 = encode_image(highlighted_image)

    if max_box:
        print(f"processed: {filename}")
    else:
        print(f"No box detected: {filename}")

    return plate_number, highlighted_image_base64

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Generate a unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = str(uuid.uuid4()) + file_extension
    filename = secure_filename(unique_filename)  # Sanitize the filename

    # Save the file to the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Process the image using your models
    plate_number, highlighted_image_base64 = process_image(file_path, filename)
    
    # Remove the temporary file
    os.remove(file_path)
    print("plate_number: ", plate_number)
    return jsonify({'plate_number': plate_number, 'highlighted_image': highlighted_image_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8787, debug=True)