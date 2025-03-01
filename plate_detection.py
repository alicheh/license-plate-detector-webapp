import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
import base64

def load_models(yolo_path, crnn_path):
    """Loads the YOLO and CRNN models."""
    yolo_model = YOLO(yolo_path)
    crnn_model = keras.models.load_model(crnn_path)
    return yolo_model, crnn_model

def preprocess_image(image):
    """Preprocesses the image for YOLO detection."""
    # Resize, normalize, etc.
    # Add your preprocessing steps here
    return image

def detect_license_plate(image, yolo_model, conf_threshold=0.8, iou_threshold=0.5):
    """Detects the license plate in the image using YOLO."""
    results = yolo_model(image, conf=conf_threshold, iou=iou_threshold)
    max_box = None
    max_area = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_box = (x1, y1, x2, y2)

    return max_box

def crop_license_plate(image, box):
    """Crops the license plate from the image given the bounding box."""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    cropped_plate = image[y1:y2, x1:x2]
    return cropped_plate

def decode_plate(image, crnn_model):
    """Decodes the license plate from the cropped image using CRNN."""
    # Add your CRNN decoding logic here
    return "ABC123"  # Placeholder

def highlight_license_plate(image, box):
    """Highlights the license plate in the original image."""
    if box is None:
        return image
    x1, y1, x2, y2 = box
    highlighted_image = image.copy()
    cv2.rectangle(highlighted_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green rectangle
    return highlighted_image

def encode_image(image):
    """Encodes the image to base64."""
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return img_base64