import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def load_char_to_id_mapping(csv_path):
    """Loads the character to ID mapping from a CSV file."""
    df = pd.read_csv(csv_path)
    char_to_id = dict(zip(df['Character'], df['ID']))
    return char_to_id

def preprocess_plate(image, img_height=32, img_width=128):
    """Preprocesses the cropped license plate image."""
    if image is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image
    resized = cv2.resize(gray, (img_width, img_height))

    # Normalize pixel values to be between 0 and 1
    normalized = resized / 255.0

    # Add channel dimension (CRNN expects a channel dimension)
    processed_image = np.expand_dims(normalized, axis=-1)

    return processed_image

def predict_and_decode_plate(image, crnn_model, char_to_id_mapping):
    """Predicts the license plate sequence using the CRNN model and decodes it."""
    # Add batch dimension
    batch_image = np.expand_dims(image, axis=0)

    # Predict the output using the CRNN model
    prediction = crnn_model.predict(batch_image)

    # Decode the prediction using CTC greedy decoding
    predicted_ids = ctc_decode(prediction, char_to_id_mapping)

    # Convert the predicted IDs to characters
    predicted_seq = convert_ids_to_text(predicted_ids, char_to_id_mapping)

    return {'predicted_seq': predicted_seq}

def ctc_decode(prediction, char_to_id_mapping):
    """Decodes the prediction using CTC greedy (best path) decoding."""
    # Get the index of the character with the highest probability at each time step
    id_to_char = {v: k for k, v in char_to_id_mapping.items()}
    characters = list(id_to_char.values())
    
    input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
    
    # Use TensorFlow's ctc_greedy_decoder
    results = tf.keras.backend.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0].numpy()
    
    # Return the first sequence
    return results[0]

def convert_ids_to_text(decoded_seq, char_to_id_mapping):
    """Converts a sequence of IDs to a string using the provided mapping."""
    id_to_char = {v: k for k, v in char_to_id_mapping.items()}
    seq = decoded_seq.numpy() if hasattr(decoded_seq, 'numpy') else decoded_seq
    text = ""
    for char_id in seq:
        # Skip blank tokens (you can adjust if your blank index is different)
        # if char_id in id_to_char:
        if char_id != 0 and char_id in id_to_char:
            text += id_to_char[char_id]
    print(text)
    return text

