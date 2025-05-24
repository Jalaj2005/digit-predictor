import sys
print(sys.executable)

import os
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'mnist_model.keras'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model if available
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print(f"⚠️ Model file not found at {MODEL_PATH}. Please make sure it's present.")

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img  # Invert colors (white digit on black bg)
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files or not model:
            return "Model not loaded or no file uploaded", 500

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image = preprocess_image(filepath)
        pred = model.predict(image)
        prediction = np.argmax(pred)
        image_url = file.filename

    return render_template('index.html', prediction=prediction, image_url=image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

