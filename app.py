import sys
print(sys.executable)

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model('mnist_model.keras')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img  # invert colors
    img = img / 255.0
    img = img.reshape(1, 28, 28)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
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
