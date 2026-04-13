import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = Path(__file__).parent / "model" / "transfer_vgg16_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = ['cat', 'cow', 'deep', 'dog', 'lion']
IMG_SIZE = (150, 150)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = tf.keras.applications.vgg16.preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не передан'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Неподдерживаемый формат'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        img_tensor = preprocess_image(filepath)
        pred_probs = model.predict(img_tensor, verbose=0)[0]
        pred_class_idx = np.argmax(pred_probs)
        pred_class = CLASSES[pred_class_idx]
        confidence = float(pred_probs[pred_class_idx])
        probabilities = {cls: float(pred_probs[i]) for i, cls in enumerate(CLASSES)}
        return jsonify({
            'class': pred_class,
            'confidence': confidence,
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)