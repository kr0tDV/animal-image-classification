import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import sys

CLASSES = ['cat', 'cow', 'deep', 'dog', 'lion']
IMG_SIZE = (150, 150)

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    # Предобработка для VGG16
    img = tf.keras.applications.vgg16.preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img = load_and_preprocess_image(image_path)
    pred_probs = model.predict(img, verbose=0)[0]
    pred_class = np.argmax(pred_probs)
    confidence = pred_probs[pred_class]
    return CLASSES[pred_class], confidence, pred_probs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python predict.py <путь_к_изображению>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    # Путь к модели: корень проекта/model/transfer_vgg16_best.keras
    model_path = Path(__file__).parent.parent / "model" / "transfer_vgg16_best.keras"
    
    if not model_path.exists():
        print(f"Модель не найдена: {model_path}")
        print("Убедитесь, что файл модели существует в папке model/")
        sys.exit(1)
    
    class_name, confidence, all_probs = predict_image(model_path, image_path)
    print(f"Предсказанный класс: {class_name}")
    print(f"Уверенность: {confidence:.4f}")
    print("Вероятности по классам:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {all_probs[i]:.4f}")