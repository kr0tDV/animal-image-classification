import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Параметры
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
CLASSES = ['cat', 'cow', 'deep', 'dog', 'lion']

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = BASE_DIR / "model"

# Загрузка лучшей модели
model_path = MODEL_DIR / "transfer_vgg16_best.keras"
model = tf.keras.models.load_model(model_path)
print(f"Модель загружена из {model_path}")

# Генератор тестовых данных (только нормализация, без аугментации)
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

# Предсказания
steps = test_generator.samples // BATCH_SIZE + 1
y_pred_probs = model.predict(test_generator, steps=steps)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes[:len(y_pred)]

# Метрики
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Transfer Learning (VGG16)')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'confusion_matrix.png')
plt.show()

# Точность на тесте
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")