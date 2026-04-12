import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 1. ПАРАМЕТРЫ ==========
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
CLASSES = ['cat', 'cow', 'deep', 'dog', 'lion']  # именно так, как в папках

# Пути (относительно корня проекта)
BASE_DIR = Path(__file__).parent.parent   # поднимаемся из src/ в корень
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

# Папка для сохранения модели
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ========== 2. ГЕНЕРАТОРЫ ДАННЫХ ==========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    shuffle=False
)

print("\nСоответствие классов и индексов:", train_generator.class_indices)

# Проверка: сколько изображений в каждом генераторе
print(f"Train samples: {train_generator.samples}")
print(f"Val samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# Посмотрим один батч
x_batch, y_batch = next(train_generator)
print(f"Batch shape: {x_batch.shape}, min={x_batch.min():.2f}, max={x_batch.max():.2f}")

# ========== 3. ПОСТРОЕНИЕ БАЗОВОЙ CNN ==========
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ========== 4. КОЛБЭКИ ==========
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_DIR / 'baseline_cnn_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# ========== 5. ОБУЧЕНИЕ ==========
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ========== 6. ГРАФИКИ ==========
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'baseline_cnn_history.png')
plt.show()

# ========== 7. ОЦЕНКА НА ТЕСТЕ ==========
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"\nТестовая точность базовой CNN: {test_acc:.4f}")
print(f"Тестовые потери: {test_loss:.4f}")

# Сохраним финальную модель (не только лучшую)
model.save(MODEL_DIR / 'baseline_cnn_final.keras')
print(f"Модель сохранена в {MODEL_DIR}")