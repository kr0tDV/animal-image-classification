import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from pathlib import Path

# ========== ПАРАМЕТРЫ ==========
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20
CLASSES = ['cat', 'cow', 'deep', 'dog', 'lion']

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ========== ГЕНЕРАТОРЫ С ПРЕДОБРАБОТКОЙ ДЛЯ VGG16 ==========
# VGG16 требует нормализации через tf.keras.applications.vgg16.preprocess_input
def preprocess_image(x):
    # x приходит в диапазоне [0,255] от ImageDataGenerator(rescale=1/255)?
    # Лучше не использовать rescale, а передавать в preprocess_input значения [0,255]
    return tf.keras.applications.vgg16.preprocess_input(x * 255.0)

# Генераторы без масштабирования (оставляем пиксели 0-255)
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator()

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

# Применяем предобработку VGG16 к изображениям на лету
def apply_preprocess(generator):
    for x, y in generator:
        yield tf.keras.applications.vgg16.preprocess_input(x), y

train_dataset = tf.data.Dataset.from_generator(
    lambda: apply_preprocess(train_generator),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None, 5])
).repeat().prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: apply_preprocess(val_generator),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None, 5])
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: apply_preprocess(test_generator),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None, 5])
).prefetch(tf.data.AUTOTUNE)

# ========== ПОСТРОЕНИЕ МОДЕЛИ С VGG16 ==========
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model.trainable = False  # замораживаем базовые слои

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

model.summary()

# ========== КОЛБЭКИ ==========
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    MODEL_DIR / 'transfer_vgg16_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# ========== ОБУЧЕНИЕ ==========
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = val_generator.samples // BATCH_SIZE

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ========== ГРАФИКИ ==========
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'transfer_vgg16_history.png')
plt.show()

# ========== ОЦЕНКА НА ТЕСТЕ ==========
test_steps = test_generator.samples // BATCH_SIZE
test_loss, test_acc = model.evaluate(test_dataset, steps=test_steps, verbose=1)
print(f"\nТестовая точность (VGG16): {test_acc:.4f}")
print(f"Тестовые потери: {test_loss:.4f}")

model.save(MODEL_DIR / 'transfer_vgg16_final.keras')
print(f"Модель сохранена в {MODEL_DIR}")