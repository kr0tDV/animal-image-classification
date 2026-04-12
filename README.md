```markdown
# Animal Image Classification (5 species)

**Классификация изображений домашних и диких животных: кошка, корова, олень (deep), собака, лев.**  
Реализованы базовая свёрточная нейронная сеть и трансферное обучение (VGG16).  
Добавлен скрипт для предсказания класса одного изображения.

---

## 📊 О датасете

**Источник:** [Kaggle – Animal Image Classification (5 Species)](https://www.kaggle.com/datasets/miadul/animal-image-classification-5-species)  
**Содержание:** 629 изображений пяти классов животных. Данные разбиты на три папки: `train` (обучение), `validation` (валидация), `test` (тест). В каждой папке – подпапки по классам.

**Классы:**
- `cat` – кошка
- `cow` – корова
- `deep` – олень (в датасете название `deep`, соответствует `deer`)
- `dog` – собака
- `lion` – лев

**Количество изображений:**
- Train: 464
- Validation: 83
- Test: 82

**Формат:** JPEG

---

## 🧹 Обработка данных

- Все изображения приведены к размеру 150×150 пикселей.
- Применена нормализация пикселей (деление на 255).
- Для тренировочных данных использована аугментация: повороты, сдвиги, отражения, изменение яркости.
- Для валидации и теста – только нормализация.

---

## 🔍 Анализ и моделирование

### 1. Базовая свёрточная нейронная сеть (CNN)

Построена сеть с нуля, состоящая из четырёх свёрточных блоков (Conv2D + MaxPooling2D), полносвязного слоя 512 нейронов с Dropout 0.5 и выходного softmax слоя на 5 классов.

**Результат:** точность на тесте ~20.7% (уровень случайного угадывания). Модель не обучилась из-за малого объёма данных.

### 2. Трансферное обучение (VGG16)

Использована предобученная на ImageNet модель VGG16 без верхушки. Базовые слои заморожены, добавлены:
- GlobalAveragePooling2D
- Полносвязный слой 256 нейронов (ReLU)
- Dropout 0.5
- Выходной слой 5 нейронов (softmax)

Обучение проводилось с оптимизатором Adam (learning rate = 0.0001) и ранней остановкой.

**Результат:** точность на тесте **89.1%**, loss = 0.466.

### 3. Оценка качества

Построены confusion matrix и classification report (precision, recall, f1-score).

**Confusion matrix** (сохранена в `model/confusion_matrix.png`):

|            | cat | cow | deep | dog | lion |
|------------|-----|-----|------|-----|------|
| **cat**    | 17  | 0   | 0    | 3   | 0    |
| **cow**    | 0   | 10  | 1    | 0   | 0    |
| **deep**   | 0   | 1   | 16   | 1   | 0    |
| **dog**    | 2   | 0   | 0    | 12  | 0    |
| **lion**   | 0   | 0   | 0    | 0   | 19   |

**Classification report (macro avg):**
- Precision: 0.90
- Recall: 0.89
- F1-score: 0.89

### 4. Скрипт предсказания для одного изображения

Разработан скрипт `predict.py`, который:
- Загружает обученную модель (`transfer_vgg16_best.keras`)
- Принимает путь к изображению
- Выводит предсказанный класс, уверенность и вероятности по всем классам.

Пример вывода:
```
Предсказанный класс: cat
Уверенность: 0.9797
Вероятности по классам:
  cat: 0.9797
  cow: 0.0000
  deep: 0.0000
  dog: 0.0203
  lion: 0.0000
```

---

## 🛠️ Стек технологий

| Категория | Технологии |
|-----------|-------------|
| Язык | Python 3.8+ |
| Фреймворк для DL | TensorFlow 2.x + Keras |
| Работа с данными | NumPy, Pandas, OpenCV |
| Визуализация | Matplotlib, Seaborn |
| Метрики | scikit-learn |
| Среда | Jupyter Notebook / VS Code |
| Управление зависимостями | pip + requirements.txt |

---

## 📁 Структура проекта

```
Animal Image Classification/
├── data/
│   ├── train/                # тренировочные данные по классам
│   ├── validation/           # валидационные данные
│   └── test/                 # тестовые данные
├── model/                    # сохранённые модели и графики
│   ├── transfer_vgg16_best.keras
│   ├── transfer_vgg16_final.keras
│   ├── transfer_vgg16_history.png
│   ├── confusion_matrix.png
│   ├── baseline_cnn_best.keras
│   └── baseline_cnn_history.png
├── scr/                      # скрипты
│   ├── check.py              # первичный анализ данных
│   ├── train_cnn.py          # обучение базовой CNN
│   ├── transfer_learning.py  # обучение VGG16
│   ├── evaluate.py           # метрики и матрица ошибок
│   └── predict.py            # консольное приложение для предсказаний
├── notebooks/                # Jupyter ноутбук с полным анализом
├── report.ipynb              # итоговый отчёт
├── README.md
└── requirements.txt
```

---

## ▶️ Как запустить

1. **Клонировать репозиторий:**
   ```bash
   git clone https://github.com/kr0tDV/animal-image-classification.git
   cd animal-image-classification
   ```

2. **Установить зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Обучить модели (опционально – модели уже в `model/`):**
   ```bash
   python scr/train_cnn.py           # базовая CNN
   python scr/transfer_learning.py   # VGG16
   ```

4. **Оценить лучшую модель:**
   ```bash
   python scr/evaluate.py
   ```

5. **Запустить предсказание для одного изображения:**
   ```bash
   python scr/predict.py "путь/к/изображению.jpeg"
   ```

6. **Просмотреть полный анализ:** откройте `report.ipynb` в Jupyter Notebook.

---

## 📌 Основные выводы

- Базовая CNN (с нуля) показала низкую точность (~20.7%), что ожидаемо для небольшого датасета без предобучения.
- Трансферное обучение с VGG16 достигло **89.1% точности**, подтверждая эффективность использования предобученных признаков.
- Наиболее сложные для различения пары – `cat` и `dog` (визуальная схожесть).
- Разработанный скрипт предсказания позволяет классифицировать новые изображения пяти видов животных.

**Лучшая модель сохранена:** `model/transfer_vgg16_best.keras`

---

## 📚 Источники

- [Kaggle Dataset: Animal Image Classification (5 species)](https://www.kaggle.com/datasets/miadul/animal-image-classification-5-species)
- TensorFlow / Keras documentation
- VGG16: Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition"
