# pylint: disable=no-member
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Пути
data_dir = Path("./data")
train_dir = data_dir / "train"

# Классы
classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
print("Классы:", classes)

# Подсчёт изображений
print("\nКоличество изображений в train:")
for cls in classes:
    cls_path = train_dir / cls
    imgs = list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
    print(f"{cls}: {len(imgs)}")

# Показать примеры
fig, axes = plt.subplots(1, len(classes), figsize=(15, 4))
for i, cls in enumerate(classes):
    img_files = list((train_dir / cls).glob("*.jpeg"))
    if not img_files:
        print(f"В классе {cls} нет .jpeg файлов, пропускаем")
        continue
    img_path = img_files[0]
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].set_title(cls)
    axes[i].axis('off')
plt.tight_layout()
plt.show()