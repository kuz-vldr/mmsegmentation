import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict


IMG_DIR = "../dataset/train_dataset_for_students/img/train"
ANN_DIR = "../dataset/train_dataset_for_students/labels/new_train"
OUT_DIR = "../viz/eda"

os.makedirs(OUT_DIR, exist_ok=True)

def get_basename(f):
    return os.path.splitext(f)[0]


class_pixel_count = defaultdict(int)
image_sizes = []
object_areas = []

print("Сбор статистики...")
img_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
ann_files = {get_basename(f): f for f in os.listdir(ANN_DIR) if f.endswith('.png')}

for img_name in img_files:
    base = get_basename(img_name)
    if base not in ann_files:
        continue


    img_path = os.path.join(IMG_DIR, img_name)
    with Image.open(img_path) as img:
        w, h = img.size
        image_sizes.append((w, h))


    ann_path = os.path.join(ANN_DIR, ann_files[base])
    mask = np.array(Image.open(ann_path))


    unique, counts = np.unique(mask, return_counts=True)
    for cls, cnt in zip(unique, counts):
        class_pixel_count[cls] += cnt


    for cls in np.unique(mask):
        if cls == 0:
            continue
        object_mask = (mask == cls)
        area = np.sum(object_mask)
        object_areas.append(area)


print("Генерация графика: распределение классов...")
classes = sorted(class_pixel_count.keys())
pixels = [class_pixel_count[cls] for cls in classes]
total = sum(pixels)
percentages = [p / total * 100 for p in pixels]

plt.figure(figsize=(8, 5))
bars = plt.bar([f"Класс {c}" for c in classes], percentages, color=['gray', 'salmon', 'skyblue'][:len(classes)])
plt.ylabel("Доля пикселей, %")
plt.title("Распределение классов в датасете")
plt.xticks(rotation=0)


for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{pct:.1f}%", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.close()


print("Генерация графика: площадь объектов...")
plt.figure(figsize=(8, 5))
plt.hist(object_areas, bins=50, color='orange', alpha=0.7)
plt.xlabel("Площадь объекта (пиксели)")
plt.ylabel("Количество объектов")
plt.title("Распределение площади объектов")
plt.axvline(np.mean(object_areas), color='red', linestyle='--', label=f'Среднее: {np.mean(object_areas):.0f}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "object_areas.png"), dpi=150)
plt.close()

print(f"✅ Все графики сохранены в: {OUT_DIR}")