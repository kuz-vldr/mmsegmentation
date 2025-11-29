import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


IMG_PATH = "../dataset/train_dataset_for_students/img/train/000000028253_7169.jpg" 
ANN_PATH = "../dataset/train_dataset_for_students/labels/new_train/000000028253_7169.png"
OUT_PATH = "../dataset/eda/class_2/new"


PALETTE = [
    [0, 0, 0],      
    [255, 0, 0],     
    [0, 0, 255],     
]

def apply_color_map(mask, palette):
    """Преобразует маску (H, W) с ID в цветное изображение (H, W, 3)."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask


img = np.array(Image.open(IMG_PATH))
ann = np.array(Image.open(ANN_PATH))


if img.shape[:2] != ann.shape:
    print(f"Размеры не совпадают: {img.shape} vs {ann.shape}")

    ann = np.array(Image.fromarray(ann).resize((img.shape[1], img.shape[0]), Image.NEAREST))

color_ann = apply_color_map(ann, PALETTE)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Изображение")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(color_ann)
plt.title("Маска (цвет)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img)
plt.imshow(color_ann, alpha=0.6)
plt.title("Наложение")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close()

print(f"Визуализация сохранена: {OUT_PATH}")