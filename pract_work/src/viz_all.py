import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


BASE_IMG_DIR = "../dataset/train_dataset_for_students/img"
BASE_ANN_DIR = "../dataset/train_dataset_for_students/labels"
OUT_ROOT = "../dataset/full_dataset"

SPLITS = ["train", "val", "test"]
IMG_SUFFIX = ".jpg"
ANN_SUFFIX = ".png"


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

def visualize_sample(img_path, ann_path, out_path, palette):
    img = np.array(Image.open(img_path))
    ann = np.array(Image.open(ann_path))
    

    if img.shape[:2] != ann.shape:
        print(f"⚠️  Размеры не совпадают: {img_path}")

        ann = np.array(Image.fromarray(ann).resize((img.shape[1], img.shape[0]), Image.NEAREST))
    
    color_ann = apply_color_map(ann, palette)
    
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
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


for split in SPLITS:
    img_dir = os.path.join(BASE_IMG_DIR, split)
    ann_dir = os.path.join(BASE_ANN_DIR, split)
    out_dir = os.path.join(OUT_ROOT, split)
    
    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
        print(f"Пропуск сплита {split}: не найдены директории")
        continue
        
    os.makedirs(out_dir, exist_ok=True)
    processed = 0
    
    for img_name in os.listdir(img_dir):
        if not img_name.endswith(IMG_SUFFIX):
            continue
        base = os.path.splitext(img_name)[0]
        ann_name = base + ANN_SUFFIX
        
        img_path = os.path.join(img_dir, img_name)
        ann_path = os.path.join(ann_dir, ann_name)
        
        if not os.path.exists(ann_path):
            continue
            
        out_path = os.path.join(out_dir, f"{base}.jpg")
        try:
            visualize_sample(img_path, ann_path, out_path, PALETTE)
            processed += 1
        except Exception as e:
            print(f"Ошибка при обработке {img_name}: {e}")
    
    print(f"✅ Сплит {split}: обработано {processed} изображений → {out_dir}")