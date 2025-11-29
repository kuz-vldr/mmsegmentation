import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

IMG_DIR = "../dataset/train_dataset_for_students/img/train"
ANN_DIR = "../dataset/train_dataset_for_students/labels/train"
OUT_ROOT = "../dataset/eda"
IMG_SUFFIX = ".jpg"
ANN_SUFFIX = ".png"
NUM_SAMPLES_PER_CLASS = 10

def get_basename_without_ext(f):
    return os.path.splitext(f)[0]

def get_common_files():
    imgs = {get_basename_without_ext(f) for f in os.listdir(IMG_DIR) if f.endswith(IMG_SUFFIX)}
    anns = {get_basename_without_ext(f) for f in os.listdir(ANN_DIR) if f.endswith(ANN_SUFFIX)}
    return sorted(imgs & anns)

def apply_palette(mask, palette):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(palette):
        color_mask[mask == cls_id] = color
    return color_mask

def visualize_sample(img_path, ann_path, out_path, palette):
    img = np.array(Image.open(img_path))
    ann = np.array(Image.open(ann_path))
    color_ann = apply_palette(ann, palette)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Изображение")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(ann, cmap="tab20")
    plt.title("Маска (ID)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(color_ann, alpha=0.6)
    plt.title("Наложение")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    common_files = get_common_files()
    print(f"Найдено {len(common_files)} корректных пар.")

    class_to_files = defaultdict(list)
    class_ids_found = set()

    print("Сканирование масок для определения классов...")
    for base in common_files:
        ann_path = os.path.join(ANN_DIR, base + ANN_SUFFIX)
        ann = np.array(Image.open(ann_path))
        unique_classes = np.unique(ann)
        class_ids_found.update(unique_classes)
        for cls in unique_classes:
            class_to_files[cls].append(base)

    class_ids = sorted([c for c in class_ids_found if c != 255])
    print(f"Найдены классы: {class_ids}")

    from matplotlib.cm import get_cmap
    cmap = get_cmap('tab20', len(class_ids))
    PALETTE = [(np.array(cmap(i)[:3]) * 255).astype(int).tolist() for i in range(len(class_ids))]
    FULL_PALETTE = [[0, 0, 0]] * (max(class_ids) + 1)
    for i, cls in enumerate(class_ids):
        FULL_PALETTE[cls] = PALETTE[i]

    for cls in class_ids:
        files = class_to_files[cls]
        samples = files[:NUM_SAMPLES_PER_CLASS]
        out_dir = os.path.join(OUT_ROOT, f"class_{cls}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Класс {cls}: найдено {len(files)} изображений → сохраняю {len(samples)} примеров в {out_dir}")

        for i, base in enumerate(samples):
            img_path = os.path.join(IMG_DIR, base + IMG_SUFFIX)
            ann_path = os.path.join(ANN_DIR, base + ANN_SUFFIX)
            out_path = os.path.join(out_dir, f"sample_{i+1:02d}_{base}.png")
            visualize_sample(img_path, ann_path, out_path, FULL_PALETTE)

    print(f"\nВизуализация по классам завершена. Результаты в: {OUT_ROOT}")