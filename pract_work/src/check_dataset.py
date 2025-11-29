import os
from PIL import Image
import numpy as np
from collections import Counter

def get_basename_without_ext(filename):
    return os.path.splitext(filename)[0]

def analyze_dataset(img_dir, ann_dir, img_suffix='.jpg', ann_suffix='.png'):

    img_files = {get_basename_without_ext(f): f for f in os.listdir(img_dir) if f.endswith(img_suffix)}
    ann_files = {get_basename_without_ext(f): f for f in os.listdir(ann_dir) if f.endswith(ann_suffix)}

    img_basenames = set(img_files.keys())
    ann_basenames = set(ann_files.keys())

    common = img_basenames & ann_basenames
    only_img = img_basenames - ann_basenames
    only_ann = ann_basenames - img_basenames

    print("Проверка соответствия имён (по базовому имени)...")
    if only_img:
        print(f"Изображения без аннотаций ({len(only_img)} шт.):")
        for name in sorted(list(only_img))[:5]:
            print(f"    {name}{img_suffix}")
    if only_ann:
        print(f"Аннотации без изображений ({len(only_ann)} шт.):")
        for name in sorted(list(only_ann))[:5]:
            print(f"    {name}{ann_suffix}")

    if not only_img and not only_ann:
        print("Все файлы имеют пару.")
    else:
        print(f"ℹБудет обработано {len(common)} пар.")


    all_pixels = []
    size_mismatch = []

    for base in sorted(common):
        img_path = os.path.join(img_dir, img_files[base])
        ann_path = os.path.join(ann_dir, ann_files[base])

        img = Image.open(img_path)
        ann = Image.open(ann_path)

        if img.size != ann.size:
            size_mismatch.append(base)

        ann_arr = np.array(ann)
        all_pixels.extend(ann_arr.flatten())

    if size_mismatch:
        print(f"\nНесовпадение размеров у {len(size_mismatch)} пар (примеры):")
        for name in size_mismatch[:5]:
            print(f"    {name}")
    else:
        print("\nРазмеры изображений и масок совпадают.")

    pixel_counter = Counter(all_pixels)
    classes = sorted(pixel_counter.keys())
    total = sum(pixel_counter.values())

    print(f"\nУникальные значения масок (классы): {classes}")
    print("Распределение пикселей по классам:")
    for cls in classes:
        count = pixel_counter[cls]
        pct = count / total * 100 if total > 0 else 0
        print(f"  класс {cls}: {count:>10} пикселей ({pct:>6.3f}%)")

    return list(common)


IMG_DIR = "../dataset/train_dataset_for_students/img/train"
ANN_DIR = "../dataset/train_dataset_for_students/labels/new_train"

if __name__ == "__main__":
    print("🔎 Начинаю анализ датасета...\n")
    common_files = analyze_dataset(IMG_DIR, ANN_DIR, img_suffix='.jpg', ann_suffix='.png')