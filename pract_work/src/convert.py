import os
import numpy as np
from PIL import Image

MASK_SRC = "../dataset/train_dataset_for_students/labels/train"
MASK_DST = "../dataset/train_dataset_for_students/labels/new_train"

os.makedirs(MASK_DST, exist_ok=True)

color_to_class = {
    (0, 0, 0): 0,        
    (51, 221, 255): 2,   
    (250, 50, 83): 1,     
}

for fname in os.listdir(MASK_SRC):
    if not fname.endswith('.png'):
        continue
    rgb_mask = np.array(Image.open(os.path.join(MASK_SRC, fname)))
    h, w = rgb_mask.shape[:2]
    id_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            color = tuple(rgb_mask[i, j])
            id_mask[i, j] = color_to_class.get(color, 0)  
    Image.fromarray(id_mask).save(os.path.join(MASK_DST, fname))

print("✅ Конвертация завершена.")