import os
import random
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from preprocess_target import canonicalizza_bordi


class SkeletonKeypointDataset(Dataset):
    """
    augment=True attiva un set di trasformazioni pensate per ridurre il
    divario di dominio tra le illustrazioni stilizzate del training set e
    foto reali di fossili (che vedra' solo in inferenza, mai in training):
    - flip orizzontale (i 14 keypoint non distinguono destra/sinistra, sono
      tutti sul profilo visibile, quindi il flip e' geometricamente sicuro
      e basta ribaltare la coordinata x)
    - color jitter e desaturazione occasionale (le foto reali hanno
      illuminazione/colore molto piu' vari delle illustrazioni)
    - blend occasionale con i bordi (Canny), la stessa trasformazione che
      preprocess_target.py puo' applicare a un fossile vero in inferenza:
      addestrando anche su questa variante la rete generalizza meglio a
      quello stile.
    Da usare solo in train.py; per build_DB.py/inferenza augment=False.
    """
    def __init__(self, img_dir, json_dir, img_size=512, sigma=5.0, augment=False):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.img_size = img_size
        self.sigma = sigma
        self.augment = augment

        self.img_names = [f for f in sorted(os.listdir(img_dir)) if f.endswith(('.png', '.jpg'))]

        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)

        self.to_tensor_normalize = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.nomi_punti = [
            "punta_muso", "retro_cranio", "base_collo", "spalla",
            "gomito_ant_1", "gomito_ant_2", "zampa_ant", "dorso", "anca",
            "ginocchio_post_1", "ginocchio_post_2", "zampa_post", "base_coda", "punta_coda"
        ]

    def _generate_heatmap(self, x, y):
        grid_y, grid_x = np.mgrid[0:self.img_size, 0:self.img_size]
        heatmap = np.exp(-((grid_x - x)**2 + (grid_y - y)**2) / (2 * self.sigma**2))
        return heatmap

    def __len__(self):
        return len(self.img_names)

    def _applica_augmentation(self, image, coords, orig_w):
        """image: PIL.Image. coords: dict nome->[x,y] o None, in pixel
        dell'immagine ORIGINALE (prima del resize a img_size). Ritorna
        (image_augmentata, coords_augmentate)."""
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            coords = {
                nome: (None if p is None else [orig_w - p[0], p[1]])
                for nome, p in coords.items()
            }

        image = self.color_jitter(image)

        if random.random() < 0.25:
            image = transforms.functional.to_grayscale(image, num_output_channels=3)

        if random.random() < 0.25:
            arr_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            arr_bgr = canonicalizza_bordi(arr_bgr, alpha=random.uniform(0.3, 0.7))
            image = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))

        return image, coords

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        json_path = os.path.join(self.json_dir, f"{base_name}.json")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        with open(json_path, 'r', encoding='utf-8') as f:
            annotazioni = json.load(f)

        if self.augment:
            image, annotazioni = self._applica_augmentation(image, annotazioni, orig_w)

        image_tensor = self.to_tensor_normalize(image)

        heatmaps = np.zeros((14, self.img_size, self.img_size), dtype=np.float32)
        for i, nome in enumerate(self.nomi_punti):
            coord = annotazioni.get(nome)
            if coord is not None:
                x = (coord[0] / orig_w) * self.img_size
                y = (coord[1] / orig_h) * self.img_size
                heatmaps[i] = self._generate_heatmap(x, y)

        heatmap_tensor = torch.from_numpy(heatmaps)

        return image_tensor, heatmap_tensor


if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(_base_dir, "data", "processed", "input_x")
    json_dir = os.path.join(_base_dir, "data", "processed", "Labels_X")

    dataset = SkeletonKeypointDataset(img_dir=img_dir, json_dir=json_dir, img_size=512, sigma=5.0, augment=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    print(f"Dataset caricato: {len(dataset)} campioni.")

    immagini, heatmaps = next(iter(dataloader))

    print(f"Shape tensore Input (Immagini): {immagini.shape}")
    print(f"Shape tensore Target (Heatmaps): {heatmaps.shape}")

    img_vis = immagini[0].numpy().transpose(1, 2, 0)
    img_vis = (img_vis * 0.5) + 0.5

    heatmap_vis = heatmaps[0].numpy().sum(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_vis)
    ax[0].set_title("Tensore Input (512x512, con augmentation)")
    ax[0].axis("off")

    ax[1].imshow(img_vis)
    ax[1].imshow(heatmap_vis, cmap="jet", alpha=0.5)
    ax[1].set_title("Tensore Target (14 Heatmap aggregate)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
