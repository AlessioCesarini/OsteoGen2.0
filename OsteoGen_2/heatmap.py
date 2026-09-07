import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SkeletonKeypointDataset(Dataset):
    def __init__(self, img_dir, json_dir, img_size=512, sigma=5.0):
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.img_size = img_size
        self.sigma = sigma
        
        # Carica solo immagini valide
        self.img_names = [f for f in sorted(os.listdir(img_dir)) if f.endswith(('.png', '.jpg'))]
        
        # Trasformazioni base per i tensori di input
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # Normalizzazione standard per reti pre-addestrate (es. ResNet)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.nomi_punti = [
            "punta_muso", "retro_cranio", "base_collo", "spalla",
            "gomito_ant_1", "gomito_ant_2", "zampa_ant", "dorso", "anca",
            "ginocchio_post_1", "ginocchio_post_2", "zampa_post", "base_coda", "punta_coda"
        ]

    def _generate_heatmap(self, x, y):
        # Genera una griglia di coordinate 512x512
        grid_y, grid_x = np.mgrid[0:self.img_size, 0:self.img_size]
        
        # Funzione gaussiana per creare il picco sul keypoint
        heatmap = np.exp(-((grid_x - x)**2 + (grid_y - y)**2) / (2 * self.sigma**2))
        return heatmap

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        json_path = os.path.join(self.json_dir, f"{base_name}.json")
        
        # Lettura e trasformazione immagine
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image_tensor = self.transform(image)
        
        # Lettura annotazioni
        with open(json_path, 'r') as f:
            annotazioni = json.load(f)
            
        # Inizializza il tensore target (14 canali, 512, 512)
        heatmaps = np.zeros((14, self.img_size, self.img_size), dtype=np.float32)
        
        for i, nome in enumerate(self.nomi_punti):
            coord = annotazioni.get(nome)
            if coord is not None:
                # Proporziona le coordinate se l'immagine originale non era esattamente 512x512
                x = (coord[0] / orig_w) * self.img_size
                y = (coord[1] / orig_h) * self.img_size
                heatmaps[i] = self._generate_heatmap(x, y)
                
        heatmap_tensor = torch.from_numpy(heatmaps)
        
        return image_tensor, heatmap_tensor


if __name__ == "__main__":
    # Setup dei percorsi
    img_dir = os.path.join("Data", "Processed", "Input_X")
    json_dir = os.path.join("Data", "Processed", "Labels_X")

    # Inizializza il dataset
    dataset = SkeletonKeypointDataset(img_dir=img_dir, json_dir=json_dir, img_size=512, sigma=5.0)
    
    # DataLoader configurato per saturare i CUDA core
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    print(f"Dataset caricato: {len(dataset)} campioni.")

    # Estrae un batch per il test
    immagini, heatmaps = next(iter(dataloader))
    
    print(f"Shape tensore Input (Immagini): {immagini.shape}")
    print(f"Shape tensore Target (Heatmaps): {heatmaps.shape}")

    # --- Visualizzazione di Test del primo elemento del batch ---
    # Denormalizziamo l'immagine per poterla visualizzare correttamente con matplotlib
    img_vis = immagini[0].numpy().transpose(1, 2, 0)
    img_vis = (img_vis * 0.5) + 0.5 
    
    # Collassiamo i 14 canali delle heatmap in un'unica mappa 2D sommando lungo l'asse dei canali (dim 0)
    heatmap_vis = heatmaps[0].numpy().sum(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Mostra l'immagine originale
    ax[0].imshow(img_vis)
    ax[0].set_title("Tensore Input (512x512)")
    ax[0].axis("off")
    
    # Mostra l'immagine con le gaussiane sovrapposte (canale alpha per la trasparenza)
    ax[1].imshow(img_vis)
    ax[1].imshow(heatmap_vis, cmap="jet", alpha=0.5)
    ax[1].set_title("Tensore Target (14 Heatmap aggregate)")
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.show()