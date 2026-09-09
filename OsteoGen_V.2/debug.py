import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from resnet import ResNet18KeypointDetector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def salva_heatmap_inferenza(image_path, model_path, output_path, img_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inizializza il modello pre-addestrato
    model = ResNet18KeypointDetector(num_keypoints=14).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Pre-processing identico a quello di training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Estrazione delle heatmap grezze (senza threshold o conversioni in coordinate)
    with torch.no_grad():
        output = model(img_tensor)
        heatmaps = output.squeeze(0).cpu().numpy()

    # Denormalizza il tensore per la visualizzazione corretta dei colori
    img_vis = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_vis = (img_vis * 0.5) + 0.5

    # Collassa i 14 canali in un'unica mappa termica 2D
    heatmap_vis = heatmaps.sum(axis=0)

    # Plot e Salvataggio
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_vis)
    ax[0].set_title("Input Target (512x512)")
    ax[0].axis("off")
    
    ax[1].imshow(img_vis)
    ax[1].imshow(heatmap_vis, cmap="jet", alpha=0.5)
    ax[1].set_title("Heatmap Interne della ResNet18")
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Heatmap di diagnostica salvata in: {output_path}")
    plt.show()

if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    trex_path = os.path.join(_base_dir, "tests", "horse.jpg")
    pesi = os.path.join(_base_dir, "training_outputs_2", "Weights", "best_keypoint_detector.pth")
    output_path = os.path.join(_base_dir, "outputs", "debug_heatmap_trex.jpg")

    # Crea la cartella se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    salva_heatmap_inferenza(trex_path, pesi, output_path)