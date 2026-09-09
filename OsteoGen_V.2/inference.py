import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from resnet import ResNet18KeypointDetector

def estrai_coordinate(image_path, model_path, img_size=512, threshold=0.2, ritorna_confidenza=False):
    # Setup del device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inizializza e carica il modello
    model = ResNet18KeypointDetector(num_keypoints=14).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Pre-processing identico al training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    img_tensor = transform(image).unsqueeze(0).to(device)

    nomi_punti = [
        "punta_muso", "retro_cranio", "base_collo", "spalla",
        "gomito_ant_1", "gomito_ant_2", "zampa_ant", "dorso", "anca",
        "ginocchio_post_1", "ginocchio_post_2", "zampa_post", "base_coda", "punta_coda"
    ]

    coordinate_estratte = {}
    confidenze = {}

    with torch.no_grad():
        output = model(img_tensor) # Shape: (1, 14, 512, 512)
        heatmaps = output.squeeze(0).cpu().numpy()

    for i, nome in enumerate(nomi_punti):
        heatmap = heatmaps[i]

        # Trova il valore massimo e la sua posizione
        max_val = np.max(heatmap)
        confidenze[nome] = float(max_val)
        if max_val > threshold:
            # np.unravel_index converte l'indice 1D in coordinate 2D (y, x)
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

            # Riproporziona le coordinate alla dimensione originale dell'immagine
            x_orig = int((x / img_size) * original_size[0])
            y_orig = int((y / img_size) * original_size[1])

            coordinate_estratte[nome] = (x_orig, y_orig)
        else:
            coordinate_estratte[nome] = None # Punto non trovato o assente

    if ritorna_confidenza:
        return coordinate_estratte, confidenze
    return coordinate_estratte

if __name__ == "__main__":
    # Test paths
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    pesi = os.path.join(_base_dir, "training_outputs_2", "Weights", "best_keypoint_detector.pth")

    # Pick a random skeleton photo from the dataset to test with
    immagine_test = os.path.join(_base_dir, "data", "processed", "input_x", "aquila.png")

    if os.path.exists(immagine_test) and os.path.exists(pesi):
        coords = estrai_coordinate(immagine_test, pesi)
        print("Coordinates extracted by the network:")
        for punto, val in coords.items():
            print(f"{punto}: {val}")
    else:
        print("Point to a valid test image path.")