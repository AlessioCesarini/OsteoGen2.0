import os
import matplotlib.pyplot as plt
from heatmap import SkeletonKeypointDataset # Assicurati che il file precedente si chiami heatmap.py

def salva_controlli_visivi():
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(_base_dir, "data", "processed", "input_x")
    json_dir = os.path.join(_base_dir, "data", "processed", "Labels_X")
    out_dir = os.path.join(_base_dir, "data", "processed", "Heatmap_Checks")
    
    # Crea la cartella di output se non esiste
    os.makedirs(out_dir, exist_ok=True)

    # Inizializza il dataset
    dataset = SkeletonKeypointDataset(img_dir=img_dir, json_dir=json_dir)

    print(f"Inizio salvataggio di {len(dataset)} immagini di controllo...")
    
    for i in range(len(dataset)):
        # Estrae un singolo campione
        img_tensor, heatmap_tensor = dataset[i]
        
        # Denormalizza l'immagine per la visualizzazione
        img_vis = img_tensor.numpy().transpose(1, 2, 0)
        img_vis = (img_vis * 0.5) + 0.5
        
        # Somma le 14 heatmap per vederle tutte insieme
        heatmap_vis = heatmap_tensor.numpy().sum(axis=0)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_vis)
        ax[0].set_title("Input Originale")
        ax[0].axis("off")
        
        ax[1].imshow(img_vis)
        ax[1].imshow(heatmap_vis, cmap="jet", alpha=0.5)
        ax[1].set_title("Target: 14 Heatmap Gaussiane")
        ax[1].axis("off")
        
        # Salva l'immagine nella cartella Checks
        nome_file = dataset.img_names[i]
        plt.savefig(os.path.join(out_dir, nome_file), bbox_inches='tight')
        
        # Chiude la figura per liberare la memoria RAM
        plt.close(fig) 
        
        print(f"Salvata verifica per: {nome_file}")

    print(f"\nFinito! Controlla la cartella: {out_dir}")

if __name__ == "__main__":
    salva_controlli_visivi()