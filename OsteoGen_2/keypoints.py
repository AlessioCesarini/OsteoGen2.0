import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

nomi_punti = [
    "punta_muso", "retro_cranio", "base_collo", "spalla",
    "gomito_ant_1", "gomito_ant_2", "zampa_ant", "dorso", "anca",
    "ginocchio_post_1","ginocchio_post_2", "zampa_post", "base_coda", "punta_coda"
]

def annota_immagine(img_path, json_path):
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(f"{os.path.basename(img_path)}\nClicca i 12 punti in ordine.\n(Clicca in alto a sinistra [0,0] per saltare un punto)")
    
    # Raccoglie esattamente 12 click. Tasto destro per rimuovere l'ultimo click.
    coords = plt.ginput(n=14, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3)
    plt.close(fig)

    annotazioni = {}
    for nome, coord in zip(nomi_punti, coords):
        # Se si clicca vicino all'origine (es. x<10 e y<10), salviamo come null
        if coord[0] < 10 and coord[1] < 10:
            annotazioni[nome] = None
        else:
            annotazioni[nome] = coord 

    with open(json_path, 'w') as f:
        json.dump(annotazioni, f, indent=4)

if __name__ == "__main__":
    # Percorsi aggiornati alla tua struttura
    cartella_immagini = os.path.join("Data", "Processed", "Input_X")
    cartella_output = os.path.join("Data", "Processed", "Labels_X")
    
    # Crea la cartella dei JSON se non esiste
    os.makedirs(cartella_output, exist_ok=True)
    
    # Cerca jpg e png nella cartella degli scheletri
    immagini = glob.glob(os.path.join(cartella_immagini, "*.jpg")) + \
               glob.glob(os.path.join(cartella_immagini, "*.png"))
    
    for img_path in sorted(immagini):
        nome_file = os.path.basename(img_path)
        nome_base = os.path.splitext(nome_file)[0]
        json_path = os.path.join(cartella_output, f"{nome_base}.json")
        
        # Salta le immagini già annotate per permetterti di fare pause
        if os.path.exists(json_path):
            print(f"Skipping {nome_file}, JSON già esistente.")
            continue
            
        print(f"Annotando: {nome_file}")
        annota_immagine(img_path, json_path)