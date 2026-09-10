import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

nomi_punti = [
    "punta_muso", "retro_cranio", "base_collo", "spalla",
    "gomito_ant_1", "gomito_ant_2", "zampa_ant", "dorso", "anca",
    "ginocchio_post_1", "ginocchio_post_2", "zampa_post", "base_coda", "punta_coda"
]

def annota_immagine(img_path, json_path):
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(f"{os.path.basename(img_path)}\nClick the 14 points in order.\n(Click near the top-left corner [0,0] to skip a point)")

    # Collects exactly 14 clicks. Right-click to undo the last one.
    coords = plt.ginput(n=14, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3)
    plt.close(fig)

    annotazioni = {}
    for nome, coord in zip(nomi_punti, coords):
        # A click near the origin (x<10 and y<10) marks the point as absent.
        if coord[0] < 10 and coord[1] < 10:
            annotazioni[nome] = None
        else:
            annotazioni[nome] = coord

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(annotazioni, f, indent=4)

if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    cartella_immagini = os.path.join(_base_dir, "data", "processed", "input_x")
    cartella_output = os.path.join(_base_dir, "data", "processed", "Labels_X")

    os.makedirs(cartella_output, exist_ok=True)

    # Look for skeleton images (jpg and png).
    immagini = glob.glob(os.path.join(cartella_immagini, "*.jpg")) + \
               glob.glob(os.path.join(cartella_immagini, "*.png"))

    for img_path in sorted(immagini):
        nome_file = os.path.basename(img_path)
        nome_base = os.path.splitext(nome_file)[0]
        json_path = os.path.join(cartella_output, f"{nome_base}.json")

        # Skip already-annotated images, so this can be resumed across sessions.
        if os.path.exists(json_path):
            print(f"Skipping {nome_file}, JSON already exists.")
            continue

        print(f"Annotating: {nome_file}")
        annota_immagine(img_path, json_path)
