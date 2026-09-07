import os
import json
import numpy as np
from inference import estrai_coordinate # Assicurati che il file precedente si chiami inference.py

def calcola_distanza(p1, p2):
    """Calcola la distanza euclidea tra due punti, se esistono."""
    if p1 is None or p2 is None:
        return 0.0
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def costruisci_database():
    img_dir_x = r"C:\Users\alexc\Desktop\OsteoGen_2\data\processed\input_x"
    img_dir_y = r"C:\Users\alexc\Desktop\OsteoGen_2\data\processed\target_y"
    pesi_modello = r"C:\Users\alexc\Desktop\OsteoGen_2\training_outputs_2\Weights\best_keypoint_detector.pth"
    out_json = r"C:\Users\alexc\Desktop\OsteoGen_2\data\processed\geometric_database.json"

    immagini = [f for f in os.listdir(img_dir_x) if f.endswith(('.png', '.jpg'))]
    database = {}

    print(f"Avvio estrazione coordinate su {len(immagini)} animali...")

    for img_name in immagini:
        path_x = os.path.join(img_dir_x, img_name)
        
        # Supponiamo che il target Y abbia lo stesso nome
        path_y = os.path.join(img_dir_y, img_name) 
        
        # Estrae i 14 keypoint usando la nostra ResNet18
        coords = estrai_coordinate(path_x, pesi_modello)
        
        # Segmentazione in parti anatomiche
        parti = {
            "testa": {
                "punti": ["punta_muso", "retro_cranio", "base_collo"],
                "lunghezza_base": calcola_distanza(coords.get("punta_muso"), coords.get("base_collo"))
            },
            "torso": {
                "punti": ["base_collo", "spalla", "dorso", "anca", "base_coda"],
                "lunghezza_base": calcola_distanza(coords.get("base_collo"), coords.get("base_coda"))
            },
            "arto_anteriore": {
                "punti": ["spalla", "gomito_ant_1", "gomito_ant_2", "zampa_ant"],
                "lunghezza_base": calcola_distanza(coords.get("spalla"), coords.get("zampa_ant"))
            },
            "arto_posteriore": {
                "punti": ["anca", "ginocchio_post_1", "ginocchio_post_2", "zampa_post"],
                "lunghezza_base": calcola_distanza(coords.get("anca"), coords.get("zampa_post"))
            },
            "coda": {
                "punti": ["base_coda", "punta_coda"],
                "lunghezza_base": calcola_distanza(coords.get("base_coda"), coords.get("punta_coda"))
            }
        }

        # Salva la struttura dell'animale
        database[img_name] = {
            "path_scheletro": path_x,
            "path_texture": path_y,
            "coordinate_grezze": coords,
            "segmentazione": parti
        }
        
        print(f"Processato: {img_name}")

    with open(out_json, 'w') as f:
        json.dump(database, f, indent=4)
        
    print(f"\nDatabase completato! Salvato in: {out_json}")

if __name__ == "__main__":
    costruisci_database()