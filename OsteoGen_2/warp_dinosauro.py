import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def applica_texture_parte(img_src, canvas_dst, punti_src_raw, punti_dst_raw):
    # Filtro punti mancanti
    p_src_clean = [p for p, d in zip(punti_src_raw, punti_dst_raw) if p is not None and d is not None]
    p_dst_clean = [d for p, d in zip(punti_src_raw, punti_dst_raw) if p is not None and d is not None]
            
    if len(p_src_clean) < 2:
        return canvas_dst

    h_dst, w_dst = canvas_dst.shape[:2]
    h_src, w_src = img_src.shape[:2]
    
    pt_src = np.array(p_src_clean, dtype=np.float32)
    pt_dst = np.array(p_dst_clean, dtype=np.float32)
    pt_src_int = np.int32(pt_src)
    
    # MASCHERA TUBOLARE (Segue l'anatomia A -> B)
    mask_src = np.zeros((h_src, w_src), dtype=np.uint8)
    
    # Traccia una linea spessa per ogni segmento osseo
    for i in range(len(pt_src_int) - 1):
        pt1 = tuple(pt_src_int[i])
        pt2 = tuple(pt_src_int[i+1])
        
        # Calcola la distanza tra giuntura A e B
        distanza = np.linalg.norm(pt_src[i] - pt_src[i+1])
        
        # Lo spessore della maschera (es. 40% della lunghezza dell'osso)
        spessore = max(15, int(distanza * 0.40)) 
        
        # Disegna il "tubo" di carne attorno all'osso
        cv2.line(mask_src, pt1, pt2, 255, thickness=spessore)
        
        # Disegna cerchi sulle giunture per smussare le articolazioni (gomiti, ginocchia)
        cv2.circle(mask_src, pt1, spessore // 2, 255, -1)
        cv2.circle(mask_src, pt2, spessore // 2, 255, -1)
    
    # Ritaglia i pixel esatti che seguono le ossa
    texture_isolata = cv2.bitwise_and(img_src, img_src, mask=mask_src)
    
    # Warping Rigido (assembla i blocchi sul dinosauro)
    matrix, inliers = cv2.estimateAffinePartial2D(pt_src, pt_dst)
    
    if matrix is None:
        return canvas_dst
        
    texture_warped = cv2.warpAffine(texture_isolata, matrix, (w_dst, h_dst), borderMode=cv2.BORDER_CONSTANT)
    mask_warped = cv2.warpAffine(mask_src, matrix, (w_dst, h_dst), borderMode=cv2.BORDER_CONSTANT)
    
    # Fusione
    mask_3c = cv2.cvtColor(mask_warped, cv2.COLOR_GRAY2BGR) / 255.0
    canvas_aggiornato = (texture_warped * mask_3c + canvas_dst * (1.0 - mask_3c)).astype(np.uint8)
    
    return canvas_aggiornato


def genera_chimera(match_dict, db_json_path, trex_coords, trex_path):
    # Canvas dinamico basato sulle dimensioni reali del T-Rex
    img_trex = cv2.imread(trex_path)
    h_t, w_t = img_trex.shape[:2]
    canvas = np.zeros((h_t, w_t, 3), dtype=np.uint8)

    with open(db_json_path, 'r') as f:
        db = json.load(f)

    for parte, nome_animale in match_dict.items():
        if nome_animale is None:
            continue
            
        dati_animale = db[nome_animale]
        img_path_y = dati_animale["path_texture"]
        
        if not os.path.exists(img_path_y):
            print(f"Texture non trovata: {img_path_y}")
            continue
            
        img_animale = cv2.imread(img_path_y)
        nomi_punti_parte = dati_animale["segmentazione"][parte.lower()]["punti"]
        
        punti_src_raw = [dati_animale["coordinate_grezze"].get(p) for p in nomi_punti_parte]
        punti_dst_raw = [trex_coords.get(p) for p in nomi_punti_parte]
        
        # La nuova funzione applica_texture_parte gestisce tutto internamente (sia >=3 punti che ==2 punti)
        canvas = applica_texture_parte(img_animale, canvas, punti_src_raw, punti_dst_raw)

    output_dir = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "trex_chimera_rigida.jpg")
    cv2.imwrite(output_file_path, canvas)
    print(f"Render salvato in: {output_file_path}")

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas_rgb)
    plt.title("Chimera (Assemblaggio Rigido)")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    match_trovati = {
        "testa": None,
        "torso": "airone.png",
        "arto_anteriore": None,
        "arto_posteriore": "echidna.png",
        "coda": "mucca.png"
    }
    
    from inference import estrai_coordinate
    
    # Percorsi aggiornati alla cartella OsteoGen2.0
    trex_path = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\tests\t-rex.jpg"
    pesi = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\training_outputs_2\Weights\best_keypoint_detector.pth"
    db_path = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\data\processed\geometric_database.json"

    print("Estrazione coordinate T-Rex...")
    coords_trex = estrai_coordinate(trex_path, pesi, threshold=0.02)
    
    print("Generazione render ad assemblaggio rigido in corso...")
    genera_chimera(match_trovati, db_path, coords_trex, trex_path)