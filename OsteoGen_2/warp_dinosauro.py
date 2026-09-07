import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def applica_texture_parte(img_src, canvas_dst, punti_src_raw, punti_dst_raw):
    # 0. FILTRO PUNTI MANCANTI
    punti_src_puliti = []
    punti_dst_puliti = []
    
    for p_src, p_dst in zip(punti_src_raw, punti_dst_raw):
        if p_src is not None and p_dst is not None:
            punti_src_puliti.append(p_src)
            punti_dst_puliti.append(p_dst)
            
    if len(punti_src_puliti) < 2:
        return canvas_dst

    # 1. ACQUISIZIONE DIRETTA COORDINATE
    h_dst, w_dst = canvas_dst.shape[:2]
    h_src, w_src = img_src.shape[:2]
    
    punti_src = np.array(punti_src_puliti, dtype=np.float32)
    punti_dst = np.array(punti_dst_puliti, dtype=np.float32)
    punti_src_int = np.int32(punti_src)
    
    # 2. ISOLAMENTO ANATOMICO (Maschera Dinamica)
    mask_src = np.zeros((h_src, w_src), dtype=np.uint8)
    
    if len(punti_src_int) >= 3:
        hull_src = cv2.convexHull(punti_src_int)
        cv2.fillConvexPoly(mask_src, hull_src, 255)
    else:
        # Calcola uno spessore dinamico (es. 15% della lunghezza della coda)
        pt1 = tuple(punti_src_int[0])
        pt2 = tuple(punti_src_int[1])
        distanza = np.linalg.norm(punti_src[0] - punti_src[1])
        spessore_dinamico = max(5, int(distanza * 0.15))
        cv2.line(mask_src, pt1, pt2, 255, thickness=spessore_dinamico) 
    
    texture_isolata = cv2.bitwise_and(img_src, img_src, mask=mask_src)
    
    # 3. DEFORMAZIONE SPAZIALE (Warping Rigido Forzato)
    # Rimuoviamo estimateAffine2D per impedire lo stretching infinito.
    # estimateAffinePartial2D esegue SOLO rotazione, scala e traslazione.
    matrix, inliers = cv2.estimateAffinePartial2D(punti_src, punti_dst)
    
    if matrix is None:
        return canvas_dst
        
    texture_warped = cv2.warpAffine(texture_isolata, matrix, (w_dst, h_dst), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_warped = cv2.warpAffine(mask_src, matrix, (w_dst, h_dst), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # 4. FUSIONE (Blending)
    mask_warped_3c = cv2.cvtColor(mask_warped, cv2.COLOR_GRAY2BGR) / 255.0
    canvas_aggiornato = (texture_warped * mask_warped_3c + canvas_dst * (1.0 - mask_warped_3c)).astype(np.uint8)
    
    return canvas_aggiornato

def genera_chimera(match_dict, db_json_path, trex_coords, trex_path):
    # Legge il file del target e crea un canvas dinamico basato sulle sue dimensioni reali
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
        
        punti_src = [dati_animale["coordinate_grezze"].get(p) for p in nomi_punti_parte]
        punti_dst = [trex_coords.get(p) for p in nomi_punti_parte]
        
        canvas = applica_texture_parte(img_animale, canvas, punti_src, punti_dst)

    output_dir = r"C:\Users\alexc\Desktop\OsteoGen_2\outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "trex_chimera_render.jpg")
    cv2.imwrite(output_file_path, canvas)
    print(f"Immagine salvata con successo in: {output_file_path}")

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas_rgb)
    plt.title("Dinosauro Renderizzato (Composizione Texture)")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    match_trovati = {
        "testa": "salamandra.png",
        "torso": "airone.png",
        "arto_anteriore": "scimpanze.png",
        "arto_posteriore": "echidna.png",
        "coda": "mucca.png"
    }
    from inference import estrai_coordinate
    
    # ASSICURATI CHE QUESTO FILE SIA LO SCHELETRO DEL T-REX, NON LA FOTO DI UN KIWI
    trex_path = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\tests\t-rex.jpg"
    pesi = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\training_outputs_2\Weights\best_keypoint_detector.pth"
    db_path = r"C:\Users\alexc\Desktop\OsteoGen2.0\OsteoGen2.0\OsteoGen_2\data\processed\geometric_database.json"

    print("Estrazione coordinate T-Rex...")
    coords_trex = estrai_coordinate(trex_path, pesi, threshold=0.02)
    
    print("Generazione render in corso...")
    genera_chimera(match_trovati, db_path, coords_trex, trex_path)