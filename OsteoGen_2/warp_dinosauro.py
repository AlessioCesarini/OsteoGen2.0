import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def applica_texture_parte(img_src, canvas_dst, punti_src_512, punti_dst_512):
    # 0. FILTRO PUNTI MANCANTI E ALLINEAMENTO
    valid_src = []
    valid_dst = []
    
    for p_src, p_dst in zip(punti_src_512, punti_dst_512):
        if p_src is not None and p_dst is not None:
            valid_src.append(p_src)
            valid_dst.append(p_dst)
            
    # OpenCV richiede almeno 3 punti non allineati per una trasformazione affine completa
    if len(valid_src) < 3:
        return canvas_dst

    # 1. RIPROPORZIONAMENTO COORDINATE (Scaling)
    h_src, w_src = img_src.shape[:2]
    h_dst, w_dst = canvas_dst.shape[:2]
    
    punti_src = np.array([(int(x * w_src / 512), int(y * h_src / 512)) for x, y in valid_src], dtype=np.float32)
    punti_dst = np.array([(int(x * w_dst / 512), int(y * h_dst / 512)) for x, y in valid_dst], dtype=np.float32)
    
    punti_src_int = np.int32(punti_src)
    
    # 2. ISOLAMENTO ANATOMICO (Convex Hull)
    mask_src = np.zeros((h_src, w_src), dtype=np.uint8)
    hull_src = cv2.convexHull(punti_src_int)
    cv2.fillConvexPoly(mask_src, hull_src, 255)
    
    texture_isolata = cv2.bitwise_and(img_src, img_src, mask=mask_src)
    
    # 3. DEFORMAZIONE SPAZIALE (Warping)
    matrix, inliers = cv2.estimateAffine2D(punti_src, punti_dst)
    
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

def genera_chimera(match_dict, db_json_path, trex_coords, output_size=(512, 512)):
    # Canvas nero vuoto per la chimera finale
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    with open(db_json_path, 'r') as f:
        db = json.load(f)

    # Itera sulle parti anatomiche e sui match trovati
    for parte, nome_animale in match_dict.items():
        if nome_animale is None:
            continue
            
        dati_animale = db[nome_animale]
        img_path_y = dati_animale["path_texture"]
        
        if not os.path.exists(img_path_y):
            print(f"Texture non trovata: {img_path_y}")
            continue
            
        img_animale = cv2.imread(img_path_y)
        
        # Recupera i nomi dei punti per questa specifica parte
        nomi_punti_parte = dati_animale["segmentazione"][parte.lower()]["punti"]
        
        # Estrae le coordinate (x,y)
        punti_src = [dati_animale["coordinate_grezze"].get(p) for p in nomi_punti_parte]
        punti_dst = [trex_coords.get(p) for p in nomi_punti_parte]
        
        # Deforma e fonde sul canvas
        canvas = applica_texture_parte(canvas, img_animale, punti_src, punti_dst)

    # Salvataggio automatico dell'immagine su disco (in formato BGR nativo di OpenCV)
    output_dir = r"C:\Users\alexc\Desktop\OsteoGen_2\outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "trex_chimera_render.jpg")
    cv2.imwrite(output_file_path, canvas)
    print(f"Immagine salvata con successo in: {output_file_path}")

    # Converte da BGR (OpenCV) a RGB per Matplotlib
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas_rgb)
    plt.title("Dinosauro Renderizzato (Composizione Texture)")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # INCOLLA QUI I RISULTATI DEL TUO SCRIPT PRECEDENTE
    match_trovati = {
        "testa": "uomo.png",
        "torso": "airone.png",
        "arto_anteriore": "struzzo.png",
        "arto_posteriore": "armadillo.png",
        "coda": "kiwi.png"
    }
    
    # Eseguiamo l'inferenza al volo per riavere le coordinate esatte del T-Rex
    from inference import estrai_coordinate
    trex_path = r"C:\Users\alexc\Desktop\OsteoGen_2\tests\t-rex.jpg"
    pesi = r"C:\Users\alexc\Desktop\OsteoGen_2\training_outputs_2\Weights\best_keypoint_detector.pth"
    db_path = r"C:\Users\alexc\Desktop\OsteoGen_2\data\processed\geometric_database.json"

    print("Estrazione coordinate T-Rex...")
    coords_trex = estrai_coordinate(trex_path, pesi, threshold=0.02)
    
    print("Generazione render in corso...")
    genera_chimera(match_trovati, db_path, coords_trex)