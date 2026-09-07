import cv2
import numpy as np
import os

def prepara_immagine_target(input_path, output_path, target_size=512):
    # Carica l'immagine scaricata da internet
    img = cv2.imread(input_path)
    if img is None:
        print(f"Errore: Impossibile caricare l'immagine da {input_path}")
        return

    h, w = img.shape[:2]
    
    # Calcola il fattore di scala per farla rientrare in 512x512 senza deformarla
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Ridimensiona mantenendo l'aspect ratio
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Crea un canvas completamente nero (512x512)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calcola gli offset per centrare l'immagine ridimensionata sul canvas nero
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    # Incolla l'immagine centrata
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    # Salva il risultato pronto per l'inferenza
    cv2.imwrite(output_path, canvas)
    print(f"Immagine pre-processata salvata in: {output_path}")

if __name__ == "__main__":
    # Inserisci il percorso della foto scaricata
    immagine_grezza = r"C:\Users\alexc\Desktop\OsteoGen_2\input\t-rex.jpg"
    
    # Questo sarà il file che passerai al tuo script di retrieval
    immagine_pronta = r"C:\Users\alexc\Desktop\OsteoGen_2\tests\t-rex.jpg"
    
    prepara_immagine_target(immagine_grezza, immagine_pronta)