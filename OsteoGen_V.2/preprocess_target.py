"""
preprocess_target.py

Prepara un'immagine "grezza" (scaricata da internet, o una vera foto di
fossile) per l'inferenza, in 3 passi opzionali:

1. rimuovi_sfondo: isola il soggetto da uno sfondo non nero (una vera foto
   di scavo/museo non arriva mai gia' su sfondo nero come le illustrazioni
   di training). Usa `rembg` se installato (rete U2Net, buona qualita'),
   altrimenti un fallback euristico grezzo via GrabCut.
2. canonicalizza_bordi: riduce il divario di dominio tra le illustrazioni
   stilizzate del training set e una fotografia reale, enfatizzando i
   contorni ossei invece dei dettagli fotografici (illuminazione, texture
   della pietra, ecc.). Lo stesso tipo di trasformazione va applicato anche
   in fase di training come augmentation (vedi heatmap.py) cosi' la rete
   impara a riconoscere entrambi gli stili.
3. prepara_immagine_target: il letterboxing gia' esistente (invariato),
   sempre l'ultimo passo prima di salvare/passare il file alla rete.
"""
import cv2
import numpy as np
import os


def rimuovi_sfondo(img_bgr):
    """Ritorna un'immagine BGR con lo sfondo sostituito da nero. Se non
    riesce a isolare un soggetto plausibile, ritorna l'immagine originale
    invariata (meglio un fondo non pulito che un soggetto cancellato)."""
    try:
        from rembg import remove
        import io
        from PIL import Image

        ok, buf = cv2.imencode(".png", img_bgr)
        risultato = remove(buf.tobytes())
        rgba = np.array(Image.open(io.BytesIO(risultato)).convert("RGBA"))
        alpha = rgba[:, :, 3:4] / 255.0
        rgb = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        return (rgb * alpha).astype(np.uint8)
    except ImportError:
        print("[preprocess] 'rembg' not installed, using the GrabCut fallback "
              "(less accurate: add 'rembg' to requirements.txt for a better result).")
        return _rimuovi_sfondo_grabcut(img_bgr)
    except Exception as e:
        print(f"[preprocess] Background removal failed ({e}), using the original image.")
        return img_bgr


def _rimuovi_sfondo_grabcut(img_bgr):
    """Fallback senza dipendenze extra: assume che il soggetto sia
    approssimativamente centrato e occupi la porzione centrale
    dell'inquadratura, inizializza GrabCut con quel rettangolo."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margine_w, margine_h = int(w * 0.08), int(h * 0.08)
    rect = (margine_w, margine_h, w - 2 * margine_w, h - 2 * margine_h)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return img_bgr
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return img_bgr * mask2[:, :, np.newaxis]


def canonicalizza_bordi(img_bgr, alpha=0.5):
    """Mescola l'immagine originale con la sua mappa dei bordi (Canny), per
    avvicinare stilisticamente una foto reale alle illustrazioni vettoriali
    del training set. alpha=0 -> immagine originale, alpha=1 -> soli bordi."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bordi = cv2.Canny(gray, 60, 160)
    bordi_bgr = cv2.cvtColor(bordi, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img_bgr, 1 - alpha, bordi_bgr, alpha, 0)


def prepara_immagine_target(input_path, output_path, target_size=512,
                             rimuovi_sfondo_flag=False, canonicalizza_flag=False):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not load the image from {input_path}")
        return None

    if rimuovi_sfondo_flag:
        img = rimuovi_sfondo(img)
    if canonicalizza_flag:
        img = canonicalizza_bordi(img)

    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    cv2.imwrite(output_path, canvas)
    print(f"Preprocessed image saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    immagine_grezza = os.path.join(_base_dir, "input", "horse.jpg")
    immagine_pronta = os.path.join(_base_dir, "tests", "horse.jpg")

    prepara_immagine_target(immagine_grezza, immagine_pronta,
                             rimuovi_sfondo_flag=True, canonicalizza_flag=False)
