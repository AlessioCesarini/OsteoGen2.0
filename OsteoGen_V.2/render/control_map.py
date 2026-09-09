"""
render/control_map.py

Ex warp_dinosauro.py. Il warping rigido a base di convexHull/estimateAffine2D
NON produce piu' l'immagine finale (i frammenti triangolari incollati in
outputs/trex_chimera_render.jpg ne sono la prova: non e' un animale
coerente). Resta pero' utile come generatore di due input per lo stadio
generativo in render/generate.py:

1. disegna_control_map: una mappa di controllo linea-bianca-su-nero dello
   scheletro target, da dare in pasto a un ControlNet (scribble/lineart) -
   vincola il render generato a rispettare la posa/proporzioni reali del
   fossile.
2. genera_guida_grezza: lo stesso collage a blocchi di texture di prima, ma
   usato solo come immagine di partenza (init image) per un img2img a bassa
   "strength": da' al modello di diffusione un punto di partenza cromatico
   plausibile (dove sono grosso modo le zampe, il muso, ecc.) invece di
   puro rumore, senza pretendere che il collage stesso sia il risultato.
"""
import cv2
import numpy as np
import os
import json
import sys

# Permette sia `python render/control_map.py` sia `import render.control_map`
# da OsteoGen_2/, riusando geometry.py che vive a livello di progetto (non
# e' stato spacchettato in render/ perche' serve anche a build_DB.py e
# compatibility.py fuori da questo package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry import SEGMENTI


def disegna_control_map(coords, size=(512, 512), spessore=6):
    """Disegna lo scheletro del target come polilinee bianche su sfondo
    nero: una condizione di controllo semplice, indipendente dallo stile
    (funziona sia per illustrazioni sia per foto di fossili reali) perche'
    dipende solo dalle coordinate dei keypoint, non dai pixel originali."""
    h, w = size[1], size[0]
    canvas = np.zeros((h, w), dtype=np.uint8)

    for parte, punti in SEGMENTI.items():
        pts = [coords.get(p) for p in punti]
        for p1, p2 in zip(pts, pts[1:]):
            if p1 is None or p2 is None:
                continue
            cv2.line(canvas, tuple(np.int32(p1)), tuple(np.int32(p2)), 255, thickness=spessore)

    for p in coords.values():
        if p is not None:
            cv2.circle(canvas, tuple(np.int32(p)), spessore, 255, -1)

    return canvas  # singolo canale, il caller lo converte come serve (RGB per ControlNet)


def _applica_texture_parte(img_src, canvas_dst, punti_src_raw, punti_dst_raw):
    """Isola la texture del donatore lungo la catena ossea (maschera
    'tubolare' attorno a ogni link) e la deforma sui punti del target con
    una trasformazione affine parziale. Solo per la guida grezza (v. sopra),
    non per l'output finale."""
    p_src_clean = [p for p, d in zip(punti_src_raw, punti_dst_raw) if p is not None and d is not None]
    p_dst_clean = [d for p, d in zip(punti_src_raw, punti_dst_raw) if p is not None and d is not None]

    if len(p_src_clean) < 2:
        return canvas_dst

    h_dst, w_dst = canvas_dst.shape[:2]
    h_src, w_src = img_src.shape[:2]

    pt_src = np.array(p_src_clean, dtype=np.float32)
    pt_dst = np.array(p_dst_clean, dtype=np.float32)
    pt_src_int = np.int32(pt_src)

    mask_src = np.zeros((h_src, w_src), dtype=np.uint8)
    for i in range(len(pt_src_int) - 1):
        pt1, pt2 = tuple(pt_src_int[i]), tuple(pt_src_int[i + 1])
        distanza = np.linalg.norm(pt_src[i] - pt_src[i + 1])
        spessore = max(15, int(distanza * 0.40))
        cv2.line(mask_src, pt1, pt2, 255, thickness=spessore)
        cv2.circle(mask_src, pt1, spessore // 2, 255, -1)
        cv2.circle(mask_src, pt2, spessore // 2, 255, -1)

    texture_isolata = cv2.bitwise_and(img_src, img_src, mask=mask_src)
    matrix, _ = cv2.estimateAffinePartial2D(pt_src, pt_dst)
    if matrix is None:
        return canvas_dst

    texture_warped = cv2.warpAffine(texture_isolata, matrix, (w_dst, h_dst), borderMode=cv2.BORDER_CONSTANT)
    mask_warped = cv2.warpAffine(mask_src, matrix, (w_dst, h_dst), borderMode=cv2.BORDER_CONSTANT)

    mask_3c = cv2.cvtColor(mask_warped, cv2.COLOR_GRAY2BGR) / 255.0
    return (texture_warped * mask_3c + canvas_dst * (1.0 - mask_3c)).astype(np.uint8)


def genera_guida_grezza(donatori_per_segmento, db, target_coords, target_shape):
    """donatori_per_segmento: dict parte -> nome_animale (il donatore
    migliore, es. report['per_segmento'][parte]['donatori'][0]['animale']).
    Ritorna un canvas BGR: collage grezzo, SOLO come init-image per
    render/generate.py, non come output finale."""
    h, w = target_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for parte, nome_animale in donatori_per_segmento.items():
        if nome_animale is None or nome_animale not in db:
            continue
        dati_animale = db[nome_animale]
        img_path_y = dati_animale["path_texture"]
        if not os.path.exists(img_path_y):
            continue

        img_animale = cv2.imread(img_path_y)
        nomi_punti_parte = SEGMENTI[parte]
        punti_src_raw = [dati_animale["coordinate_grezze"].get(p) for p in nomi_punti_parte]
        punti_dst_raw = [target_coords.get(p) for p in nomi_punti_parte]
        canvas = _applica_texture_parte(img_animale, canvas, punti_src_raw, punti_dst_raw)

    return canvas


if __name__ == "__main__":
    # Smoke test: disegna solo la control map (non richiede texture donatori)
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    esempio_json = os.path.join(_base_dir, "..", "data", "processed", "Labels_X", "cavallo.json")
    with open(esempio_json, encoding='utf-8') as f:
        coords = {k: (tuple(v) if v else None) for k, v in json.load(f).items()}

    mappa = disegna_control_map(coords)
    out_path = os.path.join(_base_dir, "..", "outputs", "control_map_test.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mappa)
    print(f"Control map di test salvata in {out_path}")
