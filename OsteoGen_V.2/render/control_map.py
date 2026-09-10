"""
render/control_map.py

Formerly warp_dinosauro.py. Rigid convexHull/estimateAffine2D warping no
longer produces the final image (the triangular texture fragments glued
together in outputs/trex_chimera_render.jpg are proof: not a coherent
animal). It's still useful, though, to generate two inputs for the
generative stage in render/generate.py:

1. disegna_control_map: a white-line-on-black control map of the target
   skeleton, fed to a ControlNet (scribble/lineart) - constrains the
   generated render to respect the fossil's real pose/proportions.
2. genera_guida_grezza: the same texture-block collage as before, but used
   only as a starting image (init image) for a low-"strength" img2img: it
   gives the diffusion model a plausible color starting point (roughly
   where the legs, snout, etc. are) instead of pure noise, without
   expecting the collage itself to be the result.
"""
import cv2
import numpy as np
import os
import json
import sys

# Allows both `python render/control_map.py` and `import render.control_map`
# from the project root, reusing geometry.py which lives at the project
# level (not packaged under render/ since build_DB.py and compatibility.py
# also need it, outside this package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry import SEGMENTI


def disegna_control_map(coords, size=(512, 512), spessore=6):
    """Draws the target skeleton as white polylines on a black background:
    a simple control condition, style-independent (works for both
    illustrations and real fossil photos) because it depends only on the
    keypoint coordinates, not on the original pixels."""
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

    return canvas  # single channel; the caller converts as needed (RGB for ControlNet)


def _applica_texture_parte(img_src, canvas_dst, punti_src_raw, punti_dst_raw):
    """Isolates the donor's texture along the bone chain (a 'tubular' mask
    around each link) and warps it onto the target's points with a partial
    affine transform. Only for the rough guide (see above), not for the
    final output."""
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
    """donatori_per_segmento: dict part -> animal_name (the best donor,
    e.g. report['per_segmento'][parte]['donatori'][0]['animale']).
    Returns a BGR canvas: a rough collage, ONLY as an init-image for
    render/generate.py, not as the final output."""
    from dataset_paths import resolve_dataset_path

    h, w = target_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for parte, nome_animale in donatori_per_segmento.items():
        if nome_animale is None or nome_animale not in db:
            continue
        dati_animale = db[nome_animale]
        img_path_y = resolve_dataset_path(dati_animale.get("path_texture"), "target_y", nome_animale)
        if not os.path.exists(img_path_y):
            continue

        img_animale = cv2.imread(img_path_y)
        nomi_punti_parte = SEGMENTI[parte]
        punti_src_raw = [dati_animale["coordinate_grezze"].get(p) for p in nomi_punti_parte]
        punti_dst_raw = [target_coords.get(p) for p in nomi_punti_parte]
        canvas = _applica_texture_parte(img_animale, canvas, punti_src_raw, punti_dst_raw)

    return canvas


if __name__ == "__main__":
    # Smoke test: draws only the control map (doesn't need donor textures).
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    esempio_json = os.path.join(_base_dir, "..", "data", "processed", "Labels_X", "cavallo.json")
    with open(esempio_json, encoding='utf-8') as f:
        coords = {k: (tuple(v) if v else None) for k, v in json.load(f).items()}

    mappa = disegna_control_map(coords)
    out_path = os.path.join(_base_dir, "..", "outputs", "control_map_test.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, mappa)
    print(f"Test control map saved to {out_path}")
