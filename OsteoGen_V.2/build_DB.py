"""
build_DB.py

Costruisce il database geometrico usato dal retrieval (compatibility.py).

v2: per gli animali del dataset di training usiamo le coordinate annotate a
mano in data/processed/Labels_X (ground truth), NON l'inferenza della rete.
La rete serve a leggere keypoint su immagini MAI viste (il fossile target),
non su immagini su cui abbiamo gia' l'annotazione esatta: usare l'inferenza
qui introduceva solo rumore inutile nel database e rendeva build_DB.py
dipendente da pesi gia' allenati.
"""
import os
import json

from geometry import (
    NOMI_PUNTI, SEGMENTI, normalizza_keypoints, lunghezze_segmenti,
    stima_fattore_calibrazione_cranio,
)
from dataset_paths import to_relative


# Extinct/target species kept in input_x + Labels_X only to give the
# keypoint DETECTOR a few more body plans to train on (see train.py):
# they are never real "donors" (no living photo, no texture to blend),
# so they must stay out of the retrieval database and out of the
# skull/torso calibration statistic below.
ESCLUSI_DAL_DB = {"trex.png", "brachiosauro.png", "triceratops.png"}


def costruisci_database():
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir_x = os.path.join(_base_dir, "data", "processed", "input_x")
    img_dir_y = os.path.join(_base_dir, "data", "processed", "target_y")
    json_dir_x = os.path.join(_base_dir, "data", "processed", "Labels_X")
    out_json = os.path.join(_base_dir, "data", "processed", "geometric_database.json")

    immagini = sorted(f for f in os.listdir(img_dir_x)
                       if f.endswith(('.png', '.jpg')) and f not in ESCLUSI_DAL_DB)
    database = {}
    saltati = []

    # Calibrazione: rapporto medio cranio/torso su tutto il dataset, usato
    # per normalizzare in modo coerente gli animali privi di torso annotato
    # (vedi geometry.trova_ancora_scala).
    tutte_le_coords = []
    for img_name in immagini:
        base_name = os.path.splitext(img_name)[0]
        path_json = os.path.join(json_dir_x, f"{base_name}.json")
        if os.path.exists(path_json):
            with open(path_json, 'r', encoding='utf-8') as f:
                tutte_le_coords.append(json.load(f))
    fattore_cranio = stima_fattore_calibrazione_cranio(tutte_le_coords)
    print(f"Fattore di calibrazione cranio/torso stimato sul dataset: {fattore_cranio:.4f}")

    print(f"Costruzione database geometrico su {len(immagini)} animali (da ground truth)...")

    for img_name in immagini:
        base_name = os.path.splitext(img_name)[0]
        path_x = os.path.join(img_dir_x, img_name)
        path_y = os.path.join(img_dir_y, img_name)
        path_json = os.path.join(json_dir_x, f"{base_name}.json")

        if not os.path.exists(path_json):
            print(f"  [!] Nessuna annotazione per {img_name}, salto.")
            saltati.append(img_name)
            continue

        with open(path_json, 'r', encoding='utf-8') as f:
            coords_raw = json.load(f)
        # Il JSON annotato puo' contenere liste [x, y]; le normalizziamo a
        # tuple per coerenza con il resto della pipeline.
        coords = {nome: (tuple(coords_raw[nome]) if coords_raw.get(nome) is not None else None)
                  for nome in NOMI_PUNTI}

        coords_normalizzati, ancora = normalizza_keypoints(coords, fattore_cranio)
        if ancora is None:
            print(f"  [!] {img_name}: nessuna ancora di scala disponibile (torso/cranio assenti), salto.")
            saltati.append(img_name)
            continue

        lunghezze = lunghezze_segmenti(coords)

        database[img_name] = {
            # Relative to the project root (forward-slash, portable across
            # OSes/machines) - never an absolute path: geometric_database.json
            # is committed to the repo and read on whatever machine runs the
            # pipeline, not just the one that built it.
            "path_scheletro": to_relative(path_x),
            "path_texture": to_relative(path_y),
            "coordinate_grezze": coords,
            "coordinate_normalizzate": coords_normalizzati,
            "ancora_scala": ancora,
            "segmentazione": {
                parte: {"punti": punti, "lunghezza_base": lunghezze[parte]}
                for parte, punti in SEGMENTI.items()
            },
        }
        print(f"  Processato: {img_name} (ancora: {ancora})")

    output = {
        "_meta": {"fattore_calibrazione_cranio": fattore_cranio},
        "animali": database,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)

    print(f"\nDatabase completato: {len(database)} animali salvati in {out_json}")
    if saltati:
        print(f"Animali saltati per dati insufficienti: {saltati}")


if __name__ == "__main__":
    costruisci_database()
