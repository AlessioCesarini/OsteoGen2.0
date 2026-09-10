"""
build_DB.py

Builds the geometric database used by retrieval (compatibility.py).

v2: for the training-set animals we use the hand-annotated coordinates in
data/processed/Labels_X (ground truth), NOT the network's inference. The
network is meant to read keypoints on images it has NEVER seen (the target
fossil), not on images we already have exact annotations for: using
inference here would only introduce needless noise into the database and
make build_DB.py depend on already-trained weights.
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

    # Calibration: average skull/torso ratio over the whole dataset, used
    # to consistently normalize animals with no annotated torso (see
    # geometry.trova_ancora_scala).
    tutte_le_coords = []
    for img_name in immagini:
        base_name = os.path.splitext(img_name)[0]
        path_json = os.path.join(json_dir_x, f"{base_name}.json")
        if os.path.exists(path_json):
            with open(path_json, 'r', encoding='utf-8') as f:
                tutte_le_coords.append(json.load(f))
    fattore_cranio = stima_fattore_calibrazione_cranio(tutte_le_coords)
    print(f"Skull/torso calibration factor estimated on the dataset: {fattore_cranio:.4f}")

    print(f"Building the geometric database for {len(immagini)} animals (from ground truth)...")

    for img_name in immagini:
        base_name = os.path.splitext(img_name)[0]
        path_x = os.path.join(img_dir_x, img_name)
        path_y = os.path.join(img_dir_y, img_name)
        path_json = os.path.join(json_dir_x, f"{base_name}.json")

        if not os.path.exists(path_json):
            print(f"  [!] No annotation for {img_name}, skipping.")
            saltati.append(img_name)
            continue

        with open(path_json, 'r', encoding='utf-8') as f:
            coords_raw = json.load(f)
        # The annotated JSON may contain [x, y] lists; normalize them to
        # tuples for consistency with the rest of the pipeline.
        coords = {nome: (tuple(coords_raw[nome]) if coords_raw.get(nome) is not None else None)
                  for nome in NOMI_PUNTI}

        coords_normalizzati, ancora = normalizza_keypoints(coords, fattore_cranio)
        if ancora is None:
            print(f"  [!] {img_name}: no scale anchor available (torso/skull both missing), skipping.")
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
        print(f"  Processed: {img_name} (anchor: {ancora})")

    output = {
        "_meta": {"fattore_calibrazione_cranio": fattore_cranio},
        "animali": database,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)

    print(f"\nDatabase complete: {len(database)} animals saved to {out_json}")
    if saltati:
        print(f"Animals skipped for insufficient data: {saltati}")


if __name__ == "__main__":
    costruisci_database()
