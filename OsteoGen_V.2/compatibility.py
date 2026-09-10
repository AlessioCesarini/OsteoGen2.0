"""
compatibility.py

Replaces match_dinosauro.py: from a simple "closest wins" to a real
percentage compatibility report.

Two levels of similarity, both computed on normalized keypoints (see
geometry.normalizza_keypoints, invariant to scale/position):

1. compatibilita_specie: how much the target's whole silhouette resembles
   each animal in the DB (for the report "your fossil is 50% eagle, 30%
   chicken, ...").
2. compatibilita_per_segmento: for each of the 5 body parts, which animals
   are the most plausible texture donors (used by the renderer).

Distances are converted to percentages with a softmax: the closer an
animal is geometrically to the target, the more weight it gets, but every
compatible animal contributes to the score (not just the top match).
Default temperatures are empirically calibrated on the 65-species dataset
(see notes in the repo/plan): with target = one of the DB animals itself,
its self-match stays around 50-100% and the next most plausible neighbors
land right after, instead of an almost-uniform ranking.
"""
import json
import os
import numpy as np

from geometry import (
    NOMI_PUNTI, SEGMENTI, normalizza_keypoints, lunghezze_segmenti_normalizzate,
)

TEMPERATURE_SPECIE_DEFAULT = 0.08
TEMPERATURE_SEGMENTO_DEFAULT = 0.06


def carica_database(path):
    """Returns (animali, fattore_calibrazione_cranio). `animali` is the dict
    filename -> geometric data; the factor is used to normalize the target
    consistently with the DB when it lacks a torso (see geometry.py)."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    meta = raw.get("_meta", {})
    animali = raw.get("animali", raw)  # backward-compat with old flat DBs
    return animali, meta.get("fattore_calibrazione_cranio")


def _distanza_normalizzata(vec_a, vec_b, min_punti_comuni):
    """RMS distance between the normalized keypoints shared by two
    entities. Returns (distance, n_shared_points); distance=None if too
    few points are shared for a reliable comparison."""
    diffs_sq = []
    for nome in NOMI_PUNTI:
        pa, pb = vec_a.get(nome), vec_b.get(nome)
        if pa is not None and pb is not None:
            diffs_sq.append((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)

    n_comuni = len(diffs_sq)
    if n_comuni < min_punti_comuni:
        return None, n_comuni
    return float(np.sqrt(np.mean(diffs_sq))), n_comuni


def _softmax_percentuali(coppie_nome_distanza, temperature):
    """coppie_nome_distanza: list of (name, distance). Returns a list of
    (name, pct, distance) sorted by descending pct, pct summing to 100."""
    if not coppie_nome_distanza:
        return []
    distanze = np.array([d for _, d in coppie_nome_distanza])
    pesi = np.exp(-distanze / temperature)
    tot = pesi.sum()
    pesi = pesi / tot if tot > 0 else np.full_like(pesi, 1.0 / len(pesi))
    out = [(nome, float(p) * 100.0, d) for (nome, d), p in zip(coppie_nome_distanza, pesi)]
    out.sort(key=lambda r: -r[1])
    return out


def compatibilita_specie(coords_target, db, fattore_calibrazione_cranio=None,
                          temperature=TEMPERATURE_SPECIE_DEFAULT, min_punti_comuni=3):
    """Compatibility ranking of the whole target against every animal in
    the DB. Lower temperature = a more 'peaked' distribution on fewer
    animals; higher = scores spread out more."""
    vec_target, ancora = normalizza_keypoints(coords_target, fattore_calibrazione_cranio)
    if ancora is None:
        raise ValueError(
            "The target has no valid scale anchor (missing both the "
            "torso base_collo->base_coda and the skull punta_muso->retro_cranio): "
            "cannot compute compatibility."
        )

    coppie = []
    dettagli = {}
    for nome_animale, dati in db.items():
        d, n_comuni = _distanza_normalizzata(vec_target, dati["coordinate_normalizzate"], min_punti_comuni)
        if d is None:
            continue
        coppie.append((nome_animale, d))
        dettagli[nome_animale] = n_comuni

    ranking = _softmax_percentuali(coppie, temperature)
    return [
        {"animale": nome, "compatibilita_pct": round(pct, 2),
         "distanza": round(d, 4), "punti_comuni": dettagli[nome]}
        for nome, pct, d in ranking
    ]


def compatibilita_per_segmento(coords_target, db, fattore_calibrazione_cranio=None,
                                temperature=TEMPERATURE_SEGMENTO_DEFAULT):
    """For each anatomical segment present in the target, a % ranking of
    the most compatible donors in terms of proportion relative to the
    torso/skull (not absolute shape: a pigeon and an ostrich can have
    proportionally similar wings despite being wildly different sizes)."""
    vec_target, ancora = normalizza_keypoints(coords_target, fattore_calibrazione_cranio)
    if ancora is None:
        raise ValueError(
            "The target has no valid scale anchor: cannot compute "
            "per-segment compatibility."
        )
    prop_target = lunghezze_segmenti_normalizzate(vec_target)

    risultati = {}
    for parte in SEGMENTI:
        prop_t = prop_target.get(parte)
        if prop_t is None:
            risultati[parte] = {"presente_nel_target": False, "donatori": []}
            continue

        coppie = []
        for nome_animale, dati in db.items():
            prop_a = lunghezze_segmenti_normalizzate(dati["coordinate_normalizzate"]).get(parte)
            if prop_a is None:
                continue
            coppie.append((nome_animale, abs(prop_t - prop_a)))

        ranking = _softmax_percentuali(coppie, temperature)
        risultati[parte] = {
            "presente_nel_target": True,
            "donatori": [
                {"animale": nome, "compatibilita_pct": round(pct, 2), "diff_proporzione": round(d, 4)}
                for nome, pct, d in ranking
            ],
        }
    return risultati


def genera_report_compatibilita(coords_target, db, fattore_calibrazione_cranio=None, top_k_specie=8):
    """Combines both similarity levels into a single structure, ready for
    report.py (display) and render/generate.py (IP-Adapter reference weights)."""
    ranking_specie = compatibilita_specie(coords_target, db, fattore_calibrazione_cranio)
    per_segmento = compatibilita_per_segmento(coords_target, db, fattore_calibrazione_cranio)
    return {
        "ranking_specie": ranking_specie[:top_k_specie],
        "n_animali_confrontati": len(ranking_specie),
        "per_segmento": per_segmento,
    }


def stampa_report(report):
    from display_names import species_name, segment_name

    print("\n--- OVERALL COMPATIBILITY (whole shape) ---")
    for r in report["ranking_specie"]:
        print(f"  {species_name(r['animale']):<25} {r['compatibilita_pct']:5.1f}%  "
              f"(shared points: {r['punti_comuni']})")

    print("\n--- BEST DONOR PER ANATOMICAL SEGMENT ---")
    for parte, dati in report["per_segmento"].items():
        etichetta = segment_name(parte).upper()
        if not dati["presente_nel_target"]:
            print(f"  {etichetta}: absent in target.")
            continue
        if not dati["donatori"]:
            print(f"  {etichetta}: no compatible donor found.")
            continue
        top = dati["donatori"][0]
        print(f"  {etichetta}: {species_name(top['animale'])} ({top['compatibilita_pct']:.1f}%)")


if __name__ == "__main__":
    from inference import estrai_coordinate

    _base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(_base_dir, "tests", "t-rex.jpg")
    pesi = os.path.join(_base_dir, "training_outputs_2", "Weights", "best_keypoint_detector.pth")
    db_path = os.path.join(_base_dir, "data", "processed", "geometric_database.json")

    print("Extracting target coordinates...")
    coords_target = estrai_coordinate(target_path, pesi, threshold=0.20)

    db, fattore_cranio = carica_database(db_path)
    report = genera_report_compatibilita(coords_target, db, fattore_cranio)
    stampa_report(report)
