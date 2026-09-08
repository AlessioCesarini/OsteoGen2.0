"""
compatibility.py

Sostituisce match_dinosauro.py: da un semplice "vince il piu' vicino" a un
vero report di compatibilita' percentuale.

Due livelli di similarita', entrambi calcolati sui keypoint normalizzati
(vedi geometry.normalizza_keypoints, invarianti a scala/posizione):

1. compatibilita_specie: quanto l'intera sagoma del target assomiglia a
   quella di ciascun animale del DB (per il report "il tuo fossile e' per
   il 50% aquila, 30% pollo, ...").
2. compatibilita_per_segmento: per ciascuna delle 5 parti del corpo, quali
   animali sono i donatori di texture piu' plausibili (serve al renderer).

Le distanze sono convertite in percentuali con una softmax: piu' un animale
e' vicino geometricamente al target, piu' peso riceve, ma tutti gli animali
compatibili concorrono al punteggio (non solo il primo classificato). Le
temperature di default sono calibrate empiricamente sul dataset delle 65
specie (vedi note nel repo/plan): con target=un animale del DB stesso, il
suo self-match resta intorno al 50-100% e i vicini piu' plausibili si
piazzano subito dopo, invece di un ranking quasi uniforme.
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
    """Ritorna (animali, fattore_calibrazione_cranio). `animali` e' il dict
    nome_file -> dati geometrici; il fattore serve a normalizzare il target
    in modo coerente con il DB quando gli manca il torso (vedi geometry.py)."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    meta = raw.get("_meta", {})
    animali = raw.get("animali", raw)  # retrocompatibilita' con vecchi DB piatti
    return animali, meta.get("fattore_calibrazione_cranio")


def _distanza_normalizzata(vec_a, vec_b, min_punti_comuni):
    """RMS della distanza tra i keypoint normalizzati condivisi da due
    entita'. Ritorna (distanza, n_punti_comuni); distanza=None se i punti
    in comune sono troppo pochi per un confronto affidabile."""
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
    """coppie_nome_distanza: lista di (nome, distanza). Ritorna lista di
    (nome, pct, distanza) ordinata per pct decrescente, pct che sommano
    a 100."""
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
    """Ranking di compatibilita' dell'intero target contro ogni animale del
    DB. temperature piu' bassa = distribuzione piu' 'piccata' su pochi
    animali; piu' alta = punteggi piu' spalmati."""
    vec_target, ancora = normalizza_keypoints(coords_target, fattore_calibrazione_cranio)
    if ancora is None:
        raise ValueError(
            "Il target non ha un'ancora di scala valida (mancano sia il "
            "torso base_collo->base_coda sia il cranio punta_muso->retro_cranio): "
            "impossibile calcolare la compatibilita'."
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
    """Per ciascun segmento anatomico presente nel target, ranking % dei
    donatori piu' compatibili in termini di proporzione rispetto al
    torso/cranio (non di forma assoluta: un piccione e uno struzzo possono
    avere ali proporzionalmente simili pur essendo di taglia diversissima)."""
    vec_target, ancora = normalizza_keypoints(coords_target, fattore_calibrazione_cranio)
    if ancora is None:
        raise ValueError(
            "Il target non ha un'ancora di scala valida: impossibile calcolare "
            "la compatibilita' per segmento."
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
    """Combina i due livelli di similarita' in un'unica struttura, pronta
    per report.py (visualizzazione) e render/generate.py (pesi per le
    IP-Adapter reference)."""
    ranking_specie = compatibilita_specie(coords_target, db, fattore_calibrazione_cranio)
    per_segmento = compatibilita_per_segmento(coords_target, db, fattore_calibrazione_cranio)
    return {
        "ranking_specie": ranking_specie[:top_k_specie],
        "n_animali_confrontati": len(ranking_specie),
        "per_segmento": per_segmento,
    }


def stampa_report(report):
    print("\n--- COMPATIBILITA' COMPLESSIVA (sagoma intera) ---")
    for r in report["ranking_specie"]:
        print(f"  {r['animale']:<25} {r['compatibilita_pct']:5.1f}%  "
              f"(punti condivisi: {r['punti_comuni']})")

    print("\n--- MIGLIOR DONATORE PER SEGMENTO ANATOMICO ---")
    for parte, dati in report["per_segmento"].items():
        if not dati["presente_nel_target"]:
            print(f"  {parte.upper()}: assente nel target.")
            continue
        if not dati["donatori"]:
            print(f"  {parte.upper()}: nessun donatore compatibile trovato.")
            continue
        top = dati["donatori"][0]
        print(f"  {parte.upper()}: {top['animale']} ({top['compatibilita_pct']:.1f}%)")


if __name__ == "__main__":
    from inference import estrai_coordinate

    _base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(_base_dir, "tests", "t-rex.jpg")
    pesi = os.path.join(_base_dir, "training_outputs_2", "Weights", "best_keypoint_detector.pth")
    db_path = os.path.join(_base_dir, "data", "processed", "geometric_database.json")

    print("Estrazione coordinate del target in corso...")
    coords_target = estrai_coordinate(target_path, pesi, threshold=0.20)

    db, fattore_cranio = carica_database(db_path)
    report = genera_report_compatibilita(coords_target, db, fattore_cranio)
    stampa_report(report)
