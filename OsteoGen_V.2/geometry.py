"""
geometry.py

Geometric utilities shared by build_DB.py, compatibility.py and render/.
Defines the 14-anatomical-keypoint schema, the 5-body-part segmentation,
and the normalization (translation + scale) that makes keypoints from two
animals of different size/framing comparable to each other.

The schema is null-tolerant: a missing point (absent limb, no tail, etc.)
is represented as None everywhere in the pipeline.
"""
import numpy as np

NOMI_PUNTI = [
    "punta_muso", "retro_cranio", "base_collo", "spalla",
    "gomito_ant_1", "gomito_ant_2", "zampa_ant", "dorso", "anca",
    "ginocchio_post_1", "ginocchio_post_2", "zampa_post", "base_coda", "punta_coda"
]

# Anatomical segmentation: part name -> ordered list of the keypoints that
# make it up. The order is also used by render/control_map.py to draw the
# bone segments as a polyline.
SEGMENTI = {
    "testa": ["punta_muso", "retro_cranio", "base_collo"],
    "torso": ["base_collo", "spalla", "dorso", "anca", "base_coda"],
    "arto_anteriore": ["spalla", "gomito_ant_1", "gomito_ant_2", "zampa_ant"],
    "arto_posteriore": ["anca", "ginocchio_post_1", "ginocchio_post_2", "zampa_post"],
    "coda": ["base_coda", "punta_coda"],
}

# Preferred scale unit: torso length (base_collo->base_coda). When missing
# (anatomies with no annotated tail: frog, snake, turtle, chimpanzee,
# human, bat...) an "equivalent torso" is estimated from skull length
# (punta_muso->retro_cranio), corrected by the average skull/torso ratio
# observed over the rest of the dataset (see
# stima_fattore_calibrazione_cranio) - without this correction, a "1
# skull" unit and a "1 torso" unit aren't comparable and distances between
# the two populations explode artificially.
RAPPORTO_CRANIO_TORSO_DEFAULT = 0.35  # used only if the DB doesn't provide a calibrated estimate


def dist(p1, p2):
    """Euclidean distance between two (x, y) points; None if either is missing."""
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def stima_fattore_calibrazione_cranio(elenco_coords):
    """Average skull_length/torso_length ratio computed over the dataset
    animals that have BOTH measurements (the majority). Used to convert the
    'skull' scale-anchor into an equivalent torso for animals that lack a
    torso measurement."""
    rapporti = []
    for coords in elenco_coords:
        torso = dist(coords.get("base_collo"), coords.get("base_coda"))
        cranio = dist(coords.get("punta_muso"), coords.get("retro_cranio"))
        if torso and torso > 1e-6 and cranio and cranio > 1e-6:
            rapporti.append(cranio / torso)
    return float(np.mean(rapporti)) if rapporti else RAPPORTO_CRANIO_TORSO_DEFAULT


def trova_ancora_scala(coords, fattore_calibrazione_cranio=None):
    """Determines the origin and scale for normalizza_keypoints.

    - Prefers the torso (base_collo->base_coda): exact scale, unit 'torso'.
    - Falls back to the skull (punta_muso->retro_cranio), rescaled to an
      equivalent torso via fattore_calibrazione_cranio, so the unit stays
      consistent with torso-anchored animals.
    - The origin is always base_collo when available (even if the scale
      comes from the skull), otherwise punta_muso.

    Returns (origin, scale, anchor_name), or (None, None, None) if no
    usable measurement exists at all.
    """
    fattore = fattore_calibrazione_cranio or RAPPORTO_CRANIO_TORSO_DEFAULT
    origine_collo = coords.get("base_collo")

    torso = dist(coords.get("base_collo"), coords.get("base_coda"))
    if torso is not None and torso > 1e-6:
        return origine_collo, torso, "torso"

    cranio = dist(coords.get("punta_muso"), coords.get("retro_cranio"))
    if cranio is not None and cranio > 1e-6:
        scala_equivalente = cranio / fattore
        origine = origine_collo if origine_collo is not None else coords.get("punta_muso")
        return origine, scala_equivalente, "cranio_calibrato"

    return None, None, None


def normalizza_keypoints(coords, fattore_calibrazione_cranio=None):
    """
    Makes the keypoints invariant to position and scale:
    - translates the origin to base_collo (or punta_muso if the neck is missing)
    - scales by torso length, or its skull-estimated equivalent when the
      torso isn't annotated

    Returns (dict {name: (x,y) or None}, anchor_name). If no scale
    measurement is available, returns all None (the animal isn't
    geometrically comparable and must be excluded from retrieval).
    """
    origine, scala, nome_ancora = trova_ancora_scala(coords, fattore_calibrazione_cranio)
    if origine is None:
        return {nome: None for nome in NOMI_PUNTI}, None

    ox, oy = origine
    out = {}
    for nome in NOMI_PUNTI:
        p = coords.get(nome)
        out[nome] = None if p is None else ((p[0] - ox) / scala, (p[1] - oy) / scala)
    return out, nome_ancora


def lunghezza_catena(coords, punti):
    """Length of a keypoint chain as the sum of distances between
    consecutive points (not the direct first->last distance): this follows
    the limb/torso's actual articulation instead of cutting a straight
    line. Links with a missing endpoint are skipped. Returns None if no
    link at all is measurable (e.g. a snake's torso, almost entirely None)."""
    lunghezza = 0.0
    almeno_un_link = False
    for p1, p2 in zip(punti, punti[1:]):
        d = dist(coords.get(p1), coords.get(p2))
        if d is not None:
            lunghezza += d
            almeno_un_link = True
    return lunghezza if almeno_un_link else None


def lunghezze_segmenti(coords):
    """'Base' length of each segment (sum of the bone chain's links), in
    absolute pixels. Used to build the DB. NOTE: for the 'torso' segment
    this value coincides, by construction, with the scale anchor when
    available (a direct base_collo->base_coda is a sub-case of the chain) -
    see lunghezze_segmenti_normalizzate for why the per-donor comparison
    still uses the whole chain and not just endpoint-to-endpoint."""
    out = {}
    for parte, punti in SEGMENTI.items():
        d = lunghezza_catena(coords, punti)
        out[parte] = d if d is not None else 0.0
    return out


def lunghezze_segmenti_normalizzate(coords_normalizzati):
    """Like lunghezze_segmenti, but on already-normalized keypoints: the
    result is therefore directly a proportion (length / anchor scale),
    invariant to the animal's absolute size.

    For the 'torso' segment the chain also passes through spalla/dorso/anca,
    not just the base_collo/base_coda endpoints: using only the direct
    distance between the endpoints would always give ~1.0 for any
    torso-anchored animal (that's literally the definition of the scale),
    a useless comparison. Summing the intermediate links instead reflects
    how 'arched'/elongated the back is, which varies animal to animal.

    None for segments where no link is measurable in the target.
    """
    out = {}
    for parte, punti in SEGMENTI.items():
        out[parte] = lunghezza_catena(coords_normalizzati, punti)
    return out
