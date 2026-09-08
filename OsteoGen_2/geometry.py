"""
geometry.py

Utilita' geometriche condivise tra build_DB.py, compatibility.py e render/.
Definisce lo schema dei 14 keypoint anatomici, la segmentazione in 5 parti
del corpo, e la normalizzazione (traslazione + scala) che rende i keypoint
di due animali di taglia/inquadratura diverse confrontabili tra loro.

Lo schema e' null-tolerant: un punto assente (arto mancante, coda assente,
ecc.) e' rappresentato come None ovunque nella pipeline.
"""
import numpy as np

NOMI_PUNTI = [
    "punta_muso", "retro_cranio", "base_collo", "spalla",
    "gomito_ant_1", "gomito_ant_2", "zampa_ant", "dorso", "anca",
    "ginocchio_post_1", "ginocchio_post_2", "zampa_post", "base_coda", "punta_coda"
]

# Segmentazione anatomica: nome_parte -> lista ordinata di keypoint che la
# compongono. L'ordine e' usato anche da render/control_map.py per tracciare
# i segmenti ossei come una polilinea.
SEGMENTI = {
    "testa": ["punta_muso", "retro_cranio", "base_collo"],
    "torso": ["base_collo", "spalla", "dorso", "anca", "base_coda"],
    "arto_anteriore": ["spalla", "gomito_ant_1", "gomito_ant_2", "zampa_ant"],
    "arto_posteriore": ["anca", "ginocchio_post_1", "ginocchio_post_2", "zampa_post"],
    "coda": ["base_coda", "punta_coda"],
}

# Unita' di scala preferita: la lunghezza del torso (base_collo->base_coda).
# Quando manca (anatomie senza coda annotata: rana, serpente, tartaruga,
# scimpanze, uomo, pipistrello...) si stima un "torso equivalente" dalla
# lunghezza del cranio (punta_muso->retro_cranio), corretta per il rapporto
# medio cranio/torso osservato sul resto del dataset (vedi
# stima_fattore_calibrazione_cranio) — senza questa correzione, un'unita'
# "1 cranio" e un'unita' "1 torso" non sono confrontabili tra loro e le
# distanze tra le due popolazioni esplodono artificialmente.
RAPPORTO_CRANIO_TORSO_DEFAULT = 0.35  # usato solo se il DB non fornisce una stima calibrata


def dist(p1, p2):
    """Distanza euclidea tra due punti (x, y); None se uno dei due manca."""
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def stima_fattore_calibrazione_cranio(elenco_coords):
    """Rapporto medio lunghezza_cranio/lunghezza_torso calcolato sugli
    animali del dataset che hanno ENTRAMBE le misure (la maggioranza).
    Usato per convertire lo scale-anchor 'cranio' in un torso equivalente
    per gli animali che invece il torso non ce l'hanno."""
    rapporti = []
    for coords in elenco_coords:
        torso = dist(coords.get("base_collo"), coords.get("base_coda"))
        cranio = dist(coords.get("punta_muso"), coords.get("retro_cranio"))
        if torso and torso > 1e-6 and cranio and cranio > 1e-6:
            rapporti.append(cranio / torso)
    return float(np.mean(rapporti)) if rapporti else RAPPORTO_CRANIO_TORSO_DEFAULT


def trova_ancora_scala(coords, fattore_calibrazione_cranio=None):
    """Determina origine e scala per normalizza_keypoints.

    - Preferisce il torso (base_collo->base_coda): scala esatta, unita' 'torso'.
    - In fallback usa il cranio (punta_muso->retro_cranio), riscalato al
      torso equivalente tramite fattore_calibrazione_cranio, cosi' l'unita'
      resta coerente con quella degli animali torso-anchored.
    - L'origine e' sempre base_collo quando disponibile (anche se la scala
      viene dal cranio), altrimenti punta_muso.

    Ritorna (origine, scala, nome_ancora) oppure (None, None, None) se non
    c'e' proprio nessuna misura utilizzabile.
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
    Rende i keypoint invarianti a posizione e scala:
    - trasla l'origine su base_collo (o punta_muso se il collo manca)
    - scala per la lunghezza del torso, o per il suo equivalente stimato
      dal cranio quando il torso non e' annotato

    Ritorna (dict {nome: (x,y) o None}, nome_ancora). Se non e' disponibile
    nessuna misura di scala, ritorna tutti None (l'animale non e' comparabile
    geometricamente e va escluso dal retrieval).
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
    """Lunghezza di una catena di keypoint come somma delle distanze tra
    punti consecutivi (non la distanza diretta primo->ultimo): segue quindi
    la reale articolazione dell'arto/tronco invece di tagliare in linea
    retta. I link con un estremo mancante vengono saltati. Ritorna None se
    non c'e' nessun link misurabile (es. torso di un serpente, quasi tutto
    None)."""
    lunghezza = 0.0
    almeno_un_link = False
    for p1, p2 in zip(punti, punti[1:]):
        d = dist(coords.get(p1), coords.get(p2))
        if d is not None:
            lunghezza += d
            almeno_un_link = True
    return lunghezza if almeno_un_link else None


def lunghezze_segmenti(coords):
    """Lunghezza 'base' di ciascun segmento (somma dei link della catena
    ossea), in pixel assoluti. Usata per costruire il DB. NOTA: per il
    segmento 'torso' questo valore coincide, per costruzione, con l'ancora
    di scala quando e' disponibile (base_collo->base_coda diretta e' un
    sotto-caso della catena) - vedi lunghezze_segmenti_normalizzate per il
    motivo per cui il confronto per-donatore usa comunque la catena intera
    e non il solo endpoint-to-endpoint."""
    out = {}
    for parte, punti in SEGMENTI.items():
        d = lunghezza_catena(coords, punti)
        out[parte] = d if d is not None else 0.0
    return out


def lunghezze_segmenti_normalizzate(coords_normalizzati):
    """Come lunghezze_segmenti, ma sui keypoint gia' normalizzati: il
    risultato e' quindi direttamente una proporzione (lunghezza / scala
    dell'ancora), invariante alla taglia assoluta dell'animale.

    Per il segmento 'torso' la catena passa anche per spalla/dorso/anca, non
    solo per gli estremi base_collo/base_coda: se usassimo la sola distanza
    diretta tra gli estremi otterremmo sempre ~1.0 per qualunque animale
    ancorato sul torso (e' letteralmente la definizione della scala), un
    confronto inutile. Sommando i link intermedi il valore riflette invece
    quanto e' 'arcuato'/allungato il dorso, che varia da animale ad animale.

    None per i segmenti dove non c'e' nessun link misurabile nel target.
    """
    out = {}
    for parte, punti in SEGMENTI.items():
        out[parte] = lunghezza_catena(coords_normalizzati, punti)
    return out
