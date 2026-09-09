"""
display_names.py

The dataset's file/species keys are Italian words (aquila.png, cane.png,
...) and the anatomical segment keys are Italian too (testa, torso,
arto_anteriore, ...). Those keys are left untouched everywhere they are
used as identifiers (JSON fields, dict keys, file paths) - renaming them
would mean renaming the whole dataset. This module only maps them to an
English label for anything a person actually reads: the console output,
the HTML/JSON report, and the text prompt sent to the image model.
"""
import os

SPECIES_EN = {
    "airone": "heron", "alcaimpenne": "great auk", "aquila": "eagle",
    "armadillo": "armadillo", "axolotl": "axolotl", "balaenicepsrex": "shoebill",
    "balena": "whale", "bassotto": "dachshund", "bradipo": "sloth",
    "camaleonte": "chameleon", "cane": "dog", "canguro": "kangaroo",
    "casuario": "cassowary", "cavallo": "horse", "celacanto": "coelacanth",
    "cervo": "deer", "cinghiale": "wild boar", "coccodrillo": "crocodile",
    "colibri": "hummingbird", "delfino": "dolphin", "dodo": "dodo",
    "dragodicommodo": "komodo dragon", "echidna": "echidna", "elefante": "elephant",
    "fagiano": "pheasant", "foca": "seal", "formichiere": "anteater",
    "gallina": "chicken", "gatto": "cat", "giraffa": "giraffe",
    "gorilla": "gorilla", "gufo": "owl", "iguana": "iguana",
    "ippopotamo": "hippopotamus", "kiwi": "kiwi", "levriero": "greyhound",
    "lupo": "wolf", "mammut": "mammoth", "mucca": "cow",
    "opossum": "opossum", "ornitorinco": "platypus", "pappagallo": "parrot",
    "pellicano": "pelican", "picchio": "woodpecker", "piccione": "pigeon",
    "pinguino": "penguin", "pipistrello": "bat", "rana": "frog",
    "razza": "stingray", "rinoceronte": "rhinoceros",
    "rinocerontelanoso": "woolly rhinoceros", "rondine": "swallow",
    "salamandra": "salamander", "scimpanze": "chimpanzee", "serpente": "snake",
    "squalo": "shark", "struzzo": "ostrich", "tacchino": "turkey",
    "tartaruga": "turtle", "thylacine": "thylacine", "topo": "mouse",
    "tricheco": "walrus", "trota": "trout", "tuatara": "tuatara",
    "uomo": "human",
}

SEGMENT_EN = {
    "testa": "head",
    "torso": "torso",
    "arto_anteriore": "front limb",
    "arto_posteriore": "hind limb",
    "coda": "tail",
}


def species_name(key):
    """key: a species dict key or filename, e.g. "aquila.png" or "aquila"."""
    stem = os.path.splitext(key)[0]
    return SPECIES_EN.get(stem, stem)


def segment_name(key):
    return SEGMENT_EN.get(key, key.replace("_", " "))
