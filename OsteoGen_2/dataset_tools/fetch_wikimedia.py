"""
dataset_tools/fetch_wikimedia.py

Aiuto semi-automatico per ampliare input_x/target_y da Wikimedia Commons
(fonte a licenza aperta verificabile per-immagine). Deliberatamente NON
scarica nulla in automatico durante la ricerca: elenca i candidati con
licenza e autore, la scelta di quale scaricare resta manuale.

Uso:
    python dataset_tools/fetch_wikimedia.py cerca "elephant skeleton"
    python dataset_tools/fetch_wikimedia.py scarica \\
        --url "https://upload.wikimedia.org/.../Elephant_skeleton.jpg" \\
        --dest ../data/processed/input_x/elefante.png --tipo scheletro \\
        --autore "Nome Autore" --licenza "CC BY-SA 4.0" \\
        --fonte-pagina "https://commons.wikimedia.org/wiki/File:..."
"""
import argparse
import csv
import datetime
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_target import prepara_immagine_target

API_URL = "https://commons.wikimedia.org/w/api.php"
CSV_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attributions.csv")


def cerca(query, limit=10):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": limit, "gsrnamespace": 6,
        "prop": "imageinfo", "iiprop": "url|extmetadata|size",
    }
    r = requests.get(API_URL, params=params, timeout=20, headers={"User-Agent": "OsteoGen2.0-dataset-tools/1.0"})
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})

    risultati = []
    for p in pages.values():
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        risultati.append({
            "titolo": p.get("title", ""),
            "url_immagine": info.get("url", ""),
            "pagina_descrizione": info.get("descriptionurl", ""),
            "licenza": meta.get("LicenseShortName", {}).get("value", "sconosciuta"),
            "autore": meta.get("Artist", {}).get("value", "sconosciuto"),
            "dimensioni": f"{info.get('width', '?')}x{info.get('height', '?')}",
        })
    return risultati


def scarica_e_registra(url, dest_path, tipo, fonte_pagina, autore, licenza,
                        csv_path=CSV_DEFAULT, lato_target=512):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    r = requests.get(url, timeout=30, headers={"User-Agent": "OsteoGen2.0-dataset-tools/1.0"})
    r.raise_for_status()

    tmp_path = dest_path + ".raw_download"
    with open(tmp_path, 'wb') as f:
        f.write(r.content)

    prepara_immagine_target(tmp_path, dest_path, target_size=lato_target)
    os.remove(tmp_path)

    scrivi_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if scrivi_header:
            writer.writerow(["file", "tipo", "fonte_url", "autore", "licenza", "data_aggiunta", "note"])
        writer.writerow([os.path.basename(dest_path), tipo, fonte_pagina, autore, licenza,
                          datetime.date.today().isoformat(), ""])

    print(f"Salvato: {dest_path}\nRegistrato in: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="comando", required=True)

    p_cerca = sub.add_parser("cerca", help="Elenca candidati su Wikimedia Commons (non scarica nulla).")
    p_cerca.add_argument("query")
    p_cerca.add_argument("--limit", type=int, default=10)

    p_scarica = sub.add_parser("scarica", help="Scarica un'immagine gia' scelta e la registra in attributions.csv.")
    p_scarica.add_argument("--url", required=True, help="URL diretto del file (campo url_immagine di 'cerca').")
    p_scarica.add_argument("--dest", required=True, help="Percorso di destinazione, es. ../data/processed/input_x/elefante.png")
    p_scarica.add_argument("--tipo", choices=["scheletro", "animale_vivo"], required=True)
    p_scarica.add_argument("--autore", default="sconosciuto")
    p_scarica.add_argument("--licenza", default="sconosciuta")
    p_scarica.add_argument("--fonte-pagina", default="")

    args = parser.parse_args()

    if args.comando == "cerca":
        for r in cerca(args.query, args.limit):
            print(f"- {r['titolo']} [{r['licenza']}] {r['dimensioni']}")
            print(f"    autore: {r['autore']}")
            print(f"    immagine: {r['url_immagine']}")
            print(f"    pagina:   {r['pagina_descrizione']}\n")
    elif args.comando == "scarica":
        scarica_e_registra(args.url, args.dest, args.tipo, args.fonte_pagina,
                            args.autore, args.licenza)
