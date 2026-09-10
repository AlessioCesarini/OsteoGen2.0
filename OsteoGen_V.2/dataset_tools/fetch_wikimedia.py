"""
dataset_tools/fetch_wikimedia.py

Semi-automatic helper to expand input_x/target_y from Wikimedia Commons
(a source with per-image verifiable open licensing). Deliberately does NOT
download anything automatically while searching: it lists candidates with
their license and author, and which one to actually download stays a
manual decision.

Usage:
    python dataset_tools/fetch_wikimedia.py search "elephant skeleton"
    python dataset_tools/fetch_wikimedia.py download \\
        --url "https://upload.wikimedia.org/.../Elephant_skeleton.jpg" \\
        --dest ../data/processed/input_x/elefante.png --type skeleton \\
        --author "Author Name" --license "CC BY-SA 4.0" \\
        --source-page "https://commons.wikimedia.org/wiki/File:..."
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
            "title": p.get("title", ""),
            "image_url": info.get("url", ""),
            "description_page": info.get("descriptionurl", ""),
            "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
            "author": meta.get("Artist", {}).get("value", "unknown"),
            "size": f"{info.get('width', '?')}x{info.get('height', '?')}",
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
            writer.writerow(["file", "type", "source_url", "author", "license", "date_added", "notes"])
        writer.writerow([os.path.basename(dest_path), tipo, fonte_pagina, autore, licenza,
                          datetime.date.today().isoformat(), ""])

    print(f"Saved: {dest_path}\nLogged in: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="comando", required=True)

    p_cerca = sub.add_parser("search", help="List candidates on Wikimedia Commons (downloads nothing).")
    p_cerca.add_argument("query")
    p_cerca.add_argument("--limit", type=int, default=10)

    p_scarica = sub.add_parser("download", help="Download a chosen image and log it in attributions.csv.")
    p_scarica.add_argument("--url", required=True, help="Direct file URL (the 'image_url' field from 'search').")
    p_scarica.add_argument("--dest", required=True, help="Destination path, e.g. ../data/processed/input_x/elefante.png")
    p_scarica.add_argument("--type", dest="tipo", choices=["skeleton", "living_animal"], required=True)
    p_scarica.add_argument("--author", dest="autore", default="unknown")
    p_scarica.add_argument("--license", dest="licenza", default="unknown")
    p_scarica.add_argument("--source-page", dest="fonte_pagina", default="")

    args = parser.parse_args()

    if args.comando == "search":
        for r in cerca(args.query, args.limit):
            print(f"- {r['title']} [{r['license']}] {r['size']}")
            print(f"    author: {r['author']}")
            print(f"    image:  {r['image_url']}")
            print(f"    page:   {r['description_page']}\n")
    elif args.comando == "download":
        scarica_e_registra(args.url, args.dest, args.tipo, args.fonte_pagina,
                            args.autore, args.licenza)
