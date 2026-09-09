"""
pipeline.py

CLI unico end-to-end: da un fossile/scheletro a report di compatibilita' +
render generativo.

    python pipeline.py --fossile input/mio_fossile.jpg
    python pipeline.py --fossile input/mio_fossile.jpg --no-render   # solo report, niente GPU

Passi:
1. preprocess_target: letterbox (+ rimozione sfondo/canonicalizzazione opzionali)
2. inference: estrazione dei 14 keypoint anatomici
3. compatibility: report di compatibilita' (specie intera + per segmento)
4. report: salva JSON + HTML leggibile
5. render (opzionale, richiede GPU + pesi diffusers scaricati): immagine finale
"""
import argparse
import os

from preprocess_target import prepara_immagine_target
from inference import estrai_coordinate
from compatibility import carica_database, genera_report_compatibilita, stampa_report
from report import genera_report_json, genera_report_html

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PESI_DEFAULT = os.path.join(_BASE_DIR, "training_outputs_2", "Weights", "best_keypoint_detector.pth")
_DB_DEFAULT = os.path.join(_BASE_DIR, "data", "processed", "geometric_database.json")
_OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs")


def esegui_pipeline(fossile_path, pesi_path=_PESI_DEFAULT, db_path=_DB_DEFAULT,
                     rimuovi_sfondo=False, canonicalizza=False, salta_preprocess=False,
                     genera_render=True, device="cuda", seed=None, output_dir=_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(fossile_path))[0]

    # 1. Preprocess
    if salta_preprocess:
        target_path = fossile_path
    else:
        target_path = os.path.join(output_dir, f"{nome_base}_preprocessato.png")
        prepara_immagine_target(fossile_path, target_path,
                                 rimuovi_sfondo_flag=rimuovi_sfondo,
                                 canonicalizza_flag=canonicalizza)

    # 2. Keypoint detection
    print("Estrazione keypoint anatomici...")
    coords = estrai_coordinate(target_path, pesi_path)

    n_trovati = sum(1 for v in coords.values() if v is not None)
    print(f"  {n_trovati}/14 keypoint rilevati con confidenza sufficiente.")

    # 3. Compatibility
    db, fattore_cranio = carica_database(db_path)
    report = genera_report_compatibilita(coords, db, fattore_cranio)
    stampa_report(report)

    # 4. Report leggibile
    json_path = os.path.join(output_dir, f"{nome_base}_report.json")
    html_path = os.path.join(output_dir, f"{nome_base}_report.html")
    genera_report_json(report, json_path)
    genera_report_html(report, db, target_path, html_path,
                        titolo=f"Report di compatibilita' - {nome_base}")
    print(f"\nReport salvato in:\n  {json_path}\n  {html_path}")

    render_path = None
    if genera_render:
        from render.generate import genera_render as _genera_render, salva_render
        print("\nGenerazione del render finale (richiede GPU, puo' richiedere qualche minuto "
              "e scaricare alcuni GB di pesi al primo utilizzo)...")
        immagine, control_image = _genera_render(coords, report, db, device=device, seed=seed)
        render_path = salva_render(immagine, os.path.join(output_dir, f"{nome_base}_render.png"))
        salva_render(control_image, os.path.join(output_dir, f"{nome_base}_control_map.png"))

    return {
        "target_preprocessato": target_path,
        "report_json": json_path,
        "report_html": html_path,
        "render": render_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OsteoGen 2.0: da un fossile a report + render.")
    parser.add_argument("--fossile", required=True, help="Percorso dell'immagine del fossile/scheletro.")
    parser.add_argument("--pesi", default=_PESI_DEFAULT)
    parser.add_argument("--db", default=_DB_DEFAULT)
    parser.add_argument("--rimuovi-sfondo", action="store_true")
    parser.add_argument("--canonicalizza", action="store_true",
                         help="Applica il blend con i bordi (Canny) per avvicinare foto reali allo stile del training set.")
    parser.add_argument("--salta-preprocess", action="store_true",
                         help="Usa l'immagine cosi' com'e' (deve essere gia' 512x512 su sfondo nero).")
    parser.add_argument("--no-render", action="store_true", help="Genera solo il report, salta lo stadio diffusers (niente GPU necessaria).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    esegui_pipeline(
        fossile_path=args.fossile,
        pesi_path=args.pesi,
        db_path=args.db,
        rimuovi_sfondo=args.rimuovi_sfondo,
        canonicalizza=args.canonicalizza,
        salta_preprocess=args.salta_preprocess,
        genera_render=not args.no_render,
        device=args.device,
        seed=args.seed,
    )
