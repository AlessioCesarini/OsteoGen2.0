"""
pipeline.py

Single end-to-end CLI: from a fossil/skeleton to a compatibility report +
generative render.

    python pipeline.py                                  # interactive: asks for the image
    python pipeline.py --fossile input/my_fossil.jpg
    python pipeline.py --fossile input/my_fossil.jpg --no-render   # report only, no GPU needed

Without --fossile the script asks interactively for the image path (meant
for a one-off run or a demo, not just for people used to a command line):
it shows a short summary of what the pipeline is about to do, a progress
indicator per stage, and opens the report and the produced images at the
end. Missing dependencies and the trained weights file are resolved
automatically wherever possible (see bootstrap.py / ui.resolve_weights_path).
With --fossile the behaviour stays scriptable/silent as before, for tests
and automation.

Steps:
1. preprocess_target: letterbox (+ optional background removal/canonicalization)
2. inference: extraction of the 14 anatomical keypoints
3. compatibility: compatibility report (whole species + per segment)
4. report: saves a JSON + a readable HTML report
5. render (optional, needs a GPU + downloaded diffusers weights): final image
"""
import os
import sys

# Windows: a duplicate OpenMP runtime (PyTorch's bundled copy clashing with
# another package's, e.g. numpy/matplotlib) makes Intel's runtime abort the
# whole process with "OMP: Error #15" - right after the render, before the
# final report/comparison get shown. This is the standard workaround; it
# must be set before torch/numpy/matplotlib are imported anywhere in the
# process, so it has to happen here, before any of pipeline.py's own imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if sys.platform == "win32":
    # Legacy Windows consoles (PowerShell, Anaconda Prompt) often use a
    # non-UTF-8 codepage, which garbles rich's Unicode box-drawing table
    # into mojibake. Force UTF-8 for both the console and Python's own
    # stdout/stderr.
    os.system("chcp 65001 >nul")
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from bootstrap import ensure_packages

# Needed for every run (report + retrieval), independent of --no-render.
# huggingface_hub is here (not in _RENDER_PACKAGES) because ui.resolve_weights_path
# may need it to auto-download the keypoint-detector weights before we even
# know whether the render stage will run.
_CORE_PACKAGES = [
    ("numpy", "numpy"), ("cv2", "opencv-python"), ("PIL", "Pillow"),
    ("scipy", "scipy"), ("torch", "torch"), ("torchvision", "torchvision"),
    ("matplotlib", "matplotlib"), ("rich", "rich"), ("huggingface_hub", "huggingface_hub"),
]
# Only needed if the generative render stage actually runs.
_RENDER_PACKAGES = [
    ("diffusers", "diffusers"), ("transformers", "transformers"),
    ("accelerate", "accelerate"), ("safetensors", "safetensors"),
    ("controlnet_aux", "controlnet-aux"),
]

ensure_packages(_CORE_PACKAGES, "the core pipeline (report + retrieval)")

import argparse

from preprocess_target import prepara_immagine_target
from inference import estrai_coordinate
from compatibility import carica_database, genera_report_compatibilita, stampa_report
from report import genera_report_json, genera_report_html
import ui

_PESI_DEFAULT = os.path.join(_BASE_DIR, "training_outputs_2", "Weights", "best_keypoint_detector.pth")
_DB_DEFAULT = os.path.join(_BASE_DIR, "data", "processed", "geometric_database.json")
_OUTPUT_DIR = os.path.join(_BASE_DIR, "outputs")


def _pick_device(richiesto):
    import torch
    if richiesto == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if richiesto == "cuda" and not torch.cuda.is_available():
        print("[setup] No CUDA GPU available: falling back to CPU (the render stage will be slower).")
        return "cpu"
    return richiesto


def esegui_pipeline(fossile_path, pesi_path=_PESI_DEFAULT, db_path=_DB_DEFAULT,
                     rimuovi_sfondo=False, canonicalizza=False, salta_preprocess=False,
                     genera_render=True, device="auto", seed=None, output_dir=_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(fossile_path))[0]
    pesi_path = ui.resolve_weights_path(pesi_path)
    device = _pick_device(device)

    # 1. Preprocess
    if salta_preprocess:
        target_path = fossile_path
    else:
        target_path = os.path.join(output_dir, f"{nome_base}_preprocessed.png")
        with ui.stage("Preprocessing the image (letterbox onto a 512x512 black background)"):
            prepara_immagine_target(fossile_path, target_path,
                                     rimuovi_sfondo_flag=rimuovi_sfondo,
                                     canonicalizza_flag=canonicalizza)

    # 2. Keypoint detection
    with ui.stage("Detecting the 14 anatomical keypoints (ResNet18)"):
        coords = estrai_coordinate(target_path, pesi_path)
    n_trovati = sum(1 for v in coords.values() if v is not None)
    print(f"  {n_trovati}/14 keypoints detected with sufficient confidence.")

    # 3. Compatibility
    with ui.stage("Retrieval: computing compatibility against the geometric database"):
        db, fattore_cranio = carica_database(db_path)
        report = genera_report_compatibilita(coords, db, fattore_cranio)
    stampa_report(report)

    # 4. Readable report
    json_path = os.path.join(output_dir, f"{nome_base}_report.json")
    html_path = os.path.join(output_dir, f"{nome_base}_report.html")
    with ui.stage("Generating the readable report (JSON + HTML)"):
        genera_report_json(report, json_path)
        genera_report_html(report, db, target_path, html_path,
                            titolo=f"Compatibility report - {nome_base}")
    print(f"  Report saved to:\n    {json_path}\n    {html_path}")

    render_path = None
    control_map_path = None
    if genera_render:
        ensure_packages(_RENDER_PACKAGES, "the generative render stage")
        from render.generate import genera_render as _genera_render, salva_render
        with ui.stage("Generative render (ControlNet + IP-Adapter; on a GPU this takes a "
                      "couple of minutes, on CPU much longer; downloads a few GB of "
                      "weights on first use)"):
            immagine, control_image = _genera_render(coords, report, db, device=device, seed=seed)
        render_path = salva_render(immagine, os.path.join(output_dir, f"{nome_base}_render.png"))
        control_map_path = salva_render(control_image, os.path.join(output_dir, f"{nome_base}_control_map.png"))

    return {
        "target_preprocessato": target_path,
        "report_json": json_path,
        "report_html": html_path,
        "render": render_path,
        "control_map": control_map_path,
        "report": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OsteoGen 2.0: from a fossil to a compatibility report + render.")
    parser.add_argument("--fossile", default=None,
                         help="Path to the fossil/skeleton image. If omitted, it is asked interactively.")
    parser.add_argument("--pesi", default=_PESI_DEFAULT,
                         help="Path to the trained keypoint-detector weights (.pth). "
                              "If missing, it is resolved automatically or asked interactively.")
    parser.add_argument("--db", default=_DB_DEFAULT)
    parser.add_argument("--rimuovi-sfondo", action="store_true", help="Remove the background before preprocessing.")
    parser.add_argument("--canonicalizza", action="store_true",
                         help="Blend in Canny edges to bring real photos closer to the training set's style.")
    parser.add_argument("--salta-preprocess", action="store_true",
                         help="Use the image as-is (must already be 512x512 on a black background).")
    parser.add_argument("--no-render", action="store_true", help="Report only, skip the diffusers stage (no GPU needed).")
    parser.add_argument("--device", default="auto", help="'auto' (default), 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Interactive mode: no --fossile given on the command line. Ask for the
    # path and, since the generative render is the heaviest stage (GPU +
    # first-time weight downloads), ask for explicit confirmation instead
    # of launching it as a surprise.
    interattivo = args.fossile is None
    if interattivo:
        ui.show_banner()
        fossile_path = ui.ask_image_path()
        genera_render = not args.no_render and ui.ask_yes_no(
            "\nAlso generate the final render (ControlNet + IP-Adapter; needs a GPU "
            "for a reasonable runtime and may download a few GB of weights on first use)?",
            default=True)
    else:
        fossile_path = args.fossile
        genera_render = not args.no_render

    risultati = esegui_pipeline(
        fossile_path=fossile_path,
        pesi_path=args.pesi,
        db_path=args.db,
        rimuovi_sfondo=args.rimuovi_sfondo,
        canonicalizza=args.canonicalizza,
        salta_preprocess=args.salta_preprocess,
        genera_render=genera_render,
        device=args.device,
        seed=args.seed,
    )

    if interattivo:
        ui.print_summary(risultati["report"])
        ui.open_report_in_browser(risultati["report_html"])
        ui.show_final_comparison([
            ("Preprocessed target", risultati["target_preprocessato"]),
            ("Final render", risultati["render"]),
            ("Control map", risultati["control_map"]),
        ], output_dir=_OUTPUT_DIR)
