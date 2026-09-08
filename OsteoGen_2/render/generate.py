"""
render/generate.py

Lo stadio generativo vero e proprio: sostituisce il collage a warping
rigido (render/control_map.py, che ora produce solo materiale di supporto)
con un rendering via diffusione locale, condizionato da:

- ControlNet sulla control map dello scheletro target (disegna_control_map):
  vincola posa e proporzioni al fossile reale.
- IP-Adapter sulle foto dei migliori donatori (report["ranking_specie"]):
  da' al modello un riferimento visivo di texture/colore, pesato per
  compatibilita' - la stessa idea dell'esperimento manuale con ChatGPT
  (es. 50% aquila + 30% pollo + 20% altro), ma con pesi aperti eseguiti
  in locale, niente API a pagamento.
- Un prompt testuale costruito automaticamente dal report, come guida
  aggiuntiva (debole rispetto a ControlNet+IP-Adapter, ma aiuta lo stile).

ATTENZIONE: questo modulo richiede una GPU con CUDA (o comunque un
acceleratore supportato da torch) e i pesi vengono scaricati da Hugging Face
al primo utilizzo (qualche GB). Non e' stato eseguito end-to-end in fase di
sviluppo di questo modulo: la macchina usata per scrivere il codice non ha
una GPU NVIDIA. Vanno quindi previsti aggiustamenti minori (nomi di
argomenti, versioni di `diffusers`) al primo run sulla RTX 5080.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render.control_map import disegna_control_map

MODEL_BASE = "runwayml/stable-diffusion-v1-5"
MODEL_CONTROLNET = "lllyasviel/sd-controlnet-scribble"
MODEL_IP_ADAPTER_REPO = "h94/IP-Adapter"
MODEL_IP_ADAPTER_SUBFOLDER = "models"
MODEL_IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"

MAX_DONATORI_IP_ADAPTER = 4       # quanti animali del ranking_specie usare come riferimento visivo
MAX_IMMAGINI_MEDIA_PESATA = 10    # quante copie totali nella lista "pesata per ripetizione" (vedi sotto)


def costruisci_prompt(report, max_specie=3):
    """Prompt testuale di supporto: da solo non basta a garantire coerenza
    (per quello ci sono ControlNet e IP-Adapter), ma aiuta lo stile
    complessivo e da' un fallback leggibile se IP-Adapter non e' disponibile."""
    top = report["ranking_specie"][:max_specie]
    mix = ", ".join(f"{r['compatibilita_pct']:.0f}% {os.path.splitext(r['animale'])[0]}" for r in top)
    prompt = (
        f"a photorealistic reconstruction of a prehistoric animal, anatomical and textural "
        f"blend of {mix}, detailed skin/feather/scale texture, natural daylight, "
        f"museum-quality paleoart reconstruction, full body, side profile, highly detailed"
    )
    negative = (
        "blurry, cartoon, illustration, low quality, deformed, extra limbs, missing limbs, "
        "text, watermark, logo, multiple animals, collage, human"
    )
    return prompt, negative


def _lista_donatori_pesata(report, db, max_donatori=MAX_DONATORI_IP_ADAPTER,
                            max_immagini=MAX_IMMAGINI_MEDIA_PESATA):
    """Costruisce la lista di immagini da passare a IP-Adapter ripetendo
    ogni donatore in proporzione al suo peso di compatibilita'.

    E' un modo robusto (non dipende da API interne di diffusers che
    cambiano tra versioni) di approssimare una media pesata degli embedding:
    IP-Adapter fa gia' la media delle embedding quando riceve una lista di
    immagini, quindi ripetere un'immagine 5 volte su 10 invece di 1 volta
    su 10 la pesa circa 5x nella media finale. Per un controllo esatto
    (media pesata analitica sugli embedding) vedi la nota in fondo al file."""
    top = [r for r in report["ranking_specie"][:max_donatori] if r["animale"] in db]
    if not top:
        return []

    pesi = np.array([r["compatibilita_pct"] for r in top], dtype=np.float32)
    pesi = pesi / pesi.sum()
    ripetizioni = np.maximum(1, np.round(pesi * max_immagini).astype(int))

    immagini = []
    for r, n in zip(top, ripetizioni):
        path = db[r["animale"]]["path_texture"]
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        immagini.extend([img] * int(n))
    return immagini


def carica_pipeline(device="cuda", usa_offload=True):
    """Carica base SD1.5 + ControlNet (scribble) + IP-Adapter. Import di
    torch/diffusers fatto qui dentro (non in cima al file) cosi' il resto
    del pacchetto render/ resta importabile anche su macchine senza questi
    pacchetti pesanti installati (es. per testare solo control_map.py)."""
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(MODEL_CONTROLNET, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_BASE, controlnet=controlnet, torch_dtype=dtype, safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(MODEL_IP_ADAPTER_REPO, subfolder=MODEL_IP_ADAPTER_SUBFOLDER,
                          weight_name=MODEL_IP_ADAPTER_WEIGHT)

    pipe = pipe.to(device)
    if device == "cuda" and usa_offload:
        # Utile se in futuro si passa a SDXL o si gira con meno VRAM; su una
        # 5080 con SD1.5 di norma non serve, ma non costa nulla lasciarlo.
        pipe.enable_model_cpu_offload()
    return pipe


def genera_render(coords_target, report, db, pipeline=None, device="cuda",
                   ip_adapter_scale=0.6, controlnet_scale=1.0,
                   num_inference_steps=30, guidance_scale=7.0,
                   seed=None, output_size=(512, 512)):
    """Genera l'immagine finale dell'animale ricostruito.

    coords_target: keypoint del fossile (da inference.estrai_coordinate).
    report: output di compatibility.genera_report_compatibilita.
    db: dict animali del database geometrico (da compatibility.carica_database).
    pipeline: se None ne viene caricata una nuova (lento: pesa scaricare/
        caricare i modelli). Passa una pipeline gia' caricata per generare
        piu' render in sequenza senza ricaricare tutto ogni volta.
    """
    import torch

    pipe = pipeline or carica_pipeline(device=device)

    control_map = disegna_control_map(coords_target, size=output_size)
    control_image = Image.fromarray(control_map).convert("RGB")

    immagini_riferimento = _lista_donatori_pesata(report, db)
    if immagini_riferimento:
        pipe.set_ip_adapter_scale(ip_adapter_scale)
    else:
        print("[render] Nessuna immagine di riferimento disponibile per IP-Adapter: "
              "il render si basera' solo su ControlNet + prompt testuale.")

    prompt, negative_prompt = costruisci_prompt(report)

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=output_size[0],
        height=output_size[1],
        generator=generator,
    )
    if immagini_riferimento:
        kwargs["ip_adapter_image"] = immagini_riferimento

    risultato = pipe(**kwargs)
    return risultato.images[0], control_image


def salva_render(immagine, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    immagine.save(output_path)
    print(f"Render salvato in: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Nota per un controllo piu' preciso dei pesi IP-Adapter
# ---------------------------------------------------------------------------
# _lista_donatori_pesata approssima la media pesata ripetendo le immagini.
# Se in `diffusers` (versione installata sulla macchina con la GPU) e'
# disponibile `pipe.prepare_ip_adapter_image_embeds` con supporto a pesi
# espliciti, si puo' sostituire con una media pesata analitica sugli
# embedding invece che sulle immagini ripetute: la logica in
# `_lista_donatori_pesata` va allora sostituita da una funzione che chiama
# quella API con `report["ranking_specie"]` come pesi. Non l'ho implementata
# di default perche' la firma di quella funzione e' cambiata piu' volte tra
# le versioni di diffusers e non ho potuto verificarla contro un ambiente
# con GPU in questa sessione.


if __name__ == "__main__":
    import argparse
    from inference import estrai_coordinate
    from compatibility import carica_database, genera_report_compatibilita

    parser = argparse.ArgumentParser(description="Genera il render finale da un fossile.")
    parser.add_argument("--fossile", required=True, help="Immagine del fossile/scheletro (gia' preprocessata)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pesi_keypoint = os.path.join(_base_dir, "training_outputs_2", "Weights", "best_keypoint_detector.pth")
    db_path = os.path.join(_base_dir, "data", "processed", "geometric_database.json")

    coords = estrai_coordinate(args.fossile, pesi_keypoint)
    db, fattore_cranio = carica_database(db_path)
    report = genera_report_compatibilita(coords, db, fattore_cranio)

    immagine, control_image = genera_render(coords, report, db, device=args.device, seed=args.seed)

    out_dir = os.path.join(_base_dir, "outputs")
    salva_render(immagine, os.path.join(out_dir, "render_finale.png"))
    salva_render(control_image, os.path.join(out_dir, "render_control_map.png"))
