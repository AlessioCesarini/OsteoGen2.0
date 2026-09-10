"""
render/generate.py

The actual generative stage: replaces the rigid-warping collage
(render/control_map.py, which now only produces supporting material) with
a local-diffusion render, conditioned on:

- ControlNet on the target skeleton's control map (disegna_control_map):
  constrains pose and proportions to the real fossil.
- IP-Adapter on the best donors' photos (report["ranking_specie"]): gives
  the model a visual texture/color reference, weighted by compatibility -
  the same idea as the first manual experiment done with a generic image
  generation tool (e.g. 50% eagle + 30% chicken + 20% other), but with
  open weights running locally, no paid API.
- A text prompt automatically built from the report, as additional
  guidance (weaker than ControlNet+IP-Adapter, but helps the style).

WARNING: this module requires a CUDA GPU (or any torch-supported
accelerator) and the weights are downloaded from Hugging Face on first use
(a few GB). It was not run end-to-end while developing this module: the
machine used to write the code has no NVIDIA GPU. Minor adjustments
(argument names, `diffusers` versions) should be expected on the first run
on the RTX 5080.
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

MAX_DONATORI_IP_ADAPTER = 4       # how many animals from ranking_specie to use as visual reference
MAX_IMMAGINI_MEDIA_PESATA = 10    # total copies in the "weighted by repetition" list (see below)


def costruisci_prompt(report, max_specie=3):
    """Supporting text prompt: on its own it doesn't guarantee coherence
    (that's what ControlNet and IP-Adapter are for), but it helps the
    overall style and gives a readable fallback if IP-Adapter is unavailable.

    The prompt is in English (SD1.5 is trained on English captions: an
    Italian prompt hurts the model's adherence), so the DB's species names
    (in Italian) are translated via display_names."""
    from display_names import species_name
    top = report["ranking_specie"][:max_specie]
    mix = ", ".join(f"{r['compatibilita_pct']:.0f}% {species_name(r['animale'])}" for r in top)
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
    """Builds the list of images to pass to IP-Adapter, repeating each
    donor in proportion to its compatibility weight.

    This is a robust way (doesn't depend on diffusers internal APIs that
    change between versions) to approximate a weighted average of the
    embeddings: IP-Adapter already averages the embeddings when given a
    list of images, so repeating an image 5 times out of 10 instead of 1
    out of 10 weighs it about 5x in the final average. For exact control
    (an analytic weighted average over the embeddings) see the note at the
    bottom of the file."""
    top = [r for r in report["ranking_specie"][:max_donatori] if r["animale"] in db]
    if not top:
        return []

    pesi = np.array([r["compatibilita_pct"] for r in top], dtype=np.float32)
    pesi = pesi / pesi.sum()
    ripetizioni = np.maximum(1, np.round(pesi * max_immagini).astype(int))

    from dataset_paths import resolve_dataset_path

    immagini = []
    for r, n in zip(top, ripetizioni):
        path = resolve_dataset_path(db[r["animale"]].get("path_texture"), "target_y", r["animale"])
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        immagini.extend([img] * int(n))
    return immagini


def carica_pipeline(device="cuda", usa_offload=True):
    """Loads SD1.5 base + ControlNet (scribble) + IP-Adapter. torch/diffusers
    are imported here (not at the top of the file) so the rest of the
    render/ package stays importable even on machines without these heavy
    packages installed (e.g. to test just control_map.py)."""
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
        # Useful if this ever moves to SDXL or runs with less VRAM; on a
        # 5080 with SD1.5 it's normally unnecessary, but costs nothing to
        # leave in.
        pipe.enable_model_cpu_offload()
    return pipe


def genera_render(coords_target, report, db, pipeline=None, device="cuda",
                   ip_adapter_scale=0.6, controlnet_scale=1.0,
                   num_inference_steps=30, guidance_scale=7.0,
                   seed=None, output_size=(512, 512)):
    """Generates the final image of the reconstructed animal.

    coords_target: the fossil's keypoints (from inference.estrai_coordinate).
    report: output of compatibility.genera_report_compatibilita.
    db: geometric database animal dict (from compatibility.carica_database).
    pipeline: if None, a new one is loaded (slow: downloading/loading the
        models is expensive). Pass an already-loaded pipeline to generate
        several renders in sequence without reloading everything each time.
    """
    import torch

    pipe = pipeline or carica_pipeline(device=device)

    control_map = disegna_control_map(coords_target, size=output_size)
    control_image = Image.fromarray(control_map).convert("RGB")

    immagini_riferimento = _lista_donatori_pesata(report, db)
    if immagini_riferimento:
        pipe.set_ip_adapter_scale(ip_adapter_scale)
    else:
        print("[render] No reference image available for IP-Adapter: "
              "the render will rely on ControlNet + text prompt only.")
        # carica_pipeline() always loads IP-Adapter, so without this the
        # UNet still expects image embeds on every call and pipe(...) below
        # crashes with "argument of type 'NoneType' is not iterable" the
        # moment there is nothing to condition it on.
        pipe.unload_ip_adapter()

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
        kwargs["ip_adapter_image"] = [immagini_riferimento]

    risultato = pipe(**kwargs)
    return risultato.images[0], control_image


def salva_render(immagine, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    immagine.save(output_path)
    print(f"Render saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Note on more precise control of the IP-Adapter weights
# ---------------------------------------------------------------------------
# _lista_donatori_pesata approximates the weighted average by repeating
# images. If the `diffusers` version installed on the GPU machine has
# `pipe.prepare_ip_adapter_image_embeds` with explicit weight support, it
# could be replaced with an analytic weighted average over the embeddings
# instead of repeated images: `_lista_donatori_pesata`'s logic would then
# need to be replaced by a function that calls that API with
# `report["ranking_specie"]` as weights. Not implemented by default because
# that function's signature has changed multiple times across diffusers
# versions and it wasn't possible to verify it against a GPU environment
# in this session.


if __name__ == "__main__":
    import argparse
    from inference import estrai_coordinate
    from compatibility import carica_database, genera_report_compatibilita

    parser = argparse.ArgumentParser(description="Generate the final render from a fossil.")
    parser.add_argument("--image", dest="fossile", required=True, help="Fossil/skeleton image (already preprocessed)")
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
