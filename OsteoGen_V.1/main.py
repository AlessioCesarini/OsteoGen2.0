"""
main.py (Version 1)

Runs the Version 1 prompt-conditioned ControlNet on one skeleton image
twice - once with a generic ("unknown biological animal") prompt, once
with an explicit T-Rex prompt - and saves the comparison. This is what
produced the "zero-shot vs guided" figures cited in the report.

Requires a trained ControlNet checkpoint (a diffusers-format folder). Not
included in the repo (too large for git): auto-downloaded from the "v1/"
folder of the shared Hugging Face repo on first run (see
../weights_hub.py), same as running this through the top-level run.py -
no manual setup needed either way.

Usage:
    python main.py --image path/to/skeleton.jpg
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids a CPU-library crash on Windows

if sys.platform == "win32":
    # Legacy Windows consoles often use a non-UTF-8 codepage, which garbles
    # non-ASCII console output. Force UTF-8 for both the console and
    # Python's own stdout/stderr.
    os.system("chcp 65001 >nul")
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_BASE_DIR)
sys.path.insert(0, _BASE_DIR)
sys.path.insert(0, _ROOT_DIR)

from bootstrap import ensure_packages

ensure_packages([
    ("torch", "torch"), ("matplotlib", "matplotlib"), ("PIL", "Pillow"),
    ("diffusers", "diffusers"), ("transformers", "transformers"),
    ("accelerate", "accelerate"), ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface_hub"),
], "Version 1")

import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from weights_hub import resolve_folder

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# The two prompts for the thesis's ablation study.
PROMPT_ZERO_SHOT = ("A full-body three-quarter view photo of an unknown biological animal, "
                     "featuring highly detailed and photorealistic textures, 8K resolution, "
                     "studio lighting, and fully isolated against an absolute black background.")
PROMPT_GUIDED = ("A full-body three-quarter view photo of a Tyrannosaurus Rex dinosaur, "
                  "characterized by highly detailed and photorealistic thick scaly reptilian skin, "
                  "8K resolution, studio lighting, and fully isolated against an absolute black background.")


def main(image_path, controlnet_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cpu":
        print("[setup] No CUDA GPU detected by PyTorch - this will be much slower on "
              "CPU. If this machine has an NVIDIA GPU, check with:\n"
              "    python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"\n"
              "and, if torch.version.cuda is None, reinstall PyTorch with the correct "
              "CUDA build from https://pytorch.org/get-started/locally/")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading the trained ControlNet...")
    controlnet_path = resolve_folder("v1/controlnet_best_model", controlnet_path)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype).to(device)

    print("Assembling the full pipeline...")
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,  # disabled to avoid false positives on bones/anatomy
    ).to(device)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    print(f"Loading input image from: {image_path}")
    init_image = Image.open(image_path).convert("RGB")

    num_inference_steps = 25
    guidance_scale = 7.5

    print("\n1/2 Zero-shot generation (generic prompt)...")
    image_zero_shot = pipeline(
        PROMPT_ZERO_SHOT,
        image=init_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    print("2/2 Guided generation (specific prompt)...")
    image_guided = pipeline(
        PROMPT_GUIDED,
        image=init_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    print("\nSaving the comparison grid...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(init_image)
    axs[0].set_title("Input Skeleton (Condition)", fontsize=14, fontweight='bold')
    axs[0].axis('off')

    axs[1].imshow(image_zero_shot)
    axs[1].set_title("Zero-Shot Prediction\n(Generic Prompt)", fontsize=14, fontweight='bold', color='darkblue')
    axs[1].axis('off')

    axs[2].imshow(image_guided)
    axs[2].set_title("Guided Prediction\n(Semantic Prompt)", fontsize=14, fontweight='bold', color='darkgreen')
    axs[2].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "ablation_study_results.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Full-resolution single images too.
    image_zero_shot.save(os.path.join(output_dir, "zero_shot_raw.png"))
    image_guided.save(os.path.join(output_dir, "guided_raw.png"))

    print(f"Done! Results saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version 1 ControlNet: zero-shot vs guided prompt comparison.")
    parser.add_argument("--image", default=os.path.join(_BASE_DIR, "t-rex.jpg"),
                         help="Skeleton image to condition on.")
    parser.add_argument("--controlnet-dir",
                         default=os.path.join(_BASE_DIR, "Training_Osteogen_Controlnet", "controlnet_best_model"),
                         help="Trained ControlNet checkpoint folder (diffusers format). "
                              "Auto-downloaded here if missing.")
    parser.add_argument("--output-dir", default=os.path.join(_BASE_DIR, "results"))
    args = parser.parse_args()
    main(args.image, args.controlnet_dir, args.output_dir)
