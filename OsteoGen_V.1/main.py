"""
main.py (Version 1)

Runs the Version 1 prompt-conditioned ControlNet on one skeleton image
twice - once with a generic ("unknown biological animal") prompt, once
with an explicit T-Rex prompt - and saves the comparison. This is what
produced the "zero-shot vs guided" figures cited in the report.

Requires a trained ControlNet checkpoint (a diffusers-format folder, not
included in the repo - too large for git; see README.md for how these are
hosted, if at all):
    Training_Osteogen_Controlnet/controlnet_best_model/

Usage:
    python main.py --image path/to/skeleton.jpg
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids a CPU-library crash on Windows

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    os.makedirs(output_dir, exist_ok=True)

    print("Loading the trained ControlNet...")
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
                         help="Trained ControlNet checkpoint folder (diffusers format).")
    parser.add_argument("--output-dir", default=os.path.join(_BASE_DIR, "results"))
    args = parser.parse_args()
    main(args.image, args.controlnet_dir, args.output_dir)
