"""
main.py (Version 0)

Ablation-study comparison script: runs the same skeleton image through the
three Version 0 baselines (Simple U-Net, PatchGAN generator, ControlNet)
and saves a side-by-side comparison figure - this is what produced
ablation_study_inference.png, cited in the report's Experimental Results.

Requires trained weights for all three models (not included in the repo,
too large for git - see README.md for how these are hosted, if at all):
    models/best_simple_UNET.pth
    models/best_generator_PatchGAN.pth
    models/best_controlnet.pth

Usage:
    python main.py --image path/to/skeleton.jpg
"""
import argparse
import os

import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from image_to_image import SimpleUNet as BaselineUNet
from PatchGan import SimpleUNet as GANGenerator

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_image_for_custom_models(image_path, device):
    """Prepares the [-1, 1] tensor expected by the Simple U-Net and GAN generator."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0).to(device), img


def main(test_image_path, weights_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # --- Load the custom models (U-Net & GAN) ---
    unet_model = BaselineUNet().to(device)
    unet_model.load_state_dict(torch.load(os.path.join(weights_dir, "best_simple_UNET.pth"),
                                           map_location=device))
    unet_model.eval()

    gan_generator = GANGenerator().to(device)
    gan_generator.load_state_dict(torch.load(os.path.join(weights_dir, "best_generator_PatchGAN.pth"),
                                              map_location=device))
    gan_generator.eval()

    # --- Load the ControlNet (diffusers) ---
    # A single .pth file (not a diffusers folder), hence from_single_file.
    controlnet = ControlNetModel.from_single_file(
        os.path.join(weights_dir, "best_controlnet.pth"),
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",  # base foundation model, not the ControlNet path
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # --- Inference ---
    print("Processing tensors...")

    tensor_input, img_original = load_image_for_custom_models(test_image_path, device)
    pil_input = Image.fromarray(img_original)

    with torch.no_grad():
        out_unet = unet_model(tensor_input)
        out_unet = (out_unet.squeeze().permute(1, 2, 0) * 0.5) + 0.5  # denormalize to [0,1]
        out_unet = out_unet.cpu().numpy()

        out_gan = gan_generator(tensor_input)
        out_gan = (out_gan.squeeze().permute(1, 2, 0) * 0.5) + 0.5
        out_gan = out_gan.cpu().numpy()

    # Empty prompt: tests the purity of the geometric translation (ablation).
    out_controlnet = pipe(
        prompt="Generate the animal corresponding to the skeleton shown in the input image. "
               "Infer the animal on your own and produce a realistic image, starting from the "
               "skeleton and consistent with it.",
        image=pil_input,
        num_inference_steps=20,
    ).images[0]

    # --- Comparison grid ---
    print("Generating comparison grid...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_original)
    axes[0].set_title("Input (Skeleton)")
    axes[0].axis("off")

    axes[1].imshow(out_unet.clip(0, 1))
    axes[1].set_title("Simple U-Net (Baseline)")
    axes[1].axis("off")

    axes[2].imshow(out_gan.clip(0, 1))
    axes[2].set_title("Pix2Pix GAN (Proposed)")
    axes[2].axis("off")

    axes[3].imshow(out_controlnet)
    axes[3].set_title("ControlNet (Foundation)")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = os.path.join(_BASE_DIR, "ablation_study_inference.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version 0 ablation study: U-Net vs PatchGAN vs ControlNet.")
    parser.add_argument("--image", default=os.path.join(_BASE_DIR, "t-rex.jpg"),
                         help="Skeleton image to run through all three baselines.")
    parser.add_argument("--weights-dir", default=os.path.join(_BASE_DIR, "models"),
                         help="Folder containing best_simple_UNET.pth, best_generator_PatchGAN.pth, "
                              "best_controlnet.pth.")
    args = parser.parse_args()
    main(args.image, args.weights_dir)
