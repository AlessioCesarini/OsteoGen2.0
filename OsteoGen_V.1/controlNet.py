import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoids a CPU-library crash on Windows

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from dataset import OsteoDataset

MODEL_ID = "runwayml/stable-diffusion-v1-5"
BATCH_SIZE = 4
# Updates the weights every 4 batches, simulating an effective batch size
# of 16 to stabilize the loss.
GRADIENT_ACCUMULATION_STEPS = 4
# Low learning rate, required for fine-tuning without destroying the
# pretrained weights.
LEARNING_RATE = 1e-5
EPOCHS = 1000
SAVE_DIR = "Training_Osteogen_Controlnet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATIENCE = 60  # epochs without improvement before early stopping
MIN_DELTA = 1e-4  # minimum loss decrease to count as a real improvement


def plot_loss(epoch_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', markersize=3, linestyle='-', color='b', label='Training Loss (MSE)')
    plt.title('ControlNet Training Loss Convergence - OsteoGen', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE)

    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE)

    # Clone the ControlNet from the U-Net's own encoder weights, so it
    # already recognizes images instead of starting from scratch.
    controlnet = ControlNetModel.from_unet(unet).to(DEVICE)

    # Freeze the whole foundation model; only the ControlNet is trained.
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    dataset = OsteoDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Automatic Mixed Precision, for 16-bit throughput on a capable GPU.
    scaler = torch.amp.GradScaler('cuda')

    # Always track the same first sample to visualize its evolution over
    # training, consistently across epochs.
    val_sample = dataset[0]
    val_prompt = val_sample["text"]
    if val_prompt == "":
        # Prompt dropout may have zeroed it out; use a generic fallback for
        # the visual test specifically.
        val_prompt = "A highly detailed, realistic photo of an animal, fully isolated against a pure black background."

    val_skeleton_pil = TF.to_pil_image(val_sample["conditioning_pixel_values"])
    val_real_pil = TF.to_pil_image((val_sample["pixel_values"] * 0.5) + 0.5)

    # UniPC: a fast scheduler, good visual test quality in ~20 steps instead of 50.
    val_scheduler = UniPCMultistepScheduler.from_config(noise_scheduler.config)

    epoch_losses = []
    best_loss = float('inf')
    patience_counter = 0

    try:
        for epoch in range(EPOCHS):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
            epoch_loss_sum = 0.0

            for step, batch in enumerate(progress_bar):
                pixel_values = batch["pixel_values"].to(DEVICE)  # real target animal, [-1, 1]
                cond_pixel_values = batch["conditioning_pixel_values"].to(DEVICE)  # skeleton, [0, 1]
                text_prompts = batch["text"]  # may be empty strings (prompt dropout)

                text_inputs = tokenizer(text_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]

                with torch.no_grad():
                    # scaling_factor (~0.18215) calibrates the latent variance for SD.
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=cond_pixel_values,
                        return_dict=False,
                    )

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # Cast to float32 for the loss: computing it directly in
                    # bfloat16 can be numerically unstable.
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})
                epoch_loss_sum += loss.item() * GRADIENT_ACCUMULATION_STEPS

            avg_epoch_loss = epoch_loss_sum / len(dataloader)
            epoch_losses.append(avg_epoch_loss)

            # --- Early stopping + best-model checkpoint ---
            if avg_epoch_loss < (best_loss - MIN_DELTA):
                best_loss = avg_epoch_loss
                patience_counter = 0
                best_model_path = os.path.join(SAVE_DIR, "controlnet_best_model")
                controlnet.save_pretrained(best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

            # --- Periodic checkpoint, every 50 epochs ---
            if (epoch + 1) % 50 == 0:
                backup_path = os.path.join(SAVE_DIR, f"controlnet_epoch_{epoch + 1}")
                controlnet.save_pretrained(backup_path)

            # --- Visual progress check, every 10 epochs ---
            if (epoch + 1) % 10 == 0:
                controlnet.eval()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        # Reuse the models already in memory - no extra disk
                        # load, mirrors a real inference run.
                        pipeline = StableDiffusionControlNetPipeline(
                            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                            unet=unet, controlnet=controlnet, scheduler=val_scheduler,
                            safety_checker=None, feature_extractor=None
                        ).to(DEVICE)
                        pipeline.set_progress_bar_config(disable=True)

                        pred_image = pipeline(
                            val_prompt,
                            image=val_skeleton_pil,
                            num_inference_steps=20,
                            guidance_scale=7.5
                        ).images[0]

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].imshow(val_skeleton_pil)
                axs[0].set_title("Input (Skeleton)", fontsize=12, fontweight='bold')
                axs[0].axis('off')

                axs[1].imshow(pred_image)
                axs[1].set_title(f"Prediction (Epoch {epoch + 1})", fontsize=12, fontweight='bold', color='green')
                axs[1].axis('off')

                axs[2].imshow(val_real_pil)
                axs[2].set_title("Ground Truth (Target)", fontsize=12, fontweight='bold')
                axs[2].axis('off')

                plt.tight_layout()
                val_path = os.path.join(SAVE_DIR, f"visual_test_epoch_{epoch + 1}.png")
                plt.savefig(val_path, dpi=150)
                plt.close()

                controlnet.train()

    except KeyboardInterrupt:
        interrupt_path = os.path.join(SAVE_DIR, "controlnet_interrupted")
        controlnet.save_pretrained(interrupt_path)

    finally:
        if len(epoch_losses) > 0:
            graph_path = os.path.join(SAVE_DIR, "osteogen_training_loss.png")
            plot_loss(epoch_losses, graph_path)


if __name__ == "__main__":
    main()
