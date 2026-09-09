import os
import torch
import wandb
import lpips
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers import ControlNetModel, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from src.dataset import OsteoDataset 
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_architecture(device):
    wandb.init(
        project="OsteoGen-ControlNet", 
        name="SD_1.5_LPIPS_Run_32GB",
        config={
            "learning_rate": 1e-4, 
            "batch_size": 4, 
            "epochs": 2000 
        }
    )   
    
    print("Caricamento del Foundation Model...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device) 
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    print("Inizializzazione ControlNet e Perceptual Loss...")
    controlnet = ControlNetModel.from_unet(unet).to(device) 
    controlnet.train() 
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) 
    
    return {
        "vae": vae, "text_encoder": text_encoder, "unet": unet,
        "controlnet": controlnet, "scheduler": scheduler,
        "tokenizer": tokenizer, "loss_fn_vgg": loss_fn_vgg,
        "config": wandb.config
    }

def train_model(components, device):
    print("Preparazione del DataLoader...")
    dataset = OsteoDataset(data_dir="OsteoGen/data/processed")
    dataloader = DataLoader(dataset, batch_size=components["config"].batch_size, shuffle=True)
    
    optimizer = AdamW(components["controlnet"].parameters(), lr=components["config"].learning_rate)
    
    best_loss = float('inf')
    patience = 250 
    epochs_no_improve = 0
    save_dir = "training_dir_controlNet"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Inizio dell'addestramento latente...")
    try:
        for epoch in range(components["config"].epochs):
            epoch_loss = 0.0
            components["controlnet"].train()
            progress_bar = tqdm(dataloader, desc=f"Epoca {epoch+1}/{components['config'].epochs}")
            
            for batch_idx, (ossa, animali_veri) in enumerate(progress_bar):
                ossa = ossa.to(device)
                animali_veri = animali_veri.to(device)
                
                # 1. Compressione Nello Spazio Latente
                latents = components["vae"].encode(animali_veri).latent_dist.sample()
                latents = latents * components["vae"].config.scaling_factor
                
                # 2. Aggiunta Rumore Stocastico
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, components["scheduler"].config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = components["scheduler"].add_noise(latents, noise, timesteps)
                
                # 3. Il Fantasma Matematico (Stringa Vuota)
                text_inputs = components["tokenizer"]([""] * bsz, padding="max_length", max_length=components["tokenizer"].model_max_length, return_tensors="pt").to(device)
                encoder_hidden_states = components["text_encoder"](text_inputs.input_ids)[0]
                
                # 4. Forward Pass ControlNet + U-Net con Normalizzazione Spaziale
                ossa_cond = (ossa / 2 + 0.5).clamp(0, 1)
                down_res, mid_res = components["controlnet"](
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=ossa_cond, return_dict=False
                )
                
                noise_pred = components["unet"](
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res
                ).sample
                
                # 5. Calcolo Errore e Ottimizzazione
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                wandb.log({"batch_loss": loss.item()})
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
            # === CONTROLLO FINE EPOCA (EARLY STOPPING E BEST MODEL) ===
            avg_epoch_loss = epoch_loss / len(dataloader)
            wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch + 1})
            
            if avg_epoch_loss < best_loss - 0.0001:
                best_loss = avg_epoch_loss
                epochs_no_improve = 0
                torch.save(components["controlnet"].state_dict(), f"{save_dir}/best_controlnet.pth")
                print(f"\n>>> Epoca {epoch+1}: Nuovo record MSE ({best_loss:.4f})! File best_controlnet.pth aggiornato.")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"\n[Early Stop] Nessun miglioramento da {patience} epoche. Raggiunto il plateau di addestramento.")
                break

            # === SALVATAGGIO SERIALE (Ogni 100 epoche) ===
            if (epoch + 1) % 100 == 0:
                checkpoint_path = f"{save_dir}/controlnet_epoca_{epoch+1}.pth"
                torch.save(components["controlnet"].state_dict(), checkpoint_path)
                print(f"\n>>> Backup Seriale: Modello salvato in {checkpoint_path}")
                
            # === FEEDBACK VISIVO (Ogni 10 epoche) ===
            if (epoch + 1) % 10 == 0:
                components["controlnet"].eval()
                from diffusers import StableDiffusionControlNetPipeline
                
                val_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    vae=components["vae"], text_encoder=components["text_encoder"], tokenizer=components["tokenizer"],
                    unet=components["unet"], controlnet=components["controlnet"], safety_checker=None, torch_dtype=torch.float32
                ).to(device)
                val_pipe.set_progress_bar_config(disable=True)
                
                with torch.no_grad():
                    # Generazione rigorosamente incondizionata
                    ossa_val_cond = (ossa[0] / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                    val_image = val_pipe(prompt="", image=ossa_val_cond, num_inference_steps=20).images[0]
                    
                    def tensor_to_pil(t):
                        t = t.clone().cpu()
                        if t.min() < 0:
                            t = (t / 2 + 0.5).clamp(0, 1)
                        return TF.to_pil_image(t)
                        
                    input_pil = tensor_to_pil(ossa[0])
                    target_pil = tensor_to_pil(animali_veri[0])
                    
                    w1, w2, w3 = input_pil.width, val_image.width, target_pil.width
                    h_max = max(input_pil.height, val_image.height, target_pil.height)
                    
                    combined_image = Image.new('RGB', (w1 + w2 + w3, h_max))
                    combined_image.paste(input_pil, (0, 0))
                    combined_image.paste(val_image, (w1, 0))
                    combined_image.paste(target_pil, (w1 + w2, 0))
                    
                    combined_image.save(f"{save_dir}/render_epoca_{epoch+1}.png")
                    wandb.log({f"Generazione_Visiva": wandb.Image(combined_image, caption=f"Epoca {epoch+1} - SX: Ossa | CENTRO: ControlNet | DX: Target")})

                del val_pipe 
                components["controlnet"].train()

    except KeyboardInterrupt:
        print(f"\n[Interruzione Manuale] Arresto forzato all'epoca {epoch+1}.")
        
    print(f"Addestramento concluso. L'ultimo record di best loss è {best_loss:.4f}.")
    wandb.finish()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_components = load_architecture(device)
    train_model(arch_components, device)