import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ==========================================
# 1. IMPORTA LE TUE ARCHITETTURE
# ==========================================
from image_to_image import SimpleUNet as BaselineUNet
from PatchGan import SimpleUNet as GANGenerator

def load_image_for_custom_models(image_path):
    """Prepara il tensore [-1, 1] per Simple U-Net e GAN"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0).cuda(), img

def main(test_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio inferenza su {device}...")

# ==========================================
    # 2. CARICAMENTO MODELLI CUSTOM (U-Net & GAN)
    # ==========================================
    unet_model = BaselineUNet().to(device)
    unet_model.load_state_dict(torch.load("models/best_simple_UNET.pth"))
    unet_model.eval()

    gan_generator = GANGenerator().to(device)
    gan_generator.load_state_dict(torch.load("models/best_generator_PatchGAN.pth"))
    gan_generator.eval()

    # ==========================================
    # 3. CARICAMENTO CONTROLNET (Diffusers)
    # ==========================================
    # CORREZIONE: Essendo un file .pth e non una cartella, si usa from_single_file
    controlnet = ControlNetModel.from_single_file(
        "models/best_controlnet.pth", 
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", # CORREZIONE: Qui va il modello base, non il path del ControlNet
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # ==========================================
    # 4. INFERENZA
    # ==========================================
    print("Elaborazione tensori...")
    
    # Input per U-Net e GAN
    tensor_input, img_original = load_image_for_custom_models(test_image_path)
    
    # Input per ControlNet (richiede PIL Image)
    pil_input = Image.fromarray(img_original)
    
    with torch.no_grad():
        # A. Inferenza Simple U-Net
        out_unet = unet_model(tensor_input)
        out_unet = (out_unet.squeeze().permute(1, 2, 0) * 0.5) + 0.5 # Denormalizza a [0,1]
        out_unet = out_unet.cpu().numpy()

        # B. Inferenza PatchGAN (Solo Generatore)
        out_gan = gan_generator(tensor_input)
        out_gan = (out_gan.squeeze().permute(1, 2, 0) * 0.5) + 0.5
        out_gan = out_gan.cpu().numpy()

    # C. Inferenza ControlNet
    # Usiamo un prompt vuoto per testare la purezza della traduzione geometrica (Ablation)
    out_controlnet = pipe(
        prompt="Generami l'animale corrispondente allo scheletro che vedi nell'immagine in Input. Dovrai capire autonomamente di quale animale si tratta e generare un'immagine realistica e partendo dallo scheletro e coerente con esso.", 
        image=pil_input, 
        num_inference_steps=20
    ).images[0]

    # ==========================================
    # 5. VISUALIZZAZIONE ABLATION STUDY
    # ==========================================
    print("Generazione griglia di confronto...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_original)
    axes[0].set_title("Input (Scheletro)")
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
    plt.savefig("ablation_study_inference.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main("C:\\Users\\alexc\\Desktop\\GitHub Projects\\OsteoGen\\OsteoGen\\t-rex.jpg")