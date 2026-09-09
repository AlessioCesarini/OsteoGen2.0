import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"                              # Previene crash di librerie CPU su Windows

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

# ==========================================
# CONFIGURAZIONE PATH E PARAMETRI
# ==========================================
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"                         # Il Foundation Model originale
CONTROLNET_PATH = "C:\\Users\\alexc\\Desktop\\GitHub Projects\\OsteoGen\\OsteoGen_Version1\\Training_Osteogen_Controlnet\\controlnet_best_model"   # La directory con i pesi che hai appena addestrato
TEST_IMAGE_PATH = "C:\\Users\\alexc\\Desktop\\GitHub Projects\\OsteoGen\\OsteoGen_Version1\\t-rex.jpg"                   # INSERISCI QUI il path di uno scheletro (es. un T-Rex o un animale non nel dataset)
OUTPUT_DIR = "results"                                                   # Cartella dove verranno salvate le immagini generate
DEVICE = "cuda"

# I due prompt per l'Ablation Study della tesi
PROMPT_ZERO_SHOT = "A full-body three-quarter view photo of an unknown biological animal, characterized by highly detailed and photorealistic textures, 8K resolution, studio lighting, and fully isolated against an absolute black background."
PROMPT_GUIDED = "A full-body three-quarter view photo of a Tyrannosaurus Rex dinosaur, characterized by highly detailed and photorealistic thick scaly reptilian skin, 8K resolution, studio lighting, and fully isolated against an absolute black background."

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("DEBUG - Caricamento della ControlNet addestrata...")
    # Carica esclusivamente la tua ControlNet personalizzata dal disco locale, forzando i tensori a 16 bit per la RTX 5080
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    
    print("DEBUG - Inizializzazione della Pipeline completa...")
    # Monta la tua ControlNet sul Foundation Model base, scaricando i restanti componenti (UNet, VAE) dalla cache
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        safety_checker=None                                              # Disabilitato per evitare falsi positivi su ossa/anatomia
    ).to(DEVICE)

    
    # UniPC è uno scheduler modernissimo: genera immagini eccellenti in soli 20-25 step, dimezzando i tempi di attesa
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    print(f"DEBUG - Caricamento immagine di input da: {TEST_IMAGE_PATH}")
    # Carica l'immagine dello scheletro e la forza in formato RGB
    init_image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    
    # Parametri di generazione (Classifier-Free Guidance e Steps)
    num_inference_steps = 25
    guidance_scale = 7.5                                                 # Valore standard ottimale: bilancia aderenza al testo e libertà generativa

    print("\n🎨 1/2 Generazione Zero-Shot (Prompt Generico)...")
    # Genera l'immagine basandosi unicamente sullo scheletro senza fornire l'identità dell'animale
    image_zero_shot = pipeline(
        PROMPT_ZERO_SHOT,
        image=init_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    print("🎨 2/2 Generazione Guidata (Prompt Specifico)...")
    # Genera l'immagine forzando texture specifiche (es. T-Rex, squame) sulla stessa struttura ossea
    image_guided = pipeline(
        PROMPT_GUIDED,
        image=init_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # --- SALVATAGGIO GRIGLIA COMPARATIVA PER LA TESI ---
    print("\n💾 Salvataggio della griglia di comparazione...")
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
    output_path = os.path.join(OUTPUT_DIR, "ablation_study_results.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Salva anche le immagini singole a piena risoluzione (1024x1024 o 512x512 nativi)
    image_zero_shot.save(os.path.join(OUTPUT_DIR, "zero_shot_raw.png"))
    image_guided.save(os.path.join(OUTPUT_DIR, "guided_raw.png"))

    print(f"✅ Inferenza completata! Risultati salvati in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()