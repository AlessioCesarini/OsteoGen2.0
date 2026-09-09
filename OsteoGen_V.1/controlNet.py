# Importazione dei moduli per la gestione del sistema operativo (creazione cartelle e percorsi)
import os                                                                
# Importazione della libreria core di PyTorch per il calcolo tensoriale su GPU
import torch                                                             
# Importazione del modulo funzionale di PyTorch per le funzioni matematiche (come la MSE Loss)
import torch.nn.functional as F                                          
# Importazione di matplotlib per generare i grafici della loss pronti per l'inserimento nella tesi
import matplotlib.pyplot as plt                                          
# Importazione delle utilità per convertire i tensori in immagini visualizzabili (PIL) durante il test visivo
import torchvision.transforms.functional as TF                           
# Importazione del DataLoader per parallelizzare il caricamento dei batch da CPU a GPU
from torch.utils.data import DataLoader                                  
# Importazione di tqdm per avere una barra di avanzamento visiva nel terminale durante i cicli di training
from tqdm import tqdm                                                    
# Importazione delle architetture pre-addestrate dalla libreria Diffusers di Hugging Face
from diffusers import (                                                  
    AutoencoderKL,               # Modello VAE per comprimere le immagini nello spazio latente
    UNet2DConditionModel,        # Modello UNet principale che esegue il denoising iterativo
    ControlNetModel,             # Modello ControlNet che vincolerà spazialmente la UNet
    DDPMScheduler,               # Scheduler matematico per calcolare il rumore da aggiungere nel forward pass
    StableDiffusionControlNetPipeline, # Struttura unificata per eseguire l'inferenza di test in modo semplice
    UniPCMultistepScheduler      # Scheduler super-veloce che genera ottime immagini in pochissimi step (usato nel test visivo)
)
# Importazione dei moduli NLP per elaborare i prompt in lingua inglese
from transformers import CLIPTextModel, CLIPTokenizer                    
# Importazione della tua classe Dataset personalizzata e scorporata da internet
from dataset import OsteoDataset                                         
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# ID del modello base su cui innestiamo la ControlNet (Foundation Model stabile)
MODEL_ID = "runwayml/stable-diffusion-v1-5"                              
# Quante immagini per volta vengono caricate. 4 è ottimale per bilanciare velocità e uso della VRAM (evita OOM)
BATCH_SIZE = 4                                                           
# Step di accumulazione: aggiorna i pesi ogni 4 batch simulando un batch size reale di 16, stabilizzando la loss
GRADIENT_ACCUMULATION_STEPS = 4                                          
# Tasso di apprendimento basso (1e-5) obbligatorio per fine-tuning per non distruggere i pesi preesistenti
LEARNING_RATE = 1e-5                                                     
# Numero massimo di epoche stimato per garantire la convergenza su un dataset molto piccolo (65 immagini)
EPOCHS = 1000   #Prima era 300                                                          
# Percorso locale dove salvare tutti i pesi, i modelli migliori e le griglie di validazione
SAVE_DIR = "Training_Osteogen_Controlnet"                                  
# Selettore automatico che forza l'uso della RTX 5080 tramite l'architettura CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"                  

# Numero massimo di epoche in cui tolleriamo che la loss non scenda prima di forzare lo stop (Early Stopping)
PATIENCE = 60      #Prima era 40                                                      
# La variazione minima (Delta) necessaria affinché una diminuzione della loss venga considerata valida e non puro rumore statistico
MIN_DELTA = 1e-4                                                         

# Funzione per disegnare e salvare il grafico dell'errore (MSE)
def plot_loss(epoch_losses, save_path):
    # Inizializza un canvas per il grafico con dimensioni ottimali per un documento cartaceo/PDF (rapporto 10:6)
    plt.figure(figsize=(10, 6))                                          
    # Disegna la linea blu collegando i valori medi della loss per ogni epoca
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', markersize=3, linestyle='-', color='b', label='Training Loss (MSE)')
    # Assegna un titolo chiaro e leggibile
    plt.title('ControlNet Training Loss Convergence - OsteoGen', fontsize=14, fontweight='bold')
    # Etichetta dell'asse X (tempo)
    plt.xlabel('Epochs', fontsize=12)
    # Etichetta dell'asse Y (l'errore quadratico medio che cerchiamo di minimizzare)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    # Attiva la griglia di sfondo tratteggiata per rendere quantificabili visivamente i valori
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)          
    # Inserisce la legenda in alto a destra
    plt.legend(loc='upper right')                                        
    # Calcola automaticamente i margini per evitare che il testo venga tagliato fuori dall'immagine
    plt.tight_layout()                                                   
    # Salva il file su disco forzando una risoluzione altissima (300 DPI) per evitare sgranature nella stampa della tesi
    plt.savefig(save_path, dpi=300)                                      
    # Libera la RAM di sistema allocata da Matplotlib
    plt.close()                                                          

# Entry point principale del training loop
def main():
    # Crea fisicamente la cartella su disco; se già esiste ignora l'istruzione e prosegue
    os.makedirs(SAVE_DIR, exist_ok=True)                                 
    
    # Inizializza il dizionario (Tokenizer) che divide i prompt di testo inglesi in sub-parole numeriche
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer") 
    # Inizializza il modello CLIP che trasforma i token in vettori semantici e lo carica nella VRAM della 5080
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE) 
    
    # Inizializza l'AutoEncoder per convertire i pixel RGB (512x512) nei Tensori Latenti (64x64) per la diffusione
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE) 
    # Inizializza la rete UNet che rappresenta il Foundation Model vero e proprio e la carica sulla GPU
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE) 
    
    # COPIA STRATEGICA: Crea la ControlNet clonando l'architettura e i pesi dell'encoder della UNet
    # Questo permette alla ControlNet di riconoscere da subito le immagini senza dover essere addestrata da zero
    controlnet = ControlNetModel.from_unet(unet).to(DEVICE)
    
    # Congela tutti i pesi del VAE disabilitando il calcolo dei gradienti. Evita di distruggere il Foundation Model
    vae.requires_grad_(False)                                            
    # Congela la UNet per mantenere intatte le texture realistiche (es. peli, squame) già note al modello
    unet.requires_grad_(False)                                           
    # Congela l'Encoder testuale per non alterare la mappa concettuale semantica della lingua inglese
    text_encoder.requires_grad_(False)                                   
    # Imposta la ControlNet in modalità attiva (Training). Calcolerà i gradienti esclusivamente per questa rete
    controlnet.train()                                                   
   
    
    # Inizializza il motore matematico che regola i timestep e distribuisce il rumore Gaussiano nelle epoche
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler") 
    # Imposta AdamW come ottimizzatore per aggiornare i pesi, utilizzando weight_decay per prevenire overfitting strutturale
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE, weight_decay=1e-2) 

    # Istanzia la classe OsteoDataset (esegue al volo il pre-caricamento in RAM e le traduzioni dal dizionario statico)
    dataset = OsteoDataset()                                             
    # Fornisce al training loop dei generatori batchizzati, attivando lo Shuffle per stocasticità
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) 

    # Inizializza lo Scaler per abilitare AMP (Automatic Mixed Precision). Essenziale per la 5080 per calcoli a 16-bit
    scaler = torch.amp.GradScaler('cuda')                                
    
    # Preleva sempre e solo il primissimo animale in memoria per testare costantemente la sua evoluzione visiva
    val_sample = dataset[0]                                              
    # Preleva la stringa di testo di quello specifico animale (es: "A full-body tre-quarter...")
    val_prompt = val_sample["text"]
    # Fallback di sicurezza: se per colpa del Dropout stocastico il prompt era vuoto, forzane uno generico per il test visivo
    if val_prompt == "": 
        val_prompt = "A highly detailed, realistic photo of an animal, fully isolated against a pure black background."
    
    # Trasforma il tensore [0, 1] dello scheletro fisso in immagine visualizzabile per la colonna 1 del grafico visivo
    val_skeleton_pil = TF.to_pil_image(val_sample["conditioning_pixel_values"])
    # De-normalizza il target animale da [-1, 1] a [0, 1] e lo trasforma in PIL per la colonna 3 del grafico visivo
    val_real_pil = TF.to_pil_image((val_sample["pixel_values"] * 0.5) + 0.5) 
    
    # Istanzia il campionatore UniPC, che permette di generare un test visivo nitido in soli 20 step anziché 50, salvando tempo
    val_scheduler = UniPCMultistepScheduler.from_config(noise_scheduler.config)

    # Inizializza la lista vuota per raccogliere i dati del grafico
    epoch_losses = []                                                    
    # Imposta la best_loss iniziale ad infinito, così qualsiasi primo valore sarà considerato il "migliore"
    best_loss = float('inf')                                             
    # Contatore azzerato per tenere traccia delle epoche trascorse senza miglioramenti utili
    patience_counter = 0                                                 
    
    # Blocco Try per gestire l'interruzione manuale sicura da tastiera durante le ore di addestramento
    try:
        # Loop esterno delle epoche (passa su tutto il dataset N volte)
        for epoch in range(EPOCHS):
            # Inizializza la barra tqdm iterando sul dataloader
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            # Azzera la somma della loss all'inizio di ogni nuova epoca
            epoch_loss_sum = 0.0                                         
            
            # Loop interno per processare ogni singolo batch da 4 immagini
            for step, batch in enumerate(progress_bar):
                # Sposta i Tensori Target Y (l'animale reale scalato [-1, 1]) sulla VRAM della RTX 5080
                pixel_values = batch["pixel_values"].to(DEVICE)              
                # Sposta i Tensori Input X (lo scheletro di condizionamento scalato [0, 1]) sulla VRAM 
                cond_pixel_values = batch["conditioning_pixel_values"].to(DEVICE) 
                # Estrae le 4 stringhe testuali (o stringhe vuote in caso di Dropout attivato)
                text_prompts = batch["text"]                                 
                
                # Taglia, normalizza e inserisce il padding ai testi forzandoli in tensori vettoriali da 77 slot max
                text_inputs = tokenizer(text_prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(DEVICE)
                
                # Context manager che disabilita il calcolo derivativo per risparmiare memoria durante l'encoding del testo
                with torch.no_grad():                                        
                    # Converte gli input testuali ID (numeri) in Embedding (Vettori di stato semantico ad alta dimensionalità)
                    encoder_hidden_states = text_encoder(text_inputs.input_ids)[0] 
                    
                # Context manager per usare il VAE pre-addestrato senza calcolare i gradienti
                with torch.no_grad():
                    # Esegue l'encode delle immagini degli animali, campionandole da pixel a spazio latente (distribuzione probabilistica)
                    # La moltiplicazione per scaling_factor (es. 0.18215) calibra la varianza per l'ingresso in Stable Diffusion
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    
                # Genera un tensore di Puro Rumore Gaussiano con la stessa identica grandezza dell'immagine latente
                noise = torch.randn_like(latents)
                # Misura quanti campioni ci sono in questo batch (es. 4)
                bsz = latents.shape[0]
                # Sceglie un "Timestamp" casuale da 0 a 1000 per ogni immagine del batch per stabilire quanto rumore applicare
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()
                # Applica effettivamente il calcolo matematico mischiando il rumore puro con i pixel latenti dell'animale
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Context manager AMP per forzare l'hardware RTX a usare la precisione bfloat16, dimezzando i tempi di calcolo senza perdita di stabilità
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):       
                    # Forward pass nella ControlNet, passando rumore, tempo, prompt testuale e soprattutto lo scheletro in pixel
                    # Restituisce i tensori residuali elaborati che serviranno come "binari di guida" spaziali
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,                                       
                        timesteps,                                           
                        encoder_hidden_states=encoder_hidden_states,         
                        controlnet_cond=cond_pixel_values,                   
                        return_dict=False,                                   
                    )
                    
                    # Forward pass nella UNet congelata, a cui forniamo in ingresso le uscite residuali generate dalla ControlNet (iniezione)
                    # La rete proverà a isolare/predire unicamente la traccia del Rumore
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples, 
                        mid_block_additional_residual=mid_block_res_sample,     
                        return_dict=False,
                    )[0]
                    
                    # Calcolo dell'errore (MSE) tra il rumore predetto dalla rete e il rumore reale generato casualmente prima
                    # Castiamo a float32 perché calcolare la loss in bfloat16 potrebbe causare instabilità nei decimali
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    # Dividiamo matematicamente l'errore per accumularlo, così l'ottimizzatore non reagirà a sbalzi troppo violenti 
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                    
                # Calcola il backward pass per derivare i gradienti dell'errore attraverso l'architettura in bfloat16 scalato
                scaler.scale(loss).backward()                                
                
                # Condizione di aggiornamento: verifica se abbiamo accumulato abbastanza gradienti (simulazione del batch size 16)
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    # Invia l'istruzione di discesa del gradiente all'ottimizzatore per aggiornare i pesi della ControlNet
                    scaler.step(optimizer)                                   
                    # Sincronizza ed aggiorna i pesi del gestore automatico (Scaler)
                    scaler.update()                                          
                    # Svuota i calcoli derivativi accumulati allocando il parametro set_to_none=True per liberare RAM per il loop successivo
                    optimizer.zero_grad(set_to_none=True)                    
                    
                # Aggiorna la dicitura visiva a destra della barra di caricamento scalando l'errore di nuovo per renderlo reale
                progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})
                # Aggiunge il valore reale calcolato della loss alla somma per calcolare la media finale dell'epoca
                epoch_loss_sum += loss.item() * GRADIENT_ACCUMULATION_STEPS
                
            # Calcola l'errore matematico medio di questa epoca dividendo la somma totale per il numero di cicli/batch
            avg_epoch_loss = epoch_loss_sum / len(dataloader)            
            # Appende questo dato in una lista permanente necessaria per l'asse Y del grafico da salvare
            epoch_losses.append(avg_epoch_loss)                          
            
            # --- BLOCCO LOGICA EARLY STOPPING E SALVATAGGIO OTTIMALE ---
            # Se la loss attuale è più bassa della miglior loss registrata storicamente, sottratto il fattore rumore (Delta)
            if avg_epoch_loss < (best_loss - MIN_DELTA):                 
                # Imposta il nuovo record assoluto
                best_loss = avg_epoch_loss                               
                # Resetta a zero il contatore di blocco
                patience_counter = 0                                     
                # Estrae ed imposta il percorso dedicato al modello matematicamente migliore dell'intero training
                best_model_path = os.path.join(SAVE_DIR, "controlnet_best_model")
                # Sovrascrive istantaneamente i pesi del miglior modello sul disco per non sprecare spazio con check inutili
                controlnet.save_pretrained(best_model_path)              
            else:
                # Se la loss non è migliorata, aggiunge una bandierina al contatore che ci porterà all'Early Stopping
                patience_counter += 1                                    
                
            # Se il numero di bandierine (Patience) raggiunge o supera le 40 epoche stabilite
            if patience_counter >= PATIENCE:
                # Forza l'uscita istantanea dal training per evitare pesantemente che il modello inizi ad imparare il rumore (Overfitting)
                break                                                    

            # --- BLOCCO SALVATAGGIO CHECKPOINT STATICI ---
            # Un controllo modulo: esegue questa riga ogni multiplo esatto di 50 epoche
            if (epoch + 1) % 50 == 0:
                # Definisce la cartella di backup con il nome formattato sull'epoca in corso
                backup_path = os.path.join(SAVE_DIR, f"controlnet_epoch_{epoch+1}")
                # Serializza e salva l'architettura in un binario HuggingFace
                controlnet.save_pretrained(backup_path)  

            # --- BLOCCO GENERAZIONE TEST VISIVO (L'Ablation Study Pratico) ---
            # Esegue il blocco ogni multiplo esatto di 10 epoche per tracciare il progresso visivo
            if (epoch + 1) % 10 == 0:
                # Spegne i dropout spaziali e i gradienti, mettendo la rete specificatamente in modalità di inferenza/generazione
                controlnet.eval()                                        
                # Inibisce esplicitamente a PyTorch di misurare i gradienti o calcolare derivate
                with torch.no_grad():
                    # Mantiene attivo lo standard mixed precision Bfloat16 per non crashare la VRAM
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        # Istanzia l'intera pipeline di Image-to-Image passandogli i modelli che si trovano GIA in RAM
                        # In questo modo non c'è caricamento disco aggiuntivo né OOM, e si simula esattamente un task di inferenza finale
                        pipeline = StableDiffusionControlNetPipeline(
                            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                            unet=unet, controlnet=controlnet, scheduler=val_scheduler,
                            safety_checker=None, feature_extractor=None
                        ).to(DEVICE)
                        # Disattiva la progress bar dell'inferenza singola per mantenere pulito il terminale
                        pipeline.set_progress_bar_config(disable=True)
                        
                        # Processa e genera l'immagine passandogli stringa, scheletro e soli 20 steps matematici
                        pred_image = pipeline(
                            val_prompt, 
                            image=val_skeleton_pil, 
                            num_inference_steps=20, 
                            guidance_scale=7.5
                        ).images[0]
                
                # Istanzia la griglia grafica Matplotlib con 1 riga e 3 colonne, di dimensione panoramica (15x5)
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                # Assegna il tensore Input (Lo Scheletro) PIL alla Colonna di sinistra (Indice 0)
                axs[0].imshow(val_skeleton_pil)
                # Stampa il titolo in grassetto
                axs[0].set_title("Input (Skeleton)", fontsize=12, fontweight='bold')
                # Spegne i bordi dell'asse X/Y per avere una grafica pulita (non sono dati cartesiani)
                axs[0].axis('off')
                
                # Assegna l'immagine in output generata istantaneamente dalla Rete alla Colonna Centrale (Indice 1)
                axs[1].imshow(pred_image)
                # Stampa il titolo indicando specificatamente il numero di epoca (per i report della Tesi)
                axs[1].set_title(f"Prediction (Epoch {epoch+1})", fontsize=12, fontweight='bold', color='green')
                # Spegne i bordi cartesiani
                axs[1].axis('off')
                
                # Assegna la foto reale del dataset alla colonna di destra (Indice 2) per comparare lo scostamento semantico (Ground Truth)
                axs[2].imshow(val_real_pil)
                # Stampa il titolo
                axs[2].set_title("Ground Truth (Target)", fontsize=12, fontweight='bold')
                # Spegne i bordi cartesiani
                axs[2].axis('off')
                
                # Allinea i margini delle 3 immagini eliminando gli spazi bianchi inutili
                plt.tight_layout()
                # Definisce il nome e la directory della griglia PNG basata sull'epoca corrente
                val_path = os.path.join(SAVE_DIR, f"visual_test_epoch_{epoch+1}.png")
                # Scrive l'immagine PNG sul disco con una risoluzione medio/alta
                plt.savefig(val_path, dpi=150)
                # Svuota e chiude il canvas Matplotlib
                plt.close()
                
                # Fondamentale: dopo aver generato l'immagine, RIMETTE la ControlNet in modalità addestramento per le epoche successive
                controlnet.train()                                       

    # Eccezione globale per intercettare l'interruzione di processo dal CMD di Windows (La shortcut CTRL+C)
    except KeyboardInterrupt:
        # Percorso dedicato al salvataggio dello snapshot di recupero del momento esatto dell'interruzione
        interrupt_path = os.path.join(SAVE_DIR, "controlnet_interrupted")
        # Salva la rete bypassando i controlli di epoca
        controlnet.save_pretrained(interrupt_path)                       
        
    # Costrutto Python che si esegue SEMPRE e COMUNQUE, sia in caso di interruzione, fine loop naturale o crash
    finally:
        # Verifica se è trascorso abbastanza tempo nel Training Loop da aver generato almeno un valore di Errore Medio per l'epoca
        if len(epoch_losses) > 0:
            # Definisce il nome finale del grafico
            graph_path = os.path.join(SAVE_DIR, "osteogen_training_loss.png")
            # Invia la lista popolata e la directory in pasto alla funzione creata ad inizio pagina
            plot_loss(epoch_losses, graph_path)                          

# Condizione Standard Python: esegue la funzione main() solo se viene avviata direttamente da Terminale e non importata
if __name__ == "__main__":
    main()