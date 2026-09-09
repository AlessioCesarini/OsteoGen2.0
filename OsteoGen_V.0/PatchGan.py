import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import del tuo modulo personalizzato
from src.dataset import OsteoDataset 

# Previene crash di librerie dinamiche (OpenMP) su ambiente Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ==========================================
# Y.3.1 Setup Architetturale: Baseline U-Net
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # ==========================================
        # ENCODER (Discesa spaziale, estrazione feature)
        # ==========================================
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        # Cappiamo i canali a 512 per evitare un'esplosione eccessiva dei parametri
        self.enc5 = self.conv_block(512, 512)
        self.enc6 = self.conv_block(512, 512)

        # Fermando l'espansione a 512 canali, costringiamo la rete ad andare in "profondità" spaziale (fino a vedere la postura globale in 8x8 pixel) senza far esplodere il numero di pesi matematici, mantenendola agile e forzandola a estrarre regole biologiche generali invece di memorizzare.


        #Ad ogni Max Pooling, la rete fa uno "zoom out". Questo permette ai blocchi più profondi di "vedere" l'intero scheletro nella sua interezza e capirne le macro-strutture invece di concentrarsi su un singolo dente.
        self.pool = nn.MaxPool2d(2) #Prende un blocco 2x2 di pixel e trattiene solo il valore massimo, dimezzando di fatto l'altezza e la larghezza del tensore.
        #ESEMPIO: Ho trovato la feature 'Bordo del Femore'. Non mi importa se si trovava in alto a sinistra o in basso a destra nel mio quadratino 2X2. L'ho trovato, e lo passo avanti

        
        # ==========================================
        # BOTTLENECK (Spazio Latente 8x8)
        # ==========================================
        self.bottleneck = self.conv_block(512, 512)

        
        # ==========================================
        # DECODER (Risalita e sintesi fenotipica)
        # ==========================================

        #upconv = L'Espansione Spaziale (Lo "Zoom"); dec = La Miscelazione Semantica (Il "Pittore")
        # dec6 riceve 512 (dall'upconv) + 512 (skip enc6) = 1024 canali totali in ingresso
        # upconv specchia l'enc corrispondente. cat = somma perfetta. dec dimezza per il livello successivo.
        self.upconv6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec6 = self.conv_block(1024, 512) 
        
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = self.conv_block(1024, 512)

        # La simmetria è necessaria affinché i gradienti fluiscano in modo bilanciato durante la backpropagation sulla tua RTX 5080, senza trovare colli di bottiglia asimmetrici.
        
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 256) # 512 (upconv4) + 512 (enc4) = 1024 -> comprime a 256
        
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 128)  # 256 (upconv3) + 256 (enc3) = 512 -> comprime a 128
        
        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 64)   # 128 (upconv2) + 128 (enc2) = 256 -> comprime a 64
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)   # 64 (upconv1) + 64 (enc1) = 128 -> comprime a 64
        
        # Output layer finale: riduce i 64 canali ai 3 canonici RGB
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- DISCESA ---
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))
        
        # --- BOTTLENECK ---
        b = self.bottleneck(self.pool(e6))
        
        # --- RISALITA E CONCATENAZIONE ---
        d6 = self.upconv6(b)
        d6 = torch.cat([d6, e6], dim=1)
        d6 = self.dec6(d6)
        
        d5 = self.upconv5(d6)
        d5 = torch.cat([d5, e5], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.upconv4(d5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # --- OUTPUT ---
        out = self.final_conv(d1)
        return self.tanh(out)


    # ==========================================
# Y.3.2 Setup Architetturale: Discriminatore PatchGAN
# ==========================================
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        
        # Il PatchGAN standard di Pix2Pix usa kernel 4x4.
        # Riceve 6 canali in input: 3 dello scheletro + 3 dell'animale (vero o generato)
        
        def discriminator_block(in_filters, out_filters, normalization=True, stride=2):
            """Ritorna i layer di ciascun blocco del discriminatore"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)] #l'immagine viene dimezzata spazialmente ad ogni passaggio. È un "riassunto visivo" sempre più ristretto.
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters)) #Aggiungiamo la Batch Normalization solo se richiesta.
            # LeakyReLU con pendenza 0.2 è essenziale per evitare la morte dei neuroni nel training avversario
            layers.append(nn.LeakyReLU(0.2, inplace=True)) #Il valore 0.2 permette a una piccola percentuale di numeri negativi di passare comunque. In una GAN, se il discriminatore diventa troppo sicuro di sé e blocca tutti i gradienti negativi (fenomeno del Dying ReLU), il generatore smette di imparare.
            return layers

        self.model = nn.Sequential( #Impacchettiamo in ordine sequenziale i blocchi che andiamo a generare.
            # Livello 1: 6 canali -> 64. Niente BatchNorm sul primo strato per non sporcare i colori puri.
            *discriminator_block(6, 64, normalization=False), #Il primo strato. Prende 6 canali (3 dello scheletro + 3 dell'animale) e ne restituisce 64. Nota fondamentale: la normalizzazione è disattivata (False). Se normalizzassimo subito il primo livello, distruggeremmo i colori originali (RGB) prima ancora di valutarli.
            # Livello 2: 64 -> 128
            *discriminator_block(64, 128), #Scendiamo in profondità. I canali raddoppiano, la risoluzione spaziale si dimezza.
            # Livello 3: 128 -> 256
            *discriminator_block(128, 256), #Qui c'è un trucco architetturale. Usiamo stride=1 invece di 2. Questo impedisce all'immagine di rimpicciolirsi ulteriormente, mantenendo una griglia utile per valutare le patch locali senza schiacciarle in un singolo pixel.
            # Livello 4: 256 -> 512. Riduciamo lo stride a 1 per preparare l'uscita mantenendo risoluzione spaziale
            *discriminator_block(256, 512, stride=1), #L'ultimo strato. Prende le 512 feature map e le schiaccia su 1 singolo canale. Non c'è attivazione finale (niente Sigmoide). Esce una nuda matrice di numeri (Logits) in cui ogni "pixel" della matrice rappresenta il voto (vero o falso) di una macro-area 70x70 dell'immagine originale.
            
            # Output Layer finale
            # Comprime i 512 canali semantici in 1 singolo canale spaziale.
            # Nessuna funzione di attivazione finale (es. Sigmoid) perché useremo nn.BCEWithLogitsLoss
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, skeleton, animal):
        # Concatena lo scheletro in input e l'animale bersaglio lungo l'asse dei canali
        img_input = torch.cat([skeleton, animal], dim=1) #Il discriminatore non guarda mai un'immagine da sola. Deve giudicare la "coppia". Mettiamo letteralmente l'immagine dell'animale sopra quella dello scheletro per creare un super-tensore a 6 strati. Solo così può capire se la pelliccia segue le ossa sottostanti.
        return self.model(img_input) #Spariamo il super-tensore nella rete e restituiamo la matrice di giudizio.



# ==========================================
# Y.3.3 Loop di Addestramento Pix2Pix (GAN) Intelligente
# ==========================================
def train_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs("PatchGan_results", exist_ok=True)

    print("Caricamento dataset in corso...")
    dataset = OsteoDataset(data_dir="data/processed")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Dataset caricato: {len(dataset)} coppie trovate.")

    print("Inizializzazione delle reti sulla GPU...")
    generator = SimpleUNet().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    criterion_GAN = nn.BCEWithLogitsLoss() # È la Binary Cross Entropy. Serve al Discriminatore per giudicare se un'immagine è "Vera" (Target=1) o "Falsa" (Target=0). La dicitura WithLogits significa che la funzione applica automaticamente la formula matematica Sigmoide al suo interno: questo evita errori di arrotondamento e rende l'addestramento infinitamente più stabile rispetto all'uso di una Sigmoide esplicita nell'ultimo layer del PatchGAN.
    criterion_L1 = nn.L1Loss() # Calcola la differenza assoluta pixel per pixel tra l'animale generato e quello reale.
    lambda_L1 = 100 # È il moltiplicatore di importanza. Se usassimo solo la loss GAN, il Generatore creerebbe texture iper-realistiche, ma potrebbe ignorare la postura delle ossa (es. inventando una coda dove non c'è) pur di ingannare il Discriminatore. Moltiplicare la L1 per 100 è il "golden standard" della letteratura Pix2Pix: dice alla rete "Essere fotorealistico è importante, ma rispettare millimetricamente la forma e i colori della foto reale è 100 volte più importante".

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) #LR 0.0002 e betas (0.5, 0.999) sono i valori standard della letteratura Pix2Pix. Non cambiare nulla, altrimenti la GAN collassa.
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


    # ==========================================
    # STRUTTURE DATI PER EARLY STOPPING E GRAFICI
    # ==========================================
    num_epochs = 2000 # Puoi metterne anche 3000, ci penserà l'Early Stop a fermarlo
    patience = 500    # Tolleranza altissima per le oscillazioni fisiologiche della GAN
    best_L1_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    storico_loss_G_GAN = []
    storico_loss_G_L1 = []
    storico_loss_D = []

    
    print("Inizio dell'addestramento Avversario... (Premi Ctrl+C per interrompere in sicurezza)")
    
    try:
        for epoch in range(num_epochs):
            generator.train()
            discriminator.train()

            epoch_loss_G_GAN = 0.0
            epoch_loss_G_L1 = 0.0
            epoch_loss_D = 0.0

            loop_batch = tqdm(dataloader, desc=f"Epoca [{epoch+1}/{num_epochs}]")
            
            for batch_idx, (ossa, animali_veri) in enumerate(loop_batch):
                
                ossa = ossa.to(device)
                animali_veri = animali_veri.to(device)
                
                # --- FASE 1: DISCRIMINATORE ---
                optimizer_D.zero_grad() #Pulisce la memoria dei gradienti del batch precedente per non sommarli.
                animali_finti = generator(ossa) #Il Generatore prova a disegnare il fenotipo.
                
                pred_real = discriminator(ossa, animali_veri)
                target_real = torch.ones_like(pred_real).to(device) #Creiamo una matrice di $1$ (Vero).
                loss_D_real = criterion_GAN(pred_real, target_real) #Calcoliamo quanto $D$ è stato bravo a indovinare che la foto era vera.

                pred_fake = discriminator(ossa, animali_finti.detach()) #$D$ osserva la coppia falsa. Usare .detach() taglia matematicamente il collegamento con il Generatore. Impedisce a PyTorch di calcolare i gradienti a ritroso fino a $G$, risparmiando tempo e prevenendo errori di sovrascrittura, poiché in questa fase stiamo addestrando solo $D$.
                target_fake = torch.zeros_like(pred_fake).to(device) #Creiamo una matrice di $0$ (Falso).
                loss_D_fake = criterion_GAN(pred_fake, target_fake) #Calcoliamo quanto $D$ è stato bravo a sgamare il falso.

                loss_D = (loss_D_real + loss_D_fake) * 0.5 #Mediamo gli errori. $D$ impara al 50% dalle foto vere e al 50% da quelle finte.
                loss_D.backward()
                optimizer_D.step()

                # --- FASE 2: GENERATORE ---
                optimizer_G.zero_grad()
                pred_fake_for_G = discriminator(ossa, animali_finti) #$D$ guarda di nuovo l'animale finto. Questa volta senza .detach(), perché ora l'errore deve scorrere all'indietro per aggiornare i pesi di $G$.
                
                loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real) # Il trucco psicologico della GAN. Noterai che stiamo confrontando la predizione falsa con target_real (matrice di $1$). Il Generatore viene punito matematicamente se $D$ scopre che l'immagine è falsa; il suo scopo è spingere l'output di $D$ verso l' $1$.
                loss_G_L1_pure = criterion_L1(animali_finti, animali_veri) #: Calcola la distanza matematica assoluta (MAE) pixel per pixel tra l'immagine generata e la fotografia reale. L'ho definita "pure" (pura) perché rappresenta l'errore nudo, prima di subire la moltiplicazione per 100 (lambda_L1). Questo costringe la U-Net a non inventare code o orecchie extra che non esistono nello scheletro.
                loss_G_L1_scaled = loss_G_L1_pure * lambda_L1 # Calcoliamo la distanza pixel-per-pixel tra l'animale finto e la foto reale, moltiplicandola per 100. Questo vincola il Generatore a non stravolgere la posa dello scheletro pur di ingannare il Discriminatore.
                
                loss_G = loss_G_GAN + loss_G_L1_scaled #È la fusione dei due obiettivi. Il Generatore deve soddisfare due maestri contemporaneamente: deve ingannare il critico per il realismo (loss_G_GAN) e deve combaciare spazialmente con il target (loss_G_L1_scaled). Questa somma rappresenta il suo "voto finale".
                loss_G.backward()
                optimizer_G.step()

                epoch_loss_G_GAN += loss_G_GAN.item()
                epoch_loss_G_L1 += loss_G_L1_pure.item()
                epoch_loss_D += loss_D.item()

                loop_batch.set_postfix(Loss_D=loss_D.item(), Loss_G_L1=loss_G_L1_pure.item())

            # ==========================================
            # CALCOLO MEDIE EPOCHE E EARLY STOPPING
            # ==========================================
           # ==========================================
            # CALCOLO MEDIE EPOCHE E EARLY STOPPING
            # ==========================================
            num_batches = len(dataloader)
            avg_G_L1 = epoch_loss_G_L1 / num_batches
            avg_G_GAN = epoch_loss_G_GAN / num_batches
            avg_D = epoch_loss_D / num_batches

            storico_loss_G_GAN.append(avg_G_GAN)
            storico_loss_G_L1.append(avg_G_L1)
            storico_loss_D.append(avg_D)

            # STAMPA IL RIASSUNTO DELL'EPOCA
            print(f"-> Fine Epoca {epoch+1} | Media Loss D: {avg_D:.4f} | Media Loss G (GAN): {avg_G_GAN:.4f} | Media L1 (Ricostruzione): {avg_G_L1:.4f}")

            # LOGICA DI SALVATAGGIO CON AVVISO
            if avg_G_L1 < best_L1_loss:
                best_L1_loss = avg_G_L1
                best_epoch = epoch + 1
                epochs_no_improve = 0
                
                print(f"⭐ MIGLIORAMENTO! La Loss L1 è scesa a {best_L1_loss:.4f}. Salvataggio pesi sulla RTX 5080...")
                
                torch.save(generator.state_dict(), "PatchGan_results/best_generator.pth")
                torch.save(discriminator.state_dict(), "PatchGan_results/best_discriminator.pth")
            else:
                epochs_no_improve += 1
                print(f"   (Nessun miglioramento della L1 da {epochs_no_improve} epoche)")

            if (epoch + 1) % 20 == 0:
                generator.eval()
                with torch.no_grad(): # Generazione Test Visivi
                    for test_ossa, test_animali in dataloader:
                        test_ossa = test_ossa.to(device)
                        pred_animali = generator(test_ossa)
                        vis_ossa = (test_ossa[:4] * 0.5) + 0.5
                        vis_pred = (pred_animali[:4] * 0.5) + 0.5
                        vis_veri = (test_animali[:4].to(device) * 0.5) + 0.5
                        comparison = torch.cat([vis_ossa, vis_pred, vis_veri], dim=0)
                        save_image(comparison, f"PatchGan_results/epoca_{epoch+1}.png", nrow=4)
                        break 

            if epochs_no_improve >= patience:
                print(f"\n[Early Stop] Apprendimento strutturale stagnante da {patience} epoche.")
                break

    except KeyboardInterrupt:
        # Questo blocco scatta appena premi Ctrl+C
        print("\n\n[!] ATTENZIONE: Addestramento interrotto manualmente (Ctrl+C).")
        print(f"Salvataggio del modello corrente all'epoca {epoch+1} in corso...")
        torch.save(generator.state_dict(), f"PatchGan_results/generator_interrotto_ep{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"PatchGan_results/discriminator_interrotto_ep{epoch+1}.pth")
        
    finally:
        # Questo blocco viene eseguito SEMPRE, sia alla fine naturale, sia post-interruzione
        print(f"\nGenerazione dei grafici finali... Il modello migliore è all'epoca {best_epoch} (L1: {best_L1_loss:.4f})")
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(storico_loss_G_GAN, label='Loss Generatore (Avversaria)', color='blue', alpha=0.7)
        plt.plot(storico_loss_D, label='Loss Discriminatore', color='red', alpha=0.7)
        if best_epoch > 0:
            plt.axvline(x=best_epoch-1, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title('Dinamica Avversaria (GAN Loss)')
        plt.xlabel('Epoche')
        plt.ylabel('BCE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(storico_loss_G_L1, label='Loss L1 (Ricostruzione)', color='purple')
        if best_epoch > 0:
            plt.plot(best_epoch-1, best_L1_loss, marker='*', markersize=15, color='red', label='Miglior Ricostruzione')
        plt.title('Aderenza Strutturale e Colore (L1 Loss)')
        plt.xlabel('Epoche')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("PatchGan_results/grafico_loss_complessivo.png")
        print("Grafici salvati. Training terminato in sicurezza.")

if __name__ == "__main__":
    train_model()