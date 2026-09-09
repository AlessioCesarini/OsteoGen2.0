import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from src.dataset import OsteoDataset  # Importo il file dataset.py
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ==========================================
# Y.3.1 Setup Architetturale: Baseline U-Net
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Encoder (Comprime l'immagine ed estrae feature geometriche)
        self.enc1 = self.conv_block(3, 64) #Da RGB a Feature di Bassissimo Livello
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256) #Feature di altissimo Livello con 512 canali che rappresentano concetti astratti

        #Ad ogni Max Pooling, la rete fa uno "zoom out". Questo permette ai blocchi più profondi di "vedere" l'intero scheletro nella sua interezza e capirne le macro-strutture invece di concentrarsi su un singolo dente.
        self.pool = nn.MaxPool2d(2) #Prende un blocco 2x2 di pixel e trattiene solo il valore massimo, dimezzando di fatto l'altezza e la larghezza del tensore.
        #ESEMPIO: Ho trovato la feature 'Bordo del Femore'. Non mi importa se si trovava in alto a sinistra o in basso a destra nel mio quadratino 2X2. L'ho trovato, e lo passo avanti

        
        # Bottleneck (Spazio Latente rudimentale)
        self.bottleneck = self.conv_block(256, 512) 
        
        # Decoder (Decomprime e ricostruisce l'immagine)
        #Da Altissimo Livello si inizia ad andare al microscopico
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) #Lavora sulla Geometria. Il suo unico scopo è ingrandire la "tela" (raddoppiare altezza e larghezza).
        self.dec3 = self.conv_block(512, 256) # 512 perché concateniamo le skip connection
        #Prende i 512 canali appena incollati e li fonde. Usa i suoi pesi convoluzionali per rimescolare le informazioni profonde con i bordi precisi della skip connection, sintetizzando un nuovo output pulito a 256 canali. È il pittore che prende la proiezione grande e sfocata sul muro, usa i bordi netti della skip connection come ricalco, e dipinge i dettagli fini, restituendo un'immagine nitida e coerente.
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh() # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)). Questa funzione ha preso le foto target (che avevano pixel da 0 a 255) e le ha scalate tra -1.0 e 1.0. Coerenza Matematica


    #METODO 1: Miscela i canali in ingresso per cercare feature biologiche e ne sputa fuori un numero diverso (es. entra con 512, esce con 256)
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # Da dimensione in a dimensione out. Kernel 3x3 con padding 1 (pixel nero sui bordi) mantiene le dimensioni spaziali.
            nn.BatchNorm2d(out_channels), # Costringe la rete a mantenere vivi i segnali
            nn.ReLU(inplace=True), #  Funzione di Attivazione Non-Lineare: Applica la formula $f(x) = \max(0, x)$
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), #Con la prima riga la rete capisce i dettagli piccolissimi; con questa seconda riga combina i dettagli piccolissimi per formare forme un po' più grandi, prima di passare al blocco successivo.
            nn.BatchNorm2d(out_channels), # Costringe la rete a mantenere vivi i segnali
            nn.ReLU(inplace=True)
        )

    #METODO 2: il "piano di volo" che dice al tensore (l'immagine) esattamente come muoversi attraverso i blocchi
    def forward(self, x):
        # Discesa: distruggere le coordinate spaziali (i pixel) per estrarre il "senso" (le feature) dell'immagine
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1)) #self.pool(e1) = dimezza altezza e larghezza di e1, ma mantiene i 64 canali. Quindi e2 ha dimensione dimezzata rispetto a e1, ma con 128 canali. Estrae poi forme intermedie (feature di medio livello). Salvi questo tensore in e2
        e3 = self.enc3(self.pool(e2))
        
        # Spazio Latente
        b = self.bottleneck(self.pool(e3)) #Il tensore b non è più l'immagine di un osso, è un insieme di numeri che dicono: "Ho individuato la geometria necessaria a sostenere un quadrupede di grossa stazza".
        
        # Risalita + Concatenazione (Skip Connections)
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)  # "Prendi il tensore appena ingrandito d3 e incollaci letteralmente accanto il tensore e3 che avevamo parcheggiato". dim=1 significa che li stai affiancando lungo l'asse dei canali. Incolli 256 canali geometrici perfetti a 256 canali semantici sfocati.
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)  
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1) 
        d1 = self.dec1(d1)
        
        # Output
        out = self.final_conv(d1) #Una piccola convoluzione spaziale (kernel 1x1) esegue una somma pesata delle 64 feature e le schiaccia matematicamente su 3 singoli canali. Questi corrispondono a Rosso, Verde e Blu. La rete ha finalmente dipinto l'animale.
        return self.tanh(out) #Tangente Iperbolica costringe forzatamente tutti i pixel ad assumere un valore compreso tra -1.0 e 1.0

def train_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #CUDA (Compute Unified Device Architecture) è la piattaforma di calcolo parallelo inventata da NVIDIA. Ciò che ti premette di usare la GPU invidia invece che la CPU per addestrare la rete neurale. Se non hai una GPU, userai la CPU.
    os.makedirs("training_dir_Image_to_Image", exist_ok=True) #Crei la Dir dei risultati. Se esiste già, lanci un errore.

    print("Caricamento dataset in corso...")
    dataset = OsteoDataset(data_dir="OsteoGen/data/processed") #Carichi il Dataset. La classe OsteoDataset è definita in dataset.py e si occupa di leggere le immagini dalla cartella data/processed, applicare eventuali trasformazioni e restituire coppie di immagini (input, target).

    # Crea il distributore di batch. Quanto usi la GPU (Alessio ha 32 Giga grazie alla 5080)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Dataset caricato: {len(dataset)} coppie trovate.")

    print("Inizializzazione della rete sulla GPU...")
    model = SimpleUNet().to(device) # Sposta l'intera architettura nella RTX 5080

    # La Funzione di Costo (Errore Assoluto Medio pixel per pixel)
    criterion = nn.L1Loss() #Calcola semplicemente la distanza assoluta tra il colore del pixel generato e quello reale. Non elevando al quadrato, un errore grande viene punito in modo lineare, non esponenziale.
    # Effetto Visivo (Il vantaggio): Sopportando meglio le incertezze, la rete è incoraggiata a "prendere una decisione" e generare bordi netti, texture ruvide e tratti fenotipici definiti.


    #ADAM :Adam calcola in modo adattivo la velocità. Tiene in memoria le pendenze passate (il momento): se scende dritto e sicuro, accelera; se il terreno è accidentato e i gradienti oscillano, frena bruscamente. Calcola una velocità di aggiornamento diversa per ogni singolo parametro della rete. È lo standard indiscusso per le reti generative perché è estremamente rapido a convergere.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model.parameters() : PyTorch entra nella tua classe SimpleUNet ed estrae dinamicamente la lista di tutti i tensori addestrabili (i milioni di pesi all'interno delle Conv2d e ConvTranspose2d). Passando questa lista ad Adam, gli consegni letteralmente i comandi della rete: "Queste sono le variabili matematiche che devi aggiornare alla fine di ogni calcolo dell'errore".
    # lr = 0.0002 : Dalla pubblicazione del celebre paper Pix2Pix, il valore $2 \times 10^{-4}$ abbinato ad Adam è diventato il Golden Standard della letteratura scientifica per l'addestramento di architetture U-Net. È il compromesso perfetto tra stabilità del training e velocità di apprendimento


    num_epochs = 1000 #La GPU con batch Size 16 deve fare 1000 epoche per vedere la loss scendere a valori accettabili. Se vuoi accelerare, puoi ridurre il numero di epoche, ma la qualità delle immagini generate peggiorerà.
    print("Inizio dell'addestramento...")
    storico_loss = []

    patience = 200 # Tolleranza: numero di epoche senza miglioramenti prima di fermarsi
    best_loss = float('inf')
    epochs_no_improve = 0
    # Il Ciclo Globale (Epoche)
    for epoch in range(num_epochs):
        model.train() # Comunica alla rete che stiamo addestrando (fondamentale per le BatchNorm)

        loop_batch = tqdm(dataloader, desc=f"Epoca [{epoch+1}/{num_epochs}]")
        # FASE 5: Il Ciclo Locale (Batch)
        # enumerate() ci permette di avere sia il numero del batch (batch_idx) che i dati effettivi
        for batch_idx, (ossa, animali_veri) in enumerate(loop_batch):
            
            # 5.1 Trasferimento in VRAM (spostiamo i dati sulla tua RTX 5080)
            ossa = ossa.to(device)
            animali_veri = animali_veri.to(device)
            
            # ========================================================
            # I 5 STEP DELLA BACKPROPAGATION
            # ========================================================
            # 1. Azzera i gradienti (fondamentale per non sommarli ai precedenti)
            optimizer.zero_grad() #Senza questa riga, PyTorch accumula matematicamente i gradienti. L'errore del batch 2 verrebbe sommato all'errore del batch 1, mandando l'algoritmo di ottimizzazione fuori controllo in pochissimi istanti.
            
            # 2. Forward pass: la U-Net tenta di generare l'animale dallo scheletro
            animali_finti = model(ossa) # Innesca automaticamente il metodo forward(x). Le immagini 512x512 attraversano fisicamente la GPU scendendo nell'encoder e risalendo nel decoder.
            #SPIEGAZIONE IA: Se tu richiamassi esplicitamente il forward, Python eseguirebbe matematicamente la discesa e la risalita, restituendoti l'immagine finta. Tuttavia, avresti bypassato l'orchestratore di PyTorch. La rete non registrerebbe lo storico delle operazioni nella VRAM della tua RTX 5080. Di conseguenza, arrivato allo step 4 (loss.backward()), il codice crasherebbe fatalmente perché il sistema non troverebbe il percorso per propagare all'indietro gli errori.
            # : Quando scrivi model(ossa), non stai ignorando il tuo forward, lo stai facendo eseguire sotto scorta. Il sistema di PyTorch accende il registratore matematico (il grafo computazionale), prepara i registri di memoria sulla tua RTX 5080 e, un istante dopo, richiama ed esegue esattamente il forward che hai scritto tu.

            
            # 3. Calcolo del costo: distanza assoluta tra la previsione e il target reale
            loss = criterion(animali_finti, animali_veri)
            
            # 4. Backward pass (Backpropagation): calcola le derivate parziali
            loss.backward() # ripercorrendo la rete all'indietro per assegnare a ciascuno dei milioni di pesi la sua percentuale di "colpa" per l'errore finale
            
            # 5. Ottimizzazione: Adam aggiorna i pesi usando i gradienti calcolati
            optimizer.step() #Adam modifica fisicamente i pesi nei tensori della RTX 5080, usando il learning rate (0.0002) per assicurarsi che i cambiamenti siano minuscoli e controllati
            # ========================================================

            loop_batch.set_postfix(loss=loss.item())
            # Print di debug: ci stampa a schermo la loss ogni 10 batch per vedere se scende
            if batch_idx % 10 == 0:
                print(f"Epoca [{epoch+1}/{num_epochs}] - Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}")

        storico_loss.append(loss.item()) #Salviamo la loss finale di ogni epoca per poi fare il grafico di apprendimento
        # Controllo Early Stopping e Salvataggio Pesi
        current_loss = loss.item()
        if current_loss < best_loss - 0.001:
            best_loss = current_loss
            epochs_no_improve = 0
            
            # Salva i pesi matematici (i tensori) della rete migliore finora
            torch.save(model.state_dict(), "training_dir_Image_to_Image/best_unet_baseline.pth")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\n[Early Stop] Addestramento interrotto all'epoca {epoch+1}.")
            print(f"La rete ha smesso di apprendere da {patience} epoche.")
            break

        # Scegliamo di salvare l'immagine ogni 10 epoche per evitare troppi file
        if (epoch + 1) % 10 == 0:
            model.eval() # Mette il modello in modalità valutazione (disattiva il dropout/batchnorm dinamico)
            with torch.no_grad(): # Spegne il calcolo dei gradienti (risparmia memoria VRAM)
                
                # Prendiamo un singolo batch di test (il primo che capita dal dataloader)
                for test_ossa, test_animali in dataloader:
                    test_ossa = test_ossa.to(device)
                    test_animali = test_animali.to(device)
                    
                    # La rete genera l'animale finto
                    pred_animali = model(test_ossa)
                    
                    # Denormalizzazione: passiamo l'intervallo da [-1, 1] a [0, 1]
                    # Questo perché la funzione save_image si aspetta pixel standard da 0 a 1
                    vis_ossa = (test_ossa[:4] * 0.5) + 0.5
                    vis_pred = (pred_animali[:4] * 0.5) + 0.5
                    vis_veri = (test_animali[:4] * 0.5) + 0.5
                    
                    # Uniamo le immagini in un unico blocco orizzontale/verticale
                    # Ordine: [Scheletro input, Animale generato dalla rete, Animale reale target]
                    comparison = torch.cat([vis_ossa, vis_pred, vis_veri], dim=0)
                    
                    # Salviamo fisicamente l'immagine nella cartella dell'ablation study
                    save_image(comparison, f"training_dir_Image_to_Image/epoca_{epoch+1}.png", nrow=4)
                    
                    break # Ci basta un solo batch di esempio per epoca
    plt.figure(figsize=(10, 5))
    plt.plot(storico_loss, label="L1 Loss (Baseline U-Net)", color="red")
    plt.title("Curva di Apprendimento - Ablation Study")
    plt.xlabel("Epoche")
    plt.ylabel("Errore Medio Assoluto (MAE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_dir_Image_to_Image/grafico_loss_finale.png")
    print("Addestramento concluso e grafico salvato.")               

if __name__ == "__main__":
    train_model()