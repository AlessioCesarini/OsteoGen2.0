Capitolo Y: Roadmap di Sviluppo e Fasi Operative del Progetto

Y.1 Introduzione e Ciclo di Sviluppo
La realizzazione del progetto segue un approccio ingegneristico iterativo (Agile/Incremental), strutturato in fasi sequenziali. L'obiettivo è garantire la validità scientifica dell'esperimento, dimostrando empiricamente i limiti degli approcci classici prima di validare l'architettura avanzata. Lo sviluppo è suddiviso in sei fasi operative principali.

---

Y.2 Fase 1: Data Engineering e Preparazione del Dataset 
Questa fase è dedicata alla costruzione dell'infrastruttura dati, prerequisito per l'addestramento dei modelli, con particolare attenzione alla disomogeneità delle fonti visive.
* Y.2.1 Pulizia e Scontorno: Utilizzo di script automatizzati (es. librerie basate su Segment Anything Model o rembg) per estrarre il foreground (animale/scheletro) e applicare un background neutro, eliminando le correlazioni spurie legate al paesaggio.
* Y.2.2 Normalizzazione Spaziale e Geometrica: Dato che i campioni originali presentano risoluzioni e proporzioni eterogenee, viene implementata una pipeline di standardizzazione geometrica atta a preservare la coerenza biomeccanica. Questa si divide in tre step matematici eseguiti in pre-processing:
    - Auto-Cropping (Bounding Box): Algoritmi di sogliatura individuano la regione di interesse (ROI), scartando lo sfondo in eccesso per standardizzare la scala visiva tra X e Y.
    - Preservazione dell'Aspect Ratio (Padding): Per soddisfare i vincoli topologici dei tensori quadrati richiesti dai modelli generativi (es. 512x512) senza introdurre deformazioni anatomiche, viene applicato un padding simmetrico con valori di background sul lato più corto dell'immagine.
    - Resize e Tensorizzazione: L'immagine, ora quadrata e non deformata, viene ridimensionata alla risoluzione target.
* Y.2.3 Strutturazione Dati: Creazione del repository fisico contenente unicamente le 70 coppie originarie normalizzate (Input X e Target Y).
* Y.2.4 Data Loader Custom (PyTorch): Sviluppo del modulo software per il caricamento in memoria. In questa fase viene implementata la pipeline di Online Stochastic Data Augmentation (inserimento di rumore stocastico, rotazioni minime e deformazioni elastiche) applicata on-the-fly.

---

Y.3 Fase 2: Sviluppo della Baseline e Analisi dei Limiti (Ablation Study)
Prima di implementare modelli complessi, viene addestrata un'architettura di base per isolare e documentare il problema del disallineamento spaziale.
* Y.3.1 Setup Architetturale: Implementazione di una rete Image-to-Image convoluzionale standard (es. U-Net o architettura di tipo Pix2Pix).
* Y.3.2 Funzione di Costo Base: Addestramento condotto esclusivamente con funzioni di costo pixel-wise (L1 Loss o Mean Squared Error).
* Y.3.3 Analisi dell'Errore: Documentazione dei risultati (effetto ghosting, collasso delle texture e sfocatura). Questo step funge da "Ablation Study", fornendo la giustificazione scientifica per il passaggio a funzioni di costo di tipo percettivo e architetture basate sull'Attenzione.

---

Y.4 Fase 3: Implementazione dell'Architettura Avanzata (ControlNet)
Il nucleo ingegneristico del progetto, focalizzato sulla risoluzione del disallineamento e della scarsità dei dati.
* Y.4.1 Integrazione Foundation Model: Setup di un modello di Latent Diffusion pre-addestrato (es. architetture supportate dalla libreria diffusers) che funga da base di conoscenza fenotipica.
* Y.4.2 Configurazione ControlNet: Inserimento del modulo di Cross-Attention per vincolare la generazione morfologica allo scheletro di input, trattato come guida semantica e non come stampo spaziale rigido.
* Y.4.3 Custom Loss Function: Sostituzione della L1 Loss con la Perceptual Loss (LPIPS), calcolata sullo spazio latente di una rete VGG16, per focalizzare l'errore sulla coerenza delle texture e dei concetti anatomici anziché sulla sovrapposizione geometrica esatta.

---

Y.5 Fase 4: Addestramento, Ottimizzazione e Tracking
Avvio della fase di calcolo intensivo, caratterizzata dal monitoraggio matematico del modello.
* Y.5.1 Experiment Tracking: Integrazione di piattaforme di logging (es. Weights & Biases o TensorBoard) per il tracciamento in tempo reale della loss globale e delle singole componenti (Loss di diffusione, Loss percettiva).
* Y.5.2 Hyperparameter Tuning: Ottimizzazione iterativa del learning rate, dei pesi delle loss combinate (i parametri lambda) e della probabilità di iniezione del rumore stocastico.

---

Y.6 Fase 5: Validazione Quantitativa su Specie Moderne (Test Set)
Prima di applicare il modello in ambito paleontologico, è necessario validarne la capacità di generalizzazione su un set di test (Out-of-Distribution moderno).
* Y.6.1 Zero-Shot Inference: Test del modello su scheletri di animali moderni deliberatamente esclusi dal dataset di addestramento.
* Y.6.2 Metriche di Valutazione: Calcolo di metriche generative standard, come la FID (Fréchet Inception Distance), per quantificare oggettivamente la discrepanza tra la distribuzione delle immagini generate e quelle reali.

---

Y.7 Fase 6: Inferenza Paleontologica e Analisi Morfologica (Esperimento Finale)
La fase culminante della ricerca, in cui il modello viene testato sul target finale.
* Y.7.1 Input Fossile: Somministrazione dello scheletro del Tyrannosaurus Rex alla rete ottimizzata.
* Y.7.2 Analisi Qualitativa: Valutazione dei risultati generati, discutendo come l'architettura abbia interpolato lo spazio latente per distribuire masse muscolari, texture e rivestimenti cutanei coerenti con l'evoluzione biologica, basandosi esclusivamente sui vincoli biomeccanici estratti dal grafo osteologico in input.