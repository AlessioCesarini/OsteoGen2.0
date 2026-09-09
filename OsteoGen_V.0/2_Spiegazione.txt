Capitolo X: Strategie di Ottimizzazione del Dataset e Data Augmentation in Scenari a Bassa Dimensionalità

X.1 Introduzione e Definizione del Problema
Nel contesto della generazione di immagini Image-to-Image finalizzata alla ricostruzione fenotipica a partire da dati osteologici, la dimensione del dataset rappresenta il principale collo di bottiglia. Il dataset originale è composto da sole 70 coppie di immagini (scheletro di input X e fenotipo target Y). 

L'approccio iniziale prevedeva una Data Augmentation Offline e deterministica, espandendo staticamente il dataset tramite l'applicazione di varianti fisse. Questa sezione analizza le criticità matematiche e geometriche di tale approccio e giustifica la transizione verso una pipeline di Augmentation Stocastica Online, unitamente alla gestione del disallineamento spaziale insito nei dati.

---

X.2 Criticità dell'Approccio Offline (Deterministico)
L'approccio di augmentation statica su file pre-generati presenta problematiche fondamentali che inibiscono la capacità di generalizzazione del modello (fondamentale per il task di inferenza su specie estinte o Out-of-Distribution, come il T-Rex).

X.2.1 Collasso Dimensionale e Overfitting
L'applicazione di trasformazioni fisse porta la cardinalità del dataset a poche centinaia di campioni. Modelli generativi ad alta capacità possiedono milioni di parametri addestrabili. Di fronte a un set discreto e così limitato, la rete minimizzerà la funzione di costo (Loss) eseguendo una memorizzazione mnemonica (overfitting) invece di apprendere il manifold della biologia ossea.

X.2.2 Violazione della Covarianza Spaziale
Nei task Image-to-Image, è un requisito matematico che la mappa spaziale tra input e target rimanga coerente. Applicare rotazioni estreme (es. 45 gradi) in modo rigido introduce artefatti visivi e non introduce reale varianza biologica, sprecando la capacità rappresentativa della rete.

X.2.3 Fondamento Teorico dell'Iniezione di Rumore
L'aggiunta di rumore (es. rumore Gaussiano additivo) rappresenta invece un'intuizione matematicamente solida. Aggiungere rumore ai bordi dello scheletro costringe il modello a comportarsi come un Denoising Autoencoder. La rete è forzata ad ignorare le fluttuazioni ad alta frequenza e ad apprendere rappresentazioni latenti invarianti concentrandosi sulla macro-struttura. Questo garantisce estrema robustezza in inferenza.

---

X.3 Proposta Metodologica: Online Stochastic Data Augmentation
Per risolvere i limiti descritti, si adotta una pipeline dinamica (Online). Le trasformazioni vengono campionate da distribuzioni di probabilità continue e applicate on-the-fly durante il caricamento dei mini-batch per ogni iterazione (epoca) dell'addestramento:
* Rotazioni di piccola entità: Angoli campionati da una distribuzione uniforme limitata (es. tra -10 e +10 gradi) simulano leggere variazioni di posa.
* Trasformazioni Elastiche (Elastic Deformations): Generano vettori di spostamento casuali fluidi sull'immagine, deformando l'osso per simulare specie intermedie e aumentando virtualmente la variabilità anatomica.

---

X.4 Risoluzione del Disallineamento Spaziale tra Input e Target
Un vincolo critico e ineliminabile del dataset in esame è l'impossibilità pratica di garantire una co-registrazione spaziale perfetta (allineamento millimetrico) tra lo scheletro in input (X) e la foto dell'animale reale (Y). Nonostante la posa generale sia coerente (es. entrambi con vista a 3/4), le singole articolazioni presentano inevitabili sfalsamenti geometrici dovuti alle fonti eterogenee delle immagini.

X.4.1 Il Fallimento delle Funzioni di Costo Pixel-wise
In presenza di dati disallineati, l'utilizzo di funzioni di costo classiche basate sulla distanza spaziale esatta (come L1 Loss o Mean Squared Error tipiche di architetture come Pix2Pix) porta al collasso visivo. La rete, nel tentativo di minimizzare l'errore tra un osso posizionato al pixel i e la sua massa muscolare posta al pixel i+k, genera output sfocati calcolando la media spaziale (effetto ghosting).

X.4.2 Soluzione Architetturale: Cross-Attention e Spazi Latenti Percettivi
Per gestire il disallineamento, l'architettura deve superare il mapping spaziale rigido. La soluzione implementata si fonda su due pilastri:
1. Modelli basati sull'Attenzione (ControlNet/Diffusion): A differenza delle convoluzioni rigide (CNN), il meccanismo di Cross-Attention dei Transformer valuta l'immagine globalmente. Lo scheletro disallineato non funge da stampo pixel-perfetto, ma da mappa di "suggerimenti strutturali", permettendo alla rete di traslare flessibilmente le feature semantiche per adattarle al target.
2. Perceptual Loss (LPIPS): La funzione di costo non viene calcolata nello spazio dei pixel, ma nello spazio delle feature di una rete estrattiva pre-addestrata (es. VGG16). L'errore viene minimizzato se i "concetti" (texture del pelo, forma dell'arto) sono generati correttamente nel vicinato spaziale atteso, rendendo il modello intrinsecamente invariante e robusto ai piccoli disallineamenti di posa.

---

X.5 Conclusioni della Metodologia
L'adozione di un approccio Online Stocastico accoppiato a un'architettura basata su Attenzione e Perceptual Loss permette di svincolare l'addestramento dalla necessità di avere dataset massivi e perfettamente allineati. Questo prepara lo spazio latente del modello ad accogliere e interpretare correttamente morfologie out-of-distribution e disallineate, come quelle tipiche dei reperti paleontologici.