import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

# Classe custom che eredita da torch.utils.data.Dataset, necessaria per interfacciarsi con il DataLoader di PyTorch
class OsteoDataset(Dataset):
    def __init__(self, data_dir: str = "C:\\Users\\alexc\\Desktop\\GitHub Projects\\OsteoGen\\OsteoGen_Version1\\data\\processed", transform=None):
        # Utilizziamo pathlib per gestire i percorsi in modo robusto e indipendente dal sistema operativo (Windows/Linux)
        self.data_dir = Path(data_dir)
        
        # Definiamo i percorsi delle cartelle contenenti gli input condizionali (scheletri) e i target (animali)
        self.x_dir = self.data_dir / "input_X" 
        self.y_dir = self.data_dir / "target_Y"
        
        # Estraiamo tutti i nomi dei file .png presenti nella cartella di input e li ordiniamo alfabeticamente
        self.filenames = sorted([f.name for f in self.x_dir.glob("*.png")])
        print(f"DEBUG - File estratti trovati: {len(self.filenames)}")
        
        # Assegniamo le trasformazioni spaziali (se fornite dall'esterno) oppure usiamo quelle di default
        self.transform = transform or self.get_default_transforms()
        
        # Inizializziamo il normalizzatore specifico per l'immagine target (Animale). 
        # Serve a scalare i pixel nel range [-1, 1], un requisito matematico tassativo del VAE di Stable Diffusion
        self.normalize_y = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # Lista che conterrà l'intero dataset pre-caricato in memoria RAM
        self.data = []
        
        print("DEBUG - Inizio pre-caricamento e traduzione in RAM...")
        
        # Spostiamo la mappa di traduzione fuori dal ciclo for: ricrearla 65 volte consuma CPU inutilmente.
        # Questo dizionario agisce da traduttore offline deterministico (nessun rischio di crash API)
        TRANSLATION_MAP = {
            "airone": "heron", "alcaimpenne": "great auk", "aquila": "eagle", "armadillo": "armadillo",
            "axolotl": "axolotl", "balaenicepsrex": "shoebill", "balena": "whale", "bassotto": "dachshund",
            "bradipo": "sloth", "camaleonte": "chameleon", "cane": "dog", "canguro": "kangaroo",
            "casuario": "cassowary", "cavallo": "horse", "celacanto": "coelacanth", "cervo": "deer",
            "cinghiale": "wild boar", "coccodrillo": "crocodile", "colibri": "hummingbird", "delfino": "dolphin",
            "dodo": "dodo", "dragodicommodo": "komodo dragon", "echidna": "echidna", "elefante": "elephant",
            "fagiano": "pheasant", "foca": "seal", "formichiere": "anteater", "gallina": "chicken",
            "gatto": "cat", "giraffa": "giraffe", "gorilla": "gorilla", "gufo": "owl", "iguana": "iguana",
            "ippopotamo": "hippopotamus", "kiwi": "kiwi", "levriero": "greyhound", "lupo": "wolf",
            "mammut": "mammoth", "mucca": "cow", "opossum": "opossum", "ornitorinco": "platypus",
            "pappagallo": "parrot", "pellicano": "pelican", "picchio": "woodpecker", "piccione": "pigeon",
            "pinguino": "penguin", "pipistrello": "bat", "rana": "frog", "razza": "stingray",
            "rinoceronte": "rhinoceros", "rinocerontelanoso": "woolly rhinoceros", "rondine": "swallow",
            "salamandra": "salamander", "scimpanze": "chimpanzee", "serpente": "snake", "squalo": "shark",
            "struzzo": "ostrich", "tacchino": "turkey", "tartaruga": "turtle", "thylacine": "thylacine",
            "topo": "mouse", "tricheco": "walrus", "trota": "trout", "tuatara": "tuatara", "uomo": "human"
        }

        # Iteriamo su ogni file per pre-caricarlo. Satirare la RAM della CPU evita colli di bottiglia all'I/O del disco durante il training
        for filename in self.filenames:
            x_path = self.x_dir / filename
            y_path = self.y_dir / filename
            
            # Controllo di coerenza strutturale: evitiamo crash se manca un'immagine target
            if not y_path.exists():
                continue
                
            # Leggiamo l'immagine direttamente come Tensore PyTorch RGB e normalizziamo i valori da [0, 255] a float [0.0, 1.0]
            tensor_x = read_image(str(x_path), mode=ImageReadMode.RGB).float() / 255.0
            tensor_y = read_image(str(y_path), mode=ImageReadMode.RGB).float() / 255.0
            
            # Estraiamo il nome del file senza estensione e lo convertiamo in minuscolo (es: 'dragodicommodo')
            nome_animale_ita = Path(filename).stem.lower()

            # Mappiamo il nome italiano in inglese. Il secondo argomento funge da meccanismo di fallback sicuro
            nome_animale_eng = TRANSLATION_MAP.get(nome_animale_ita, nome_animale_ita)

            # Generiamo il prompt semantico base con tag di alta qualità visiva per guidare Stable Diffusion
            prompt = f"A full-body three-quarter view photo of a {nome_animale_eng}, featuring highly detailed, photorealistic textures, 8K resolution, studio lighting, and fully isolated against a pure black background."
                
            # Salviamo l'intero campione in un dizionario in RAM (X, Y, prompt testuale e info di debug)
            self.data.append({
                "tensor_x": tensor_x,
                "tensor_y": tensor_y,
                "prompt": prompt,
                "debug_info": f"{nome_animale_ita} -> {nome_animale_eng}"
            })
            
        print("DEBUG - Pre-caricamento completato con successo.")

    # Metodo per isolare la logica della Data Augmentation
    def get_default_transforms(self):
        # Utilizziamo v2.Compose per applicare sequenzialmente le trasformazioni spaziali rigorose (niente elasticità)
        return v2.Compose([
            # Il flip orizzontale preserva la geometria anatomica aggiungendo varianza speculare
            v2.RandomHorizontalFlip(p=0.5),
            # RandomAffine introduce lievi rotazioni e scaling per rendere il modello invariante alla posizione dello scheletro
            v2.RandomApply([
                v2.RandomAffine(degrees=[-10, 10], translate=[0.05, 0.05], scale=[0.95, 1.05], fill=0)
            ], p=0.5)
        ])

    # Ritorna la grandezza totale del dataset; essenziale per il DataLoader per capire quando finisce un'epoca
    def __len__(self):
        return len(self.data)

    # Viene chiamato dal DataLoader ad ogni iterazione per ottenere un singolo campione randomico (batch)
    def __getitem__(self, idx):
        # Estraiamo il campione dalla RAM
        item = self.data[idx]
        tensor_x = item["tensor_x"]
        tensor_y = item["tensor_y"]
        prompt = item["prompt"]
        
        # PROMPT DROPOUT (CLASSIFIER-FREE GUIDANCE)
        # Nel 10% dei casi (soglia empirica standard in architetture Diffusers), forziamo il prompt a essere una stringa vuota.
        # Questo costringe la rete neurale a non affidarsi solo al testo, ma ad imparare a generare immagini
        # guidata esclusivamente dalla geometria dello scheletro. Questo è essenziale per il test "Zero-Shot" (es. sul T-Rex).
        # NOTA: Si esegue qui nel __getitem__ (e non nel pre-caricamento) in modo che sia stocastico ad ogni epoca.
        if random.random() < 0.10:
            prompt = ""
        
        # L'API v2 garantisce che le medesime trasformazioni spaziali (stesso seed) vengano applicate sia allo scheletro che all'animale
        tensor_x, tensor_y = self.transform(tensor_x, tensor_y)
        
        # Scaliamo il target animale da [0, 1] a [-1, 1] per renderlo compatibile con lo spazio latente del VAE
        tensor_y = self.normalize_y(tensor_y)
        
        # Restituiamo un dizionario strutturato esattamente come richiesto dalla libreria Diffusers di Hugging Face
        return {
            "pixel_values": tensor_y,                # Target visivo finale che il modello deve riprodurre (Animale latente)
            "conditioning_pixel_values": tensor_x,   # Vincolo geometrico spaziale (Scheletro RGB)
            "text": prompt                           # Guida semantica (Stringa inglese o stringa vuota)
        }

# Modulo main per il testing autonomo del DataLoader (non viene eseguito quando importato in controlNet.py)
if __name__ == "__main__":
    dataset = OsteoDataset()
    # Batch size 4 è ottimale per test preliminari. Shuffle=True assicura la randomizzazione dei campioni
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Esegue una singola iterazione di test per stampare i metadati e confermare il funzionamento
    for batch in dataloader:
        print(f"Prompt Esempio 1: '{batch['text'][0]}'")
        print(f"Prompt Esempio 2: '{batch['text'][1]}'")
        break