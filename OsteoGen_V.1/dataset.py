import os
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")

# Italian dataset filenames -> English, for the prompt fed to Stable
# Diffusion (an English caption gives the foundation model much better
# adherence than an Italian one, since it was trained on English captions).
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


class OsteoDataset(Dataset):
    def __init__(self, data_dir: str = _DEFAULT_DATA_DIR, transform=None):
        self.data_dir = Path(data_dir)
        self.x_dir = self.data_dir / "input_X"
        self.y_dir = self.data_dir / "target_Y"

        self.filenames = sorted([f.name for f in self.x_dir.glob("*.png")])
        print(f"DEBUG - Found {len(self.filenames)} input files.")

        self.transform = transform or self.get_default_transforms()

        # Scales the target (animal) image to [-1, 1], required by Stable
        # Diffusion's VAE.
        self.normalize_y = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # The whole dataset is pre-loaded into RAM up front.
        self.data = []

        print("DEBUG - Pre-loading and translating into RAM...")

        for filename in self.filenames:
            x_path = self.x_dir / filename
            y_path = self.y_dir / filename

            if not y_path.exists():
                continue

            tensor_x = read_image(str(x_path), mode=ImageReadMode.RGB).float() / 255.0
            tensor_y = read_image(str(y_path), mode=ImageReadMode.RGB).float() / 255.0

            animal_name_it = Path(filename).stem.lower()
            animal_name_en = TRANSLATION_MAP.get(animal_name_it, animal_name_it)

            prompt = f"A full-body three-quarter view photo of a {animal_name_en}, featuring highly detailed, photorealistic textures, 8K resolution, studio lighting, and fully isolated against a pure black background."

            self.data.append({
                "tensor_x": tensor_x,
                "tensor_y": tensor_y,
                "prompt": prompt,
                "debug_info": f"{animal_name_it} -> {animal_name_en}"
            })

        print("DEBUG - Pre-loading complete.")

    def get_default_transforms(self):
        return v2.Compose([
            # Horizontal flip preserves anatomical geometry while adding mirror variance.
            v2.RandomHorizontalFlip(p=0.5),
            # Mild rotation/scale jitter for some invariance to the skeleton's framing.
            v2.RandomApply([
                v2.RandomAffine(degrees=[-10, 10], translate=[0.05, 0.05], scale=[0.95, 1.05], fill=0)
            ], p=0.5)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tensor_x = item["tensor_x"]
        tensor_y = item["tensor_y"]
        prompt = item["prompt"]

        # Prompt dropout (classifier-free guidance): 10% of the time, force
        # an empty prompt so the network learns to rely on the skeleton's
        # geometry alone, not just the text - essential for the zero-shot
        # test (e.g. on the T-Rex). Done here (not at pre-load time) so it's
        # re-randomized on every epoch.
        if random.random() < 0.10:
            prompt = ""

        # The v2 API applies the same spatial transform to both tensors.
        tensor_x, tensor_y = self.transform(tensor_x, tensor_y)

        tensor_y = self.normalize_y(tensor_y)

        return {
            "pixel_values": tensor_y,               # target the model must reproduce
            "conditioning_pixel_values": tensor_x,  # geometric constraint (skeleton)
            "text": prompt,                         # semantic guidance (English or empty)
        }


if __name__ == "__main__":
    dataset = OsteoDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        print(f"Example prompt 1: '{batch['text'][0]}'")
        print(f"Example prompt 2: '{batch['text'][1]}'")
        break
