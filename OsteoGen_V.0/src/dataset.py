import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class OsteoDataset(Dataset):
    def __init__(self, data_dir: str = "data/processed", transform=None):
        self.data_dir = Path(data_dir)
        self.x_dir = self.data_dir / "input_x" 
        self.y_dir = self.data_dir / "target_y"
        
        # 1. Stampa il percorso reale in cui Python sta curiosando
        print(f"DEBUG - Percorso assoluto: {self.x_dir.resolve()}")
        self.filenames = sorted([f.name for f in self.x_dir.glob("*.png")])
        
        # 2. Conferma quanti file vengono effettivamente visti
        print(f"DEBUG - File estratti: {len(self.filenames)}")
        
        self.transform = transform or self.get_default_transforms()
        # Scalato tra [-1, 1] per la Tanh del Generatore
        self.normalize = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def get_default_transforms(self):
        # Data Augmentation stocastica applicata sincronizzata (Y.2.4) - Nativa PyTorch
        return v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([
                v2.RandomAffine(degrees=[-10, 10], translate=[0.05, 0.05], scale=[0.95, 1.05], fill=0)
            ], p=0.5),
            v2.RandomApply([
                v2.ElasticTransform(alpha=50.0, sigma=5.0, fill=0)
            ], p=0.3),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        x_path = str(self.x_dir / filename)
        y_path = str(self.y_dir / filename)
        
        image_x = cv2.imread(x_path)
        image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
        
        image_y = cv2.imread(y_path)
        image_y = cv2.cvtColor(image_y, cv2.COLOR_BGR2RGB)
        
        # Le transforms V2 richiedono tensori float [Canali, Altezza, Larghezza] nel range [0.0, 1.0]
        tensor_x = torch.from_numpy(image_x).permute(2, 0, 1).float() / 255.0
        tensor_y = torch.from_numpy(image_y).permute(2, 0, 1).float() / 255.0
        
        # Applicazione sincronizzata della trasfomazione su X e Y contemporaneamente
        tensor_x, tensor_y = self.transform(tensor_x, tensor_y)
        
        # Normalizzazione finale
        tensor_x = self.normalize(tensor_x)
        tensor_y = self.normalize(tensor_y)
        
        return tensor_x, tensor_y

# Test rapido di caricamento del DataLoader
if __name__ == "__main__":
    dataset = OsteoDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_x, batch_y in dataloader:
        print(f"Batch X Shape: {batch_x.shape}") # Output atteso: [4, 3, 512, 512]
        print(f"Batch Y Shape: {batch_y.shape}") # Output atteso: [4, 3, 512, 512]
        break