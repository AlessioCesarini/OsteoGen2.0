"""
finetune.py

Continua l'addestramento del keypoint detector esistente (best_keypoint_detector.pth)
per un piccolo numero di epoche aggiuntive, senza ripartire da zero, sfruttando le
specie supplementari aggiunte a data/processed/input_x/ (es. trex.png,
brachiosauro.png - proporzioni molto diverse dalle 65 specie viventi originali).

Perche' serve: il modello base generalizza benissimo sulle 65 specie di training
(confidenza media ~0.9+) ma molto peggio su forme corporee mai viste durante il
training come quella di un T-Rex o di un Brachiosauro (confidenza ~0.3-0.6, con
piu' dei 14 keypoint che collassano sullo stesso punto) - vedi
outputs/t-rex_report.html/brachiosauro_report.html per un esempio del sintomo.
Le nuove specie sono annotate solo per questo scopo e sono escluse dal database
di retrieval (vedi ESCLUSI_DAL_DB in build_DB.py): non sono donatori di texture,
sono target di riferimento in piu' per il rilevamento keypoint.

Uso (su una macchina con GPU, poche epoche/pochi minuti):
    python finetune.py
    python finetune.py --pesi altro/percorso/pesi.pth --epoche 80 --lr 5e-6

Salva il risultato in training_outputs_2/Weights/best_keypoint_detector_finetuned.pth,
SENZA sovrascrivere il file originale: cosi' si puo' confrontare la confidenza
prima/dopo (vedi diagnose_keypoints.py o semplicemente pipeline.py --pesi <nuovo file>)
prima di promuoverlo a best_keypoint_detector.pth e ricaricarlo su Hugging Face.
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from heatmap import SkeletonKeypointDataset
from resnet import ResNet18KeypointDetector

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def finetune(pesi_path, epoche=60, lr=1e-5, batch_size=16, output_path=None):
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(_base_dir, "data", "processed", "input_x")
    json_dir = os.path.join(_base_dir, "data", "processed", "Labels_X")
    output_path = output_path or os.path.join(_base_dir, "training_outputs_2", "Weights",
                                               "best_keypoint_detector_finetuned.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning in esecuzione su: {device}")

    # augment=True come in train.py: importante anche qui, altrimenti le sole
    # 1-2 immagini nuove verrebbero viste identiche ad ogni epoca.
    dataset = SkeletonKeypointDataset(img_dir=img_dir, json_dir=json_dir, img_size=512,
                                       sigma=5.0, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Dataset: {len(dataset)} campioni totali.")

    model = ResNet18KeypointDetector(num_keypoints=14).to(device)
    model.load_state_dict(torch.load(pesi_path, map_location=device))
    print(f"Pesi di partenza caricati da: {pesi_path}")

    # Learning rate piu' basso di quello usato in train.py (1e-4): stiamo
    # affinando un modello gia' convergente, non ripartendo da zero - un LR
    # alto rischierebbe di rovinare quanto imparato sulle 65 specie originali
    # solo per adattarsi alle 1-2 nuove.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float('inf')
    for epoch in range(epoche):
        model.train()
        running_loss = 0.0
        for immagini, heatmaps in dataloader:
            immagini, heatmaps = immagini.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            outputs = model(immagini)
            mse = (outputs - heatmaps) ** 2
            pesi_spaziali = torch.where(heatmaps > 0.05, 50.0, 1.0)
            loss = (mse * pesi_spaziali).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * immagini.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoca [{epoch + 1}/{epoche}] - Weighted MSE Loss: {epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), output_path)

    print(f"\nFine-tuning completato. Migliori pesi salvati in: {output_path}")
    return output_path


if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _pesi_default = os.path.join(_base_dir, "training_outputs_2", "Weights", "best_keypoint_detector.pth")

    parser = argparse.ArgumentParser(description="Fine-tuning del keypoint detector su specie supplementari.")
    parser.add_argument("--pesi", default=_pesi_default, help="Pesi di partenza (default: quelli attuali).")
    parser.add_argument("--epoche", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    finetune(args.pesi, epoche=args.epoche, lr=args.lr, batch_size=args.batch_size)
