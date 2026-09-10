"""
finetune.py

Continues training the existing keypoint detector (best_keypoint_detector.pth)
for a small number of additional epochs, without starting from scratch, using
the supplementary species added to data/processed/input_x/ (e.g. trex.png,
brachiosauro.png - proportions very different from the 65 original living
species).

Why this is needed: the base model generalizes very well on the 65 training
species (average confidence ~0.9+) but much worse on body plans never seen
during training, such as a T-Rex or a Brachiosaurus (confidence ~0.3-0.6,
with several of the 14 keypoints collapsing onto the same point) - see
outputs/t-rex_report.html / brachiosauro_report.html for an example of the
symptom. The new species are annotated only for this purpose and are
excluded from the retrieval database (see ESCLUSI_DAL_DB in build_DB.py):
they are not texture donors, they're extra reference targets for keypoint
detection.

Usage (on a GPU machine, a few epochs/minutes):
    python finetune.py
    python finetune.py --weights other/path/weights.pth --epochs 80 --lr 5e-6

Saves the result to training_outputs_2/Weights/best_keypoint_detector_finetuned.pth,
WITHOUT overwriting the original file: this way confidence can be compared
before/after (see diagnose_keypoints.py, or simply pipeline.py --weights <new file>)
before promoting it to best_keypoint_detector.pth and re-uploading it to
Hugging Face.
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
    print(f"Fine-tuning running on: {device}")

    # augment=True as in train.py: important here too, otherwise the 1-2 new
    # images alone would look identical every epoch.
    dataset = SkeletonKeypointDataset(img_dir=img_dir, json_dir=json_dir, img_size=512,
                                       sigma=5.0, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Dataset: {len(dataset)} total samples.")

    model = ResNet18KeypointDetector(num_keypoints=14).to(device)
    model.load_state_dict(torch.load(pesi_path, map_location=device))
    print(f"Starting weights loaded from: {pesi_path}")

    # Lower learning rate than train.py's (1e-4): this is refining an
    # already-converged model, not starting from scratch - a high LR would
    # risk undoing what was learned on the 65 original species just to fit
    # the 1-2 new ones.
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
        print(f"Epoch [{epoch + 1}/{epoche}] - Weighted MSE Loss: {epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), output_path)

    print(f"\nFine-tuning complete. Best weights saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _pesi_default = os.path.join(_base_dir, "training_outputs_2", "Weights", "best_keypoint_detector.pth")

    parser = argparse.ArgumentParser(description="Fine-tune the keypoint detector on supplementary species.")
    parser.add_argument("--weights", dest="pesi", default=_pesi_default,
                         help="Starting weights (default: the current ones).")
    parser.add_argument("--epochs", dest="epoche", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    finetune(args.pesi, epoche=args.epoche, lr=args.lr, batch_size=args.batch_size)
