import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from heatmap import SkeletonKeypointDataset
from resnet import ResNet18KeypointDetector

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def salva_render_confronto(model, dataset, device, epoch, save_dir):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    img_tensor, target_tensor = dataset[idx]
    
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_tensor = model(img_batch)
    
    img_vis = img_tensor.numpy().transpose(1, 2, 0)
    img_vis = (img_vis * 0.5) + 0.5
    
    target_vis = target_tensor.numpy().sum(axis=0)
    pred_vis = pred_tensor.cpu().squeeze(0).numpy().sum(axis=0)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img_vis)
    ax[0].set_title(f"Input Originale (Epoca {epoch+1})")
    ax[0].axis("off")
    
    ax[1].imshow(img_vis)
    ax[1].imshow(pred_vis, cmap="jet", alpha=0.5)
    ax[1].set_title("Predizione Rete")
    ax[1].axis("off")
    
    ax[2].imshow(img_vis)
    ax[2].imshow(target_vis, cmap="jet", alpha=0.5)
    ax[2].set_title("Target Ground Truth")
    ax[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}_render.png"), bbox_inches='tight')
    plt.close(fig)
    model.train()

def train_model(resume_path=None):
    batch_size = 16 
    num_epochs = 400
    learning_rate = 1e-4
    patience = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training in esecuzione su: {device}")

    img_dir = r"C:\Users\alexc\Desktop\GitHub Projects\OsteoGen\OsteoGen_Version2\data\processed\input_x"
    json_dir = r"C:\Users\alexc\Desktop\GitHub Projects\OsteoGen\OsteoGen_Version2\data\processed\Labels_X"
    out_dir = r"C:\Users\alexc\Desktop\GitHub Projects\OsteoGen\OsteoGen_Version2\training_outputs_2"
    weights_dir = os.path.join(out_dir, "Weights")
    renders_dir = os.path.join(out_dir, "Epoch_Renders")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(renders_dir, exist_ok=True)

    dataset = SkeletonKeypointDataset(img_dir=img_dir, json_dir=json_dir, img_size=512, sigma=5.0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ResNet18KeypointDetector(num_keypoints=14).to(device)
    
    # 1. Ottimizzatore AdamW con Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 2. Scheduler che dimezza il LR se la loss stalla per 10 epoche
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    start_epoch = 0
    history_loss = []
    epochs_no_improve = 0 

    if resume_path and os.path.exists(resume_path):
        print(f"Ripristino training dal checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        history_loss = checkpoint['history_loss']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)

    print(f"Inizio Addestramento su {len(dataset)} campioni...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for immagini, heatmaps in dataloader:
            immagini, heatmaps = immagini.to(device), heatmaps.to(device)
            
            optimizer.zero_grad()
            outputs = model(immagini)
            
            # Calcolo Weighted MSE
            mse = (outputs - heatmaps) ** 2
            pesi_spaziali = torch.where(heatmaps > 0.05, 50.0, 1.0)
            loss = (mse * pesi_spaziali).mean()
            loss.backward()
            
            # 3. Gradient Clipping per stabilità
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * immagini.size(0)
            
        epoch_loss = running_loss / len(dataset)
        history_loss.append(epoch_loss)
        
        # Aggiornamento dello Scheduler
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoca [{epoch+1}/{num_epochs}] - Weighted MSE Loss: {epoch_loss:.6f} | LR: {current_lr:.6f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_keypoint_detector.pth"))
            print(f"  >>> Nuovo record! Loss: {best_loss:.6f} | Generazione render in corso...")
            salva_render_confronto(model, dataset, device, epoch, renders_dir)
        else:
            epochs_no_improve += 1
            print(f"  --- Nessun miglioramento da {epochs_no_improve} epoche (Patience limite: {patience})")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'history_loss': history_loss,
            'epochs_no_improve': epochs_no_improve
        }, os.path.join(weights_dir, "checkpoint_latest.pth"))
        
        if epochs_no_improve >= patience:
            print(f"\n[!] EARLY STOPPING attivato all'epoca {epoch+1}.")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history_loss) + 1), history_loss, marker='o', linestyle='-', color='b')
    plt.title("Curva di Addestramento (Weighted MSE Loss)")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()
    print("Training concluso. Grafico Loss generato.")

if __name__ == "__main__":
    # Resetta i pesi partendo da zero per testare la nuova architettura
    train_model(resume_path=None)