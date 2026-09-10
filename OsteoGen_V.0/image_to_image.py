import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from src.dataset import OsteoDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ==========================================
# Baseline architecture: a plain U-Net
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Encoder: compresses the image and extracts geometric features.
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Each pooling step is a "zoom out": it lets deeper blocks see the
        # skeleton's macro-structure instead of individual bones/teeth.
        self.pool = nn.MaxPool2d(2)

        # Bottleneck (rudimentary latent space).
        self.bottleneck = self.conv_block(256, 512)

        # Decoder: decompresses and reconstructs the image.
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)  # 512 in: concatenated with the skip connection
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output layer.
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()  # matches the [-1, 1] normalization used on the target images

    def conv_block(self, in_channels, out_channels):
        """Two 3x3 convolutions (with BatchNorm+ReLU) that mix the input
        channels into out_channels while keeping the spatial resolution."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Downward path: trade spatial resolution for feature abstraction.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Latent space.
        b = self.bottleneck(self.pool(e3))

        # Upward path with skip connections.
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)  # concatenate along the channel axis
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return self.tanh(out)


def train_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs("training_dir_Image_to_Image", exist_ok=True)

    print("Loading dataset...")
    dataset = OsteoDataset(data_dir="OsteoGen/data/processed")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} pairs found.")

    print("Initializing the network...")
    model = SimpleUNet().to(device)

    # Loss: mean absolute error, pixel by pixel. Not squaring the error
    # keeps large mistakes penalized linearly rather than exponentially,
    # which in practice encourages sharper edges and more defined
    # phenotypic traits instead of blurry, averaged textures.
    criterion = nn.L1Loss()

    # Adam: the de facto standard for generative networks, adapts the
    # per-parameter step size from past gradients instead of using one
    # fixed learning rate for the whole network.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000  # needed for the loss to reach acceptable values at this batch size
    print("Starting training...")
    loss_history = []

    patience = 200  # epochs without improvement before stopping early
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()

        loop_batch = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx, (bones, real_animals) in enumerate(loop_batch):
            bones = bones.to(device)
            real_animals = real_animals.to(device)

            # --- standard training step ---
            optimizer.zero_grad()
            fake_animals = model(bones)
            loss = criterion(fake_animals, real_animals)
            loss.backward()
            optimizer.step()
            # -------------------------------

            loop_batch.set_postfix(loss=loss.item())
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}")

        loss_history.append(loss.item())

        # Early stopping + checkpointing.
        current_loss = loss.item()
        if current_loss < best_loss - 0.001:
            best_loss = current_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "training_dir_Image_to_Image/best_unet_baseline.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n[Early Stop] Training stopped at epoch {epoch + 1}.")
            print(f"The network stopped improving for {patience} epochs.")
            break

        # Save a preview grid every 10 epochs to avoid too many files.
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for test_bones, test_animals in dataloader:
                    test_bones = test_bones.to(device)
                    test_animals = test_animals.to(device)

                    pred_animals = model(test_bones)

                    # Denormalize from [-1, 1] to [0, 1] for save_image.
                    vis_bones = (test_bones[:4] * 0.5) + 0.5
                    vis_pred = (pred_animals[:4] * 0.5) + 0.5
                    vis_real = (test_animals[:4] * 0.5) + 0.5

                    # [Input skeleton, network's guess, real target]
                    comparison = torch.cat([vis_bones, vis_pred, vis_real], dim=0)
                    save_image(comparison, f"training_dir_Image_to_Image/epoch_{epoch + 1}.png", nrow=4)

                    break  # one example batch per epoch is enough

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="L1 Loss (Baseline U-Net)", color="red")
    plt.title("Learning Curve - Ablation Study")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_dir_Image_to_Image/final_loss_plot.png")
    print("Training complete, plot saved.")


if __name__ == "__main__":
    train_model()
