import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import OsteoDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # avoids an OpenMP crash on Windows


# ==========================================
# Generator: a deeper 6-level U-Net
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # --- Encoder ---
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        # Channels are capped at 512 from here on: this forces the network
        # to go spatially deep (down to an 8x8 view of the whole pose)
        # without an explosion in parameter count, encouraging it to learn
        # general structure rather than memorize.
        self.enc5 = self.conv_block(512, 512)
        self.enc6 = self.conv_block(512, 512)

        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck (8x8 latent space) ---
        self.bottleneck = self.conv_block(512, 512)

        # --- Decoder ---
        # upconv = spatial upsampling; dec = fuses it with the matching
        # skip connection (concatenated channels, halved back down).
        self.upconv6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec6 = self.conv_block(1024, 512)

        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = self.conv_block(1024, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 256)  # 512 (upconv4) + 512 (enc4) -> 256

        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 128)  # 256 + 256 -> 128

        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 64)  # 128 + 128 -> 64

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)  # 64 + 64 -> 64

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))

        b = self.bottleneck(self.pool(e6))

        d6 = self.upconv6(b)
        d6 = torch.cat([d6, e6], dim=1)
        d6 = self.dec6(d6)

        d5 = self.upconv5(d6)
        d5 = torch.cat([d5, e5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.upconv4(d5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return self.tanh(out)


# ==========================================
# Discriminator: PatchGAN
# ==========================================
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        # Standard Pix2Pix PatchGAN, 4x4 kernels. Takes 6 input channels:
        # 3 for the skeleton + 3 for the animal (real or generated).

        def discriminator_block(in_filters, out_filters, normalization=True, stride=2):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            # LeakyReLU(0.2): lets a small fraction of negative values through,
            # which matters in adversarial training to avoid dead ReLUs
            # stalling the generator's gradient.
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Layer 1: 6 -> 64 channels. No BatchNorm here so the raw RGB
            # colors aren't distorted before being evaluated.
            *discriminator_block(6, 64, normalization=False),
            # Layer 2: 64 -> 128.
            *discriminator_block(64, 128),
            # Layer 3: 128 -> 256. stride=1 here, an architectural trick:
            # it stops the spatial size shrinking further, keeping a grid
            # fine enough to evaluate local patches instead of collapsing
            # them into a single pixel.
            *discriminator_block(128, 256),
            # Layer 4: 256 -> 512, stride=1 again to prepare the output
            # resolution. No final activation (no Sigmoid): the output is a
            # raw logit matrix where each entry judges a 70x70 patch of the
            # original image as real/fake.
            *discriminator_block(256, 512, stride=1),

            # Final output layer: 512 semantic channels -> 1 spatial channel.
            # No Sigmoid here either, since nn.BCEWithLogitsLoss is used.
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, skeleton, animal):
        # The discriminator never looks at an image alone - it judges the
        # (skeleton, animal) pair, stacked along the channel axis, so it can
        # tell whether the texture actually follows the underlying bones.
        img_input = torch.cat([skeleton, animal], dim=1)
        return self.model(img_input)


# ==========================================
# Pix2Pix (GAN) training loop
# ==========================================
def train_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs("PatchGan_results", exist_ok=True)

    print("Loading dataset...")
    dataset = OsteoDataset(data_dir="data/processed")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} pairs found.")

    print("Initializing the networks...")
    generator = SimpleUNet().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    # BCEWithLogits applies the sigmoid internally, which is numerically
    # more stable than an explicit Sigmoid on the discriminator's last layer.
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    # lambda_L1=100: the Pix2Pix golden standard. Without it the generator
    # would chase photorealistic texture while ignoring the actual bone
    # structure (e.g. inventing a tail that isn't there) just to fool the
    # discriminator; weighting L1 100x says "matching the real shape and
    # color matters 100x more than looking merely realistic".
    lambda_L1 = 100

    # LR 0.0002 and betas (0.5, 0.999): standard Pix2Pix values, changing
    # them tends to collapse the GAN.
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    num_epochs = 2000  # early stopping will cut this short in practice
    patience = 500  # generous, GANs oscillate a lot by nature
    best_L1_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    history_loss_G_GAN = []
    history_loss_G_L1 = []
    history_loss_D = []

    print("Starting adversarial training... (Ctrl+C to stop safely)")

    try:
        for epoch in range(num_epochs):
            generator.train()
            discriminator.train()

            epoch_loss_G_GAN = 0.0
            epoch_loss_G_L1 = 0.0
            epoch_loss_D = 0.0

            loop_batch = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

            for batch_idx, (bones, real_animals) in enumerate(loop_batch):
                bones = bones.to(device)
                real_animals = real_animals.to(device)

                # --- Step 1: discriminator ---
                optimizer_D.zero_grad()
                fake_animals = generator(bones)

                pred_real = discriminator(bones, real_animals)
                target_real = torch.ones_like(pred_real).to(device)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # .detach(): cuts the graph so gradients don't flow back into
                # the generator while we're only training the discriminator.
                pred_fake = discriminator(bones, fake_animals.detach())
                target_fake = torch.zeros_like(pred_fake).to(device)
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()

                # --- Step 2: generator ---
                optimizer_G.zero_grad()
                # No .detach() this time: the error needs to flow back to
                # update the generator's weights.
                pred_fake_for_G = discriminator(bones, fake_animals)

                # The GAN's trick: compare the fake prediction against
                # target_real (all ones) - the generator is penalized
                # whenever D can tell the image is fake, pushing its
                # output towards fooling D.
                loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real)
                loss_G_L1_pure = criterion_L1(fake_animals, real_animals)
                loss_G_L1_scaled = loss_G_L1_pure * lambda_L1

                loss_G = loss_G_GAN + loss_G_L1_scaled
                loss_G.backward()
                optimizer_G.step()

                epoch_loss_G_GAN += loss_G_GAN.item()
                epoch_loss_G_L1 += loss_G_L1_pure.item()
                epoch_loss_D += loss_D.item()

                loop_batch.set_postfix(Loss_D=loss_D.item(), Loss_G_L1=loss_G_L1_pure.item())

            # --- per-epoch averages + early stopping ---
            num_batches = len(dataloader)
            avg_G_L1 = epoch_loss_G_L1 / num_batches
            avg_G_GAN = epoch_loss_G_GAN / num_batches
            avg_D = epoch_loss_D / num_batches

            history_loss_G_GAN.append(avg_G_GAN)
            history_loss_G_L1.append(avg_G_L1)
            history_loss_D.append(avg_D)

            print(f"-> End of epoch {epoch + 1} | Avg Loss D: {avg_D:.4f} | Avg Loss G (GAN): {avg_G_GAN:.4f} | Avg L1 (reconstruction): {avg_G_L1:.4f}")

            if avg_G_L1 < best_L1_loss:
                best_L1_loss = avg_G_L1
                best_epoch = epoch + 1
                epochs_no_improve = 0

                print(f"Improvement! L1 loss dropped to {best_L1_loss:.4f}. Saving weights...")

                torch.save(generator.state_dict(), "PatchGan_results/best_generator.pth")
                torch.save(discriminator.state_dict(), "PatchGan_results/best_discriminator.pth")
            else:
                epochs_no_improve += 1
                print(f"   (No L1 improvement for {epochs_no_improve} epochs)")

            if (epoch + 1) % 20 == 0:
                generator.eval()
                with torch.no_grad():
                    for test_bones, test_animals in dataloader:
                        test_bones = test_bones.to(device)
                        pred_animals = generator(test_bones)
                        vis_bones = (test_bones[:4] * 0.5) + 0.5
                        vis_pred = (pred_animals[:4] * 0.5) + 0.5
                        vis_real = (test_animals[:4].to(device) * 0.5) + 0.5
                        comparison = torch.cat([vis_bones, vis_pred, vis_real], dim=0)
                        save_image(comparison, f"PatchGan_results/epoch_{epoch + 1}.png", nrow=4)
                        break

            if epochs_no_improve >= patience:
                print(f"\n[Early Stop] No structural improvement for {patience} epochs.")
                break

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted manually (Ctrl+C).")
        print(f"Saving the current model at epoch {epoch + 1}...")
        torch.save(generator.state_dict(), f"PatchGan_results/generator_interrupted_ep{epoch + 1}.pth")
        torch.save(discriminator.state_dict(), f"PatchGan_results/discriminator_interrupted_ep{epoch + 1}.pth")

    finally:
        # Runs whether training finished naturally or was interrupted.
        print(f"\nGenerating final plots... Best model at epoch {best_epoch} (L1: {best_L1_loss:.4f})")
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history_loss_G_GAN, label='Generator Loss (Adversarial)', color='blue', alpha=0.7)
        plt.plot(history_loss_D, label='Discriminator Loss', color='red', alpha=0.7)
        if best_epoch > 0:
            plt.axvline(x=best_epoch - 1, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.title('Adversarial Dynamics (GAN Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history_loss_G_L1, label='L1 Loss (Reconstruction)', color='purple')
        if best_epoch > 0:
            plt.plot(best_epoch - 1, best_L1_loss, marker='*', markersize=15, color='red', label='Best Reconstruction')
        plt.title('Structural and Color Fidelity (L1 Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("PatchGan_results/overall_loss_plot.png")
        print("Plots saved. Training finished safely.")


if __name__ == "__main__":
    train_model()
