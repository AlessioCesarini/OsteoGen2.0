import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18KeypointDetector(nn.Module):
    def __init__(self, num_keypoints=14):
        super(ResNet18KeypointDetector, self).__init__()

        # Load ImageNet-pretrained weights for ready-made filters.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # ENCODER: keep the ResNet18 layers up to the last conv block,
        # dropping the average pooling and final fully-connected layer.
        # A 512x512 input is downsampled 32x -> encoder output: 16x16, 512 channels.
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # DECODER: a stack of ConvTranspose2d layers that upsamples back
        # to the native 512x512 resolution, gradually reducing channels.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Last layer: no activation (no Sigmoid/ReLU) - the Weighted
            # MSE loss is computed directly on these raw logits.
            nn.ConvTranspose2d(32, num_keypoints, kernel_size=4, stride=2, padding=1)  # -> 512x512
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Quick shape sanity check.
if __name__ == "__main__":
    model = ResNet18KeypointDetector(num_keypoints=14)
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    # Expected: torch.Size([1, 14, 512, 512])
