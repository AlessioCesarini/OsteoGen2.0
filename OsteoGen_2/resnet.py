import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18KeypointDetector(nn.Module):
    def __init__(self, num_keypoints=14):
        super(ResNet18KeypointDetector, self).__init__()
        
        # Carica i pesi pre-addestrati su ImageNet per avere filtri già pronti
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # ENCODER: teniamo i layer della ResNet18 fino all'ultimo blocco convoluzionale,
        # scartando l'Average Pooling e il layer lineare (Fully Connected) finale.
        # L'input 512x512 viene ridotto di 32 volte -> output encoder: 16x16 con 512 canali
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # DECODER: sequenza di ConvTranspose2d per "ingrandire" l'immagine (upsampling)
        # e riportarla alla risoluzione nativa di 512x512, riducendo gradualmente i canali.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 256x256
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Ultimo layer: non usiamo attivazioni (niente Sigmoide o ReLU) 
            # perché calcoleremo la MSELoss direttamente sui logit grezzi
            nn.ConvTranspose2d(32, num_keypoints, kernel_size=4, stride=2, padding=1) # -> 512x512
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Test rapido di dimensionamento
if __name__ == "__main__":
    model = ResNet18KeypointDetector(num_keypoints=14)
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"Shape Input: {dummy_input.shape}")
    print(f"Shape Output: {output.shape}") 
    # Dovrebbe stampare: torch.Size([1, 14, 512, 512])