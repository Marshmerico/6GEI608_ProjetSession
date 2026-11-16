import torch
from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self, input_shape=(1, 20, 130), num_classes=10,
                 batch_norm=True, dropout=0.3):
        """batch_norm et dropout sont des options pour comparer les versions du modèle
        (batch_norm=False, dropout=0.0 redonne le modèle d'origine)."""
        super().__init__()
        # 4 conv blocks -> flatten -> dropout -> linear (logits)
        self.conv1 = self._block(1, 16, batch_norm)
        self.conv2 = self._block(16, 32, batch_norm)
        self.conv3 = self._block(32, 64, batch_norm)
        self.conv4 = self._block(64, 128, batch_norm)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        # taille du vecteur aplati calculée avec une entrée factice,
        # pour ne pas la recalculer à la main si la forme d'entrée change (ex. mel-spectrogramme)
        with torch.no_grad():
            self.eval()
            dummy = torch.zeros(1, *input_shape)
            n_features = self._features(dummy).shape[1]
            self.train()
        self.linear = nn.Linear(n_features, num_classes)

    @staticmethod
    def _block(in_channels, out_channels, batch_norm):
        # Conv -> (BatchNorm) -> ReLU -> MaxPool
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers += [nn.ReLU(), nn.MaxPool2d(kernel_size=2)]
        return nn.Sequential(*layers)

    def _conv(self, x):
        # 4 blocs de convolution : (N, 1, bandes, trames) -> (N, 128, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def _features(self, x):
        return self.flatten(self._conv(x))

    def embed(self, x):
        """Embedding d'un segment : 128 valeurs, une par filtre de conv4.
        Moyenne de chaque carte de caractéristiques sur la fréquence et le temps
        (global average pooling) : (N, 128, h, w) -> (N, 128).
        Ni dropout ni couche linéaire : on veut la représentation, pas la décision de genre.
        Appeler avec model.eval() et torch.no_grad()."""
        return self._conv(x).mean(dim=(2, 3))

    def forward(self, input_data):
        # retourne des logits : nn.CrossEntropyLoss applique déjà le softmax,
        # le softmax n'est appliqué qu'à l'inférence pour obtenir des probabilités
        x = self._features(input_data)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


if __name__ == "__main__":
    cnn = CNNNetwork(input_shape=(1, 96, 130))
    # summary -> (model, input_shape)
    summary(cnn, (1, 96, 130), device="cpu")   # (canaux, bandes mel, trames)
    cnn.eval()
    with torch.no_grad():
        print("embed :", cnn.embed(torch.zeros(4, 1, 96, 130)).shape)   # attendu : (4, 128)
