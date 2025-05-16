import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledStdConv2d(nn.Conv2d):
    """Capa convolucional con Scaled Weight Standardization (sin BN)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, gamma=1.0, eps=1e-6):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, x):
        # Weight Standardization
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), keepdim=True)
        weight = (weight - mean) / (var + self.eps).sqrt()
        # Escalado
        weight = self.gamma * weight
        return F.conv2d(x, weight, self.bias, self.stride, self.padding)

class ScaledReLU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Convertir a parámetro aprendible

    def forward(self, x):
        return F.relu(x) * self.alpha

class NFBlock(nn.Module):
    """Bloque residual de NFNet"""
    def __init__(self, in_channels, out_channels, stride=1, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    ScaledStdConv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    )
            self.layers = nn.Sequential(
                    ScaledStdConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    ScaledReLU(alpha=self.alpha),
                    ScaledStdConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    ScaledReLU(alpha=self.alpha), 
                    )
        else:
            self.shortcut = nn.Sequential(
                    )
            self.layers = nn.Sequential(
                    ScaledStdConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                    ScaledReLU(alpha=self.alpha),
                    ScaledStdConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    ScaledReLU(alpha=self.alpha),
                    )
    def forward(self, x):
        return self.layers(x)

class NFNet(nn.Module):
    """Arquitectura NFNet-F0 simplificada"""
    def __init__(self, num_classes=10, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.neural_network_layers = nn.Sequential(
                ScaledStdConv2d(3, 32, kernel_size=3, stride=1, padding=3),
                nn.ReLU(),
                ScaledStdConv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                NFBlock(64, 128, stride=1, alpha=self.alpha),
                NFBlock(128, 256, stride=1, alpha=self.alpha),
                NFBlock(256, 512, stride=2, alpha=self.alpha),
                NFBlock(512, 1024, stride=2, alpha=self.alpha),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        self.dense_neural_network_layers = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes)
            )
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [NFBlock(in_channels, out_channels, stride=stride, alpha=self.alpha)]
        for _ in range(1, num_blocks):
            layers.append(NFBlock(out_channels, out_channels, stride=1, alpha=self.alpha))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Inicializar biases a cero

    def forward(self, x):
        x = self.neural_network_layers(x)
        x = self.dense_neural_network_layers(x.squeeze())
        return x
