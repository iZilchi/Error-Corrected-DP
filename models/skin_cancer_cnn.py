# models/skin_cancer_cnn.py - NEW FILE
import torch
import torch.nn as nn

class SkinCancerCNN(nn.Module):
    """
    CNN model for Skin Cancer HAM10000 classification
    7 classes instead of 10, 3 input channels (RGB) instead of 1
    """
    def __init__(self):
        super(SkinCancerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 3 input channels for RGB
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  # Extra layer for complexity
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 256),  # Adjusted for 28x28 -> 3x3 after 3 max pools
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)  # 7 classes for skin cancer
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def get_gradients(self):
        """For future use with gradient-based methods"""
        gradients = []
        for param in self.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
        return gradients
    
    def set_gradients(self, gradients):
        """For future use with gradient-based methods"""
        for param, grad in zip(self.parameters(), gradients):
            param.grad = grad.clone()

# Alias for backward compatibility
MNISTCNN = SkinCancerCNN