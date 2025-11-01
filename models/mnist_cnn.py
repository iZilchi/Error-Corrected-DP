import torch
import torch.nn as nn

class MNISTCNN(nn.Module):
    """
    ONE good CNN model for MNIST - use this for everything
    Balanced between speed and accuracy
    """
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
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