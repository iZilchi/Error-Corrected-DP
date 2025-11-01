import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataloaders import get_mnist_dataloaders
from models.mnist_cnn import MNISTCNN
from baselines.fed_avg import federated_average, train_client_model
import torch

def test_baseline_fl():
    print("🧪 Testing Baseline Federated Learning...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data
    client_loaders, test_loader = get_mnist_dataloaders(num_clients=3, batch_size=32)
    print(f"Created {len(client_loaders)} client dataloaders")
    
    # Create client models
    client_models = [MNISTCNN() for _ in range(len(client_loaders))]
    
    # Train each client
    print("Training client models...")
    for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
        print(f"  Client {i+1}...")
        train_client_model(model, loader, device, epochs=1)
    
    # Federated averaging
    print("Performing federated averaging...")
    global_model = federated_average(client_models)
    
    # Test accuracy
    global_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = global_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"📊 Global Model Test Accuracy: {accuracy:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    test_baseline_fl()