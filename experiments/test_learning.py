import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataloaders import get_mnist_dataloaders_fast
from models.mnist_cnn import MNISTCNN
from baselines.fed_avg import federated_average, train_client_model
import torch
import time

def test_learning_capability():
    """Test if our model can actually learn"""
    print("🎯 Testing Model Learning Capability...")
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data with more samples
    client_loaders, test_loader = get_mnist_dataloaders_fast(num_clients=2, batch_size=64)
    
    print("\n=== SINGLE MODEL TEST (No FL) ===")
    # Test if a single model can learn on one client's data
    single_model = MNISTCNN()
    single_loader = client_loaders[0]
    
    print("Training single model...")
    train_client_model(single_model, single_loader, device, epochs=5)
    
    # Test single model
    single_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = single_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    single_acc = 100 * correct / total
    print(f"📊 Single Model Accuracy: {single_acc:.2f}%")
    
    print("\n=== FEDERATED LEARNING TEST ===")
    # Now test FL
    client_models = [MNISTCNN() for _ in range(len(client_loaders))]
    
    for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
        print(f"Training client {i+1}...")
        train_client_model(model, loader, device, epochs=3)
    
    global_model = federated_average(client_models)
    
    # Test FL model
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = global_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    fl_acc = 100 * correct / total
    print(f"📊 FL Model Accuracy: {fl_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\n🏁 Total time: {total_time:.1f} seconds")
    
    # Check if learning is happening
    if single_acc > 50 or fl_acc > 50:
        print("✅ SUCCESS: Model is learning!")
        return True
    else:
        print("❌ PROBLEM: Model is not learning well")
        return False

if __name__ == "__main__":
    test_learning_capability()