import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.dataloaders import get_mnist_dataloaders_fast
from models.mnist_cnn import MNISTCNN
from baselines.fed_avg import federated_average, train_client_model
from baselines.fed_avg_dp import federated_average_dp, train_client_model_dp
import torch
import time

def test_dp_final():
    print("🔒 FINAL DP Testing with Working Model...")
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data
    client_loaders, test_loader = get_mnist_dataloaders_fast(num_clients=2, batch_size=64)
    
    # First, test WITHOUT DP to get baseline
    print("\n=== BASELINE (No DP) ===")
    client_models = [MNISTCNN() for _ in range(len(client_loaders))]
    for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
        train_client_model(model, loader, device, epochs=3)
    baseline_model = federated_average(client_models)
    
    # Test baseline
    baseline_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = baseline_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    baseline_acc = 100 * correct / total
    print(f"📊 Baseline Accuracy: {baseline_acc:.2f}%")
    
    # Now test WITH DP
    print("\n=== WITH DIFFERENTIAL PRIVACY ===")
    
    # Test reasonable privacy levels
    privacy_levels = [100.0, 50.0, 10.0, 5.0, 1.0]
    
    results = []
    for epsilon in privacy_levels:
        print(f"\n--- Testing ε={epsilon} ---")
        
        # Fresh models
        client_models = [MNISTCNN() for _ in range(len(client_loaders))]
        
        # Train with DP
        for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
            train_client_model_dp(model, loader, device, epochs=3, clip_norm=1.0)
        
        # DP aggregation
        global_model = federated_average_dp(client_models, epsilon=epsilon, delta=1e-5, clip_norm=1.0)
        
        # Test
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        drop = baseline_acc - accuracy
        results.append((epsilon, accuracy, drop))
        
        print(f"📊 DP Accuracy: {accuracy:.2f}%")
        print(f"📉 Accuracy Drop: {drop:.2f}%")
    
    # Summary
    print("\n" + "="*50)
    print("📈 PRIVACY-UTILITY TRADEOFF SUMMARY")
    print("="*50)
    print(f"{'ε':>8} {'Accuracy':>10} {'Drop':>8}")
    print("-"*50)
    for epsilon, acc, drop in results:
        print(f"{epsilon:>8.1f} {acc:>9.2f}% {drop:>7.2f}%")
    
    total_time = time.time() - start_time
    print(f"\n🏁 Total time: {total_time:.1f} seconds")

if __name__ == "__main__":
    test_dp_final()