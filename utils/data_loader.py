# utils/data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_dataloaders(num_clients=3, batch_size=64):
    """Create MNIST dataloaders for federated learning"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split data among clients (3000 samples total)
    samples_per_client = 3000 // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data = Subset(train_dataset, range(start_idx, end_idx))
        client_datasets.append(client_data)
    
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"Created {num_clients} clients with {samples_per_client} samples each")
    return client_loaders, test_loader