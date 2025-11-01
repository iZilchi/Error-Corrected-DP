import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_dataloaders(num_clients=10, batch_size=32):
    """Create MNIST dataloaders for multiple clients"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download MNIST
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split data among clients
    client_datasets = []
    data_per_client = len(train_dataset) // num_clients
    
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = start_idx + data_per_client
        client_data = Subset(train_dataset, range(start_idx, end_idx))
        client_datasets.append(client_data)
    
    # Create dataloaders
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return client_loaders, test_loader

def get_mnist_dataloaders_fast(num_clients=2, batch_size=64):
    """Fast MNIST dataloaders with MORE data"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download MNIST once
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Use MORE data for proper learning
    total_samples = 3000  # Increased from 1000 to 3000
    samples_per_client = total_samples // num_clients
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data = Subset(train_dataset, range(start_idx, end_idx))
        client_datasets.append(client_data)
    
    # Create dataloaders
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"Created {num_clients} clients with {samples_per_client} samples each")
    return client_loaders, test_loader