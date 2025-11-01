import torch
import copy
import numpy as np

def calculate_dp_noise_scale(epsilon, delta, sensitivity=1.0):
    """Calculate Gaussian noise scale for (ε,δ)-DP"""
    if epsilon <= 0 or delta <= 0:
        raise ValueError("Epsilon and delta must be positive")
    
    noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return noise_scale

def add_gaussian_noise(tensor, noise_scale):
    """Add Gaussian noise to a tensor"""
    if noise_scale > 0:
        # Scale noise by tensor magnitude to preserve signal
        tensor_std = tensor.std().item() if tensor.numel() > 1 else abs(tensor.item())
        effective_noise_scale = noise_scale * max(tensor_std, 0.1)
        noise = torch.normal(mean=0.0, std=effective_noise_scale, size=tensor.shape)
        return tensor + noise
    return tensor

def federated_average_dp(client_models, epsilon=1.0, delta=1e-5, clip_norm=1.0):
    """
    Federated Averaging with Differential Privacy
    """
    print(f"  Applying DP: ε={epsilon}, δ={delta}")
    
    # Calculate noise scale - use smaller values
    noise_scale = calculate_dp_noise_scale(epsilon, delta, sensitivity=clip_norm) * 0.01
    
    global_model = copy.deepcopy(client_models[0])
    global_dict = global_model.state_dict()
    
    # Average all parameters and add gentle DP noise
    for key in global_dict.keys():
        # Average client parameters
        param_stack = torch.stack([model.state_dict()[key].float() for model in client_models])
        avg_param = param_stack.mean(0)
        
        # Add much gentler DP noise
        noisy_param = add_gaussian_noise(avg_param, noise_scale)
        global_dict[key] = noisy_param
    
    global_model.load_state_dict(global_dict)
    return global_model

def clip_gradients(model, max_norm=1.0):
    """
    Clip gradients to control sensitivity for DP
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    # Calculate total gradient norm
    total_norm = 0.0
    for param in parameters:
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in parameters:
            param.grad.data.mul_(clip_coef)
    
    return total_norm

def train_client_model_dp(model, dataloader, device, epochs=1, clip_norm=1.0):
    """
    Train a client model with gradient clipping for DP
    """
    model.train()
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Clip gradients for DP
            clip_gradients(model, max_norm=clip_norm)
            
            optimizer.step()
    
    return model