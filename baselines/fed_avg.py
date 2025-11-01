import torch
import copy

def federated_average(client_models):
    """
    Basic Federated Averaging - average model parameters
    """
    global_model = copy.deepcopy(client_models[0])
    global_dict = global_model.state_dict()
    
    # Average all parameters
    for key in global_dict.keys():
        global_dict[key] = torch.stack([model.state_dict()[key].float() for model in client_models]).mean(0)
    
    global_model.load_state_dict(global_dict)
    return global_model

def train_client_model(model, dataloader, device, epochs=3):
    """
    Train a client model on its local data - IMPROVED!
    """
    model.train()
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    return model