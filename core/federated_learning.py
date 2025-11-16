# core/federated_learning.py
import torch
import copy
import time

class FederatedLearningBase:
    """Base federated learning framework"""
    
    def __init__(self, num_clients, model_class, device):
        self.num_clients = num_clients
        self.model_class = model_class
        self.device = device
        self.global_model = model_class().to(device)
        self.accuracy_history = []
    
    def train_round(self, client_loaders, epochs=2):
        """One round of federated learning"""
        client_models = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]
        
        # Train clients
        for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
            self._train_client(model, loader, epochs)
        
        # Aggregate
        self.global_model = self._aggregate(client_models)
        return self.global_model
    
    def _train_client(self, model, dataloader, epochs):
        """Train single client"""
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
    def _aggregate(self, client_models):
        """Base aggregation - override in subclasses"""
        global_model = copy.deepcopy(client_models[0])
        global_dict = global_model.state_dict()
        
        for key in global_dict.keys():
            param_stack = torch.stack([model.state_dict()[key].float() for model in client_models])
            global_dict[key] = param_stack.mean(0)
        
        global_model.load_state_dict(global_dict)
        return global_model
    
    def test_accuracy(self, test_loader):
        """Test current model"""
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        self.accuracy_history.append(accuracy)
        return accuracy