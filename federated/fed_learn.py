import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from cryptography.fernet import Fernet
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, encryption_key):
        self.model = model
        self.fernet = Fernet(encryption_key)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def get_parameters(self, config):
        params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        encrypted_params = self._encrypt_params(params)
        return encrypted_params
    
    def set_parameters(self, parameters):
        decrypted_params = self._decrypt_params(parameters)
        params_dict = zip(self.model.state_dict().keys(), decrypted_params)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        dummy_data = torch.randn(32, 10)
        dummy_labels = torch.randint(0, 2, (32, 1)).float()
        
        self.model.train()
        for epoch in range(5):
            self.optimizer.zero_grad()
            outputs = self.model(dummy_data)
            loss = self.criterion(outputs, dummy_labels)
            loss.backward()
            self.optimizer.step()
        
        return self.get_parameters(config={}), len(dummy_data), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        dummy_data = torch.randn(16, 10)
        dummy_labels = torch.randint(0, 2, (16, 1)).float()
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(dummy_data)
            loss = self.criterion(outputs, dummy_labels)
            accuracy = ((outputs > 0.5) == dummy_labels).float().mean()
        
        return float(loss), len(dummy_data), {"accuracy": float(accuracy)}
    
    def _encrypt_params(self, params):
        encrypted = []
        for param in params:
            param_bytes = param.tobytes()
            encrypted_bytes = self.fernet.encrypt(param_bytes)
            encrypted.append(np.frombuffer(encrypted_bytes, dtype=np.uint8))
        return encrypted
    
    def _decrypt_params(self, encrypted_params):
        decrypted = []
        for enc_param in encrypted_params:
            encrypted_bytes = enc_param.tobytes()
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            param = np.frombuffer(decrypted_bytes, dtype=np.float32).reshape(enc_param.shape)
            decrypted.append(param)
        return decrypted


class FederatedManager:
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.model = SimpleNet()
        self.strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
        
    def create_client_fn(self):
        def client_fn(cid: str) -> fl.client.Client:
            model = SimpleNet()
            return FederatedClient(model, self.encryption_key).to_client()
        return client_fn
    
    def start_simulation(self, num_clients=3, num_rounds=3):
        try:
            logger.info(f"Starting federated simulation with {num_clients} clients for {num_rounds} rounds")
            
            fl.simulation.start_simulation(
                client_fn=self.create_client_fn(),
                num_clients=num_clients,
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=self.strategy,
                client_resources={"num_cpus": 1, "num_gpus": 0.0},
            )
            
            logger.info("Federated simulation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Federated simulation failed: {e}")
            return False
    
    def encrypt_gradients(self, gradients):
        fernet = Fernet(self.encryption_key)
        encrypted = fernet.encrypt(str(gradients).encode())
        return encrypted
    
    def decrypt_gradients(self, encrypted_gradients):
        fernet = Fernet(self.encryption_key)
        decrypted = fernet.decrypt(encrypted_gradients).decode()
        return eval(decrypted)


def start_fed_sim(num_clients=3):
    manager = FederatedManager()
    return manager.start_simulation(num_clients=num_clients)