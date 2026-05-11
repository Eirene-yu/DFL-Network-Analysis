import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Agent:
    """
    Класс агента для децентрализованного федеративного обучения.
    """    
    def __init__(self, data, agent_id, model, steps_per_round=1, batch_size=64, device='cpu', random_seed=42):
        """
        Инициализирует агента с данными и моделью.
        """
        self.agent_id = agent_id
        self.random_seed = random_seed
        self.rng = np.random.RandomState(self.random_seed)

        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        self.model = model().to(self.device)
        
        X_list, y_list = [], []
        for x, y in data:
            X_list.append(x)
            y_list.append(torch.tensor(y))
        self.X = torch.stack(X_list).to(self.device)
        self.y = torch.stack(y_list).to(self.device)

        self.n_samples = len(self.X)
        self.batch_size = min(batch_size, self.n_samples)
        self.steps_per_round = steps_per_round
        self.local_sample_number = len(data)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
    
    def get_sample_number(self):
        """
        Возвращает количество примеров у агента.
        """
        return self.local_sample_number
    
    def train(self, learning_rate):
        """
        Выполняет локальное обучение на данных агента.
        """
        # SGD оптимизатор
        # optimizer = optim.SGD(self.model.parameters(), lr=learning_rate) - MNIST
        self.model.train()
        if self.optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        for _ in range(self.steps_per_round):
            indices = self.rng.choice(self.n_samples, self.batch_size, replace=False)
            
            X_batch = self.X[indices]
            y_batch = self.y[indices]
            
            #optimizer.zero_grad() - MNIST
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.loss_fn(logits, y_batch)
            loss.backward()
            #optimizer.step() - MNIST
            self.optimizer.step()
    
    def get_weights(self):
        """
        Возвращает копию весов модели агента.
        """
        return {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights):
        """
        Устанавливает веса модели агента.
        """
        device_weights = {k: v.to(self.device) for k, v in weights.items()}
        self.model.load_state_dict(device_weights)