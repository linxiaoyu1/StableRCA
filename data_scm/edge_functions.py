import torch.nn as nn
import torch
import random
import numpy as np

class LinearFunction(nn.Module):
    def __init__(self, function_params, device='cpu'):
        super().__init__()
        if function_params is None:
            function_params = {}
        self.in_dim = function_params.get('in_dim', 1)
        self.out_dim = function_params.get('out_dim', 1)
        # self.init_std = function_params.get('init_std', 1)
        self.uniform_l = function_params.get('uniform_l', 0.25)
        self.uniform_u = function_params.get('uniform_u', 1)
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=function_params.get('bias', False), device=device)  # Projects from input_dim -> 1

        for i, (n, p) in enumerate(self.linear.named_parameters()):
            if len(p.shape) == 2:
                # nn.init.normal_(p, std=self.init_std)
                r = np.random.rand()
                if r <= 0.5:
                    nn.init.uniform_(p, a=-self.uniform_u, b=-self.uniform_l)
                else:
                    nn.init.uniform_(p, a=self.uniform_l, b=self.uniform_u)
    def forward(self, x):
        return self.linear(x).squeeze(-1) 

    
class MLPFunction(nn.Module):
    def __init__(self, function_params, device='cpu'):
        '''
        params:
            d: input/output dimension
            sigma: std of noise 
        '''
        super(MLPFunction, self).__init__()
        activation_choices = {
        "identity": nn.Identity(),
        "sigmoid": nn.Hardsigmoid(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "softplus":nn.Softplus()
        }
        if function_params is None:
            function_params = {}
        self.in_dim = function_params.get('in_dim', 1)
        self.out_dim = function_params.get('out_dim', 1)
        self.hidden_dim = function_params.get('hidden_dim', 10)
        self.init_std = function_params.get('init_std', 1)
        self.n_hidden_layers = function_params.get('n_hidden_layers', 2)
        
        self.selected_name, self.activation = random.choice(list(activation_choices.items()))
        
        mlp_layers = []
        # Input layer
        mlp_layers.append(nn.Linear(self.in_dim, self.hidden_dim, bias=function_params.get('bias', False), device=device))
        mlp_layers.append(self.activation)

        # Hidden layers
        for _ in range(self.n_hidden_layers - 1):
            mlp_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=function_params.get('bias', False), device=device))
            mlp_layers.append(self.activation)

        # Output layer
        mlp_layers.append(nn.Linear(self.hidden_dim, self.out_dim, bias=function_params.get('bias', False), device=device))
        self.mlp = nn.Sequential(*mlp_layers)
        for i, (n, p) in enumerate(self.mlp.named_parameters()):
            if len(p.shape) == 2:
                nn.init.normal_(p, mean=0, std=self.init_std)

    def forward(self, x):
        with torch.no_grad():
            output = self.mlp(x).squeeze(-1)
        return output.float()