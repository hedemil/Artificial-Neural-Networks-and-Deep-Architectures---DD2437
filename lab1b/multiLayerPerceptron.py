import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr=0.001):
        super().__init__()
        # Initialize layers separately for better control
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, output_size)
        
        # Use Adam optimizer with lower learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_criteria = nn.MSELoss()

    def forward(self, x):
        # Sigmoid activation for hidden layers, but no activation for output
        x = torch.sigmoid(self.hidden1(x.to(torch.float32)))
        x = torch.sigmoid(self.hidden2(x))
        x = self.output(x)  # Linear output layer
        return x
    
        