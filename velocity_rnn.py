import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size

        # Initialize weights
        self.Wih = nn.Parameter(torch.randn(input_size, hidden_size) * np.sqrt(1. / input_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * np.sqrt(1. / hidden_size))
        self.Who = nn.Parameter(torch.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size))

        # Bias tersm init to be 0 !
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.bo = nn.Parameter(torch.zeros(output_size))

    def forward(self, input, hidden):
        """
        Forward pass through the RNN.
        input : Tensor of shape (batch_size, input_size)
        hidden : Tensor of shape (batch_size, hidden_size)
        """
        if hidden is None:
            hidden = self.init_hidden(input.size(0))
        hidden = torch.tanh(torch.matmul(input, self.Wih) + torch.matmul(hidden, self.Whh) + self.bh)
        output = torch.matmul(hidden, self.Who) + self.bo
        return output, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(batch_size, self.hidden_size)

def normalize_data(data):
    """
    Normalize the input data using mean and standard deviation.
    """
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    normalized_data = (data - mean) / std
    return normalized_data

# Load data from CSV files
neural_data = pd.read_csv('neurons.csv', header=0)  # assuming 143 columns with neuron data, 1600 rows
behavior_data = pd.read_csv('behavior.csv', header=None)  # assuming 1 column, 1600 rows

# Convert DataFrame to tensor
inputs = torch.tensor(neural_data.values, dtype=torch.float32)  # Convert to tensor
targets = torch.tensor(behavior_data.values, dtype=torch.float32)  # Convert to tensor

# Parameters
input_size = 143
hidden_size = 302
output_size = 1
batch_size = 32  # Adjust batch size according to your dataset and memory constraints
sequence_length = 50  # This should be smaller than 1600 to handle sequences

# Initialize the RNN
rnn = CustomRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# Reshape inputs and targets to batches of sequences
num_batches = inputs.size(0) // batch_size
inputs = inputs[:num_batches * batch_size].view(batch_size, num_batches, -1).transpose(0, 1)
targets = targets[:num_batches * batch_size].view(batch_size, num_batches, -1).transpose(0, 1)

# Training loop
losses = []
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for i in range(0, num_batches, sequence_length):
        input_batch = inputs[i:i+sequence_length]
        target_batch = targets[i:i+sequence_length]

        hidden = None
        optimizer.zero_grad()
        outputs = []
        for t in range(input_batch.size(0)):
            output, hidden = rnn(input_batch[t], hidden)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        loss = criterion(outputs, target_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

# Plot the training losses
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()