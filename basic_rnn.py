import torch
import torch.nn as nn
import numpy as np
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
        hidden = torch.tanh(torch.mm(input, self.Wih) + torch.mm(hidden, self.Whh) + self.bh)
        output = torch.mm(hidden, self.Who) + self.bo
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

# Parameters
input_size = 10
hidden_size = 302
output_size = 5
batch_size = 32
sequence_length = 50

rnn = CustomRNN(input_size, hidden_size, output_size)

# init hidden state
hidden = rnn.init_hidden(batch_size)

# loss & optim
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# dummy data for training loop !!
inputs = torch.randn(sequence_length, batch_size, input_size)
inputs = normalize_data(inputs)
targets = torch.randint(0, 2, (sequence_length, batch_size, output_size)).float()

losses = []
# training loop
epochs = 100
for epoch in range(epochs):
    for i in range(sequence_length):
        input = inputs[i]
        target = targets[i]

        optimizer.zero_grad()
        output, hidden = rnn(input, hidden.detach())  # Detach hidden to prevent backprop through the entire sequence, we do this because it prevents gradient explosion
        # maybe can also just use a diff activation function??
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Extract the losses from the training loop
# losses = []  # Replace this with your actual loss values

# Plot the losses
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
