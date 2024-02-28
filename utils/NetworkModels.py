import torch
import torch.nn as nn


# Define the RNN model
class MotorLearningRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_outputs):
        super(MotorLearningRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # Add a linear layer to map output dimension back to input dimension
        self.map_back_to_input = nn.Linear(output_size, input_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate RNN
        out, hn = self.rnn(x, h0)
        # Prepare the RNN output to repeat
        outputs = []
        for _ in range(self.num_outputs):
            out = self.fc(out)
            outputs.append(out)
            # Map the output back to input dimension
            out = self.map_back_to_input(out)
            # Use the mapped output as input for the next step
            out, hn = self.rnn(out, hn)
        # Stack outputs to match the expected structure
        outputs = torch.concatenate(outputs, dim=1)
        return outputs