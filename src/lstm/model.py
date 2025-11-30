# Define LSTM Class
import torch.nn as nn

class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(LSTMModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, (hn, cn) = self.lstm(x)
    
    # Use the final timestep's hidden state
    out = out[:, -1, :]
    out = self.fc(out)
    return out

    