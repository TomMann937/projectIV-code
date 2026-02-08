import torch 
from torch.utils.data import Dataset

class WindowedTimeSeries(Dataset):
  def __init__(self, input_series, assim_series=None, target_series=None, seq_length=10, horizon=1):

    self.input_series = torch.tensor(input_series, dtype=torch.float32)

    if target_series is None:
      target_series = input_series
    self.target_series = torch.tensor(target_series, dtype=torch.float32)

    if assim_series is None:
      assim_series = input_series
    self.assim_series = torch.tensor(assim_series, dtype=torch.float32)

    # Length of window
    self.seq_length = seq_length
    self.horizon = horizon

    assert len(self.input_series) == len(self.target_series)
    assert len(self.input_series) == len(self.assim_series)

  def __len__(self):
    return self.input_series.shape[0] - self.seq_length - self.horizon + 1
  
  def __getitem__(self, idx):
    # Get a single window and it's target
    x_obs = self.input_series[idx : idx + self.seq_length]
    x_assim = self.assim_series[idx + self.seq_length]
    y = self.target_series[idx + self.seq_length + self.horizon - 1]

    return{
      "lstm_input": x_obs,
      "physics_input": x_assim,
      "target": y
      }