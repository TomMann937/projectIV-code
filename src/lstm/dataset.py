import torch 
from torch.utils.data import Dataset

class WindowedTimeSeries(Dataset):
  def __init__(self, input_series, target_series=None, seq_length=10):

    self. input_series = torch.tensor(input_series, dtype=torch.float32)

    if target_series is None:
      target_series = input_series

    self.target_series = torch.tensor(target_series, dtype=torch.float32)

    # Length of window
    self.seq_length = seq_length

  def __len__(self):
    return self.series.shape[0] - self.seq_length
  
  def __getitem__(self, idx):
    # Get a single window and it's target
    x = self.input_series[idx : idx + self.seq_length]
    y = self.target_series[idx + self.seq_length]
    return x, y