import torch 
from torch.utils.data import Dataset

class WindowedTimeSeries(Dataset):
  def __init__(self, series, seq_length):
    
    self.series = torch.tensor(series, dtype=torch.float32)
    # Length of window
    self.seq_length = seq_length

  def __len__(self):
    return self.series.shape[0] - self.seq_length
  
  def __getitem__(self, idx):
    # Get a single window and it's target
    x = self.series[idx : idx + self.seq_length]
    y = self.series[idx + self.seq_length]
    return x, y