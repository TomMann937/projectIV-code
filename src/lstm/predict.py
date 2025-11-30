import torch
import numpy as np

def predict_one_step(model, window, device=None):

  if device is None:
    device = next(model.parameters()).device

  if isinstance(window, np.ndarray):
    window = torch.tensor(window, dtype=torch.float32)
  elif isinstance(window, torch.Tensor):
    window = window.float()
  else:
    raise TypeError("Window must be a numpy array or torch tensor")
  
  if window.ndim == 1:
    window = window.unsqueeze(1)

  window = window.unsqueeze(0).to(device) 

  if window.ndim != 3:
    raise ValueError(f"window must end up 3D, got shape {window.shape}")

  model.eval()
  with torch.no_grad():
    pred = model(window)

  return pred.squeeze().cpu().numpy()

