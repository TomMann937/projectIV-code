import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

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
    window = window.unsqueeze(0).unsqueeze(-1)
  elif window.ndim == 2:
    window = window.unsqueeze(0)
  elif window.ndim == 3:
    pass
  else:
    raise ValueError(f"window must end up 3D, got shape {window.shape}")
  
  window = window.to(device) 

  model.eval()
  with torch.no_grad():
    pred = model(window)

  return pred.squeeze().cpu()


def predict_autoregressive(model, window, steps=1, device=None):

  if device is None:
    device = next(model.parameters()).device

  if isinstance(window, np.ndarray):
    window = torch.tensor(window, dtype=torch.float32)
  elif isinstance(window, torch.Tensor):
    window = window.float()
  else:
    raise TypeError("Window must be a numpy array or torch tensor")
  
  if window.ndim == 1:
    window = window.unsqueeze(0).unsqueeze(-1)
  elif window.ndim == 2:
    window = window.unsqueeze(0)
  elif window.ndim == 3:
    pass
  else:
    raise ValueError(f"window must end up 3D, got shape {window.shape}")
  
  window = window.to(device) 

  model.eval()
  with torch.no_grad():
    for _ in range(steps):

      # Predict next step, shape: (batch, output_size)
      next_pred = model(window)

      # expand pred to give shape: (batch, 1, ouput_size)
      next_pred_expanded = next_pred.unsqueeze(1)

      #  Update window, remove first observation and append new prediction
      window = torch.cat((window[:, 1:, :], next_pred_expanded), dim=1)

  return next_pred

def evaluate_model(model, test_dataset, device=None):

  if device is None:
    device = next(model.parameters()).device

  model = model.to(device)

  model.eval()

  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  preds_all = []
  target_all = []

  with torch.no_grad():
    for batch in test_loader:
      x_obs_batch = batch["lstm_input"].to(device)
      y_batch = batch["target"].to(device)


      preds = predict_autoregressive(model, x_obs_batch)

      preds_all.append(preds)
      target_all.append(y_batch)


    preds_all = torch.cat(preds_all, dim=0)
    target_all = torch.cat(target_all, dim=0)
  

  mse = nn.MSELoss()(preds_all, target_all)

  return mse.item()