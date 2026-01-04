import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import LSTMModel
#from tqdm import tqdm  

def train_model(train_dataset, hidden_size, num_layers, val_dataset=None, batch_size=32, learning_rate=0.01, num_epochs=50, patience=5, device='cpu', silence=False):
  # Determine input and output size
  input_size = train_dataset[0][0].shape[1]
  output_size = train_dataset[0][1].shape[0]

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  if val_dataset is not None:
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  model = LSTMModel(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=output_size).to(device)
  
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Early stopping variables
  best_val_loss = float("inf")
  early_stop_counter = 0

  for epoch in range(num_epochs):
    # Put model in training mode
    model.train()
    train_loss = 0

    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)

      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()

      train_loss += loss.item() * x_batch.size(0)
    
    train_loss /= len(train_loader.dataset)

    # Early Stopping

    val_loss=None

    model.eval()
    val_loss = 0
    with torch.no_grad():
      for x_batch, y_batch in val_loader:
          x_batch, y_batch = x_batch.to(device), y_batch.to(device)
          val_out  = model(x_batch)
          loss = criterion(val_out, y_batch)
          val_loss += loss.item() * x_batch.size(0)
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val_loss - 1e-4:
      best_val_loss = val_loss
      early_stop_counter = 0
      torch.save(model.state_dict(), "best_model.pt")
    else:
      early_stop_counter += 1
      if early_stop_counter >= patience:
        if not silence: print(f"Early stopping triggered at epoch: {epoch}")
        break

    model.load_state_dict(torch.load("best_model.pt"))

    # Print progress every 10 epochs
    if (not silence) and (epoch + 1) % 10 == 0:
      if val_loss is not None:
        print(f'Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
      else:
        print(f'Epoch [{epoch+1}/{num_epochs}], train_loss: {train_loss:.4f}')

  return model