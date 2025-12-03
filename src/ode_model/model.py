from scipy import integrate as inte
import numpy as np
from decimal import Decimal

class LorenzModel:
  def __init__(self, c=30, sigma=10, beta=8/3, rho=28):
    self.c = c
    self.sigma = sigma
    self.beta = beta
    self.rho = rho

  def deriv(self, t, state):
    x, y, z = state
    dxdt = self.sigma * (y - self.c*np.sin(x/self.c))
    dydt = self.c*np.sin(x/self.c) * (self.rho - z) -y
    dzdt = self.c*np.sin(x/self.c) * y - self.beta * z
    return [dxdt, dydt, dzdt]
  
  def predict_scipy(self, y0, t_step, t_eval=None):

    single_input = False
    if y0.ndim == 1:
      y0 = y0[np.newaxis, :]  # make it (1, state_dim)
      single_input = True

    predictions = []

    for state in y0:
      sol = inte.solve_ivp(self.deriv, [0, t_step], state, t_eval=t_eval, method='RK45')
      predictions.append(sol.y[:, -1])  

    predictions = np.stack(predictions, axis=0)

    if single_input:
        return predictions[0]  
    return predictions
  
  def vec_deriv(self, state):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dxdt = self.sigma * (y - self.c * np.sin(x / self.c))
    dydt = self.c * np.sin(x / self.c) * (self.rho - z) - y
    dzdt = self.c * np.sin(x / self.c) * y - self.beta * z
    # Use axis=-1 as this handles the case when state is 1d nicely
    return np.stack([dxdt, dydt, dzdt], axis=-1)
  
  def predict(self, y0, t_step, dt=0.01):
    y = np.array(y0, copy=True)
    # Change y to (N, 3) if only 1 input given
    if y.ndim == 1:
      y = y[np.newaxis, :]

    if Decimal(str(t_step)) % Decimal(str(dt)) != Decimal("0.0"):
      raise ValueError(f"Time step: {t_step}, is not a multiple of dt: {dt}")
    
    steps = int(t_step / dt)

    for _ in range(steps):
      k1 = self.vec_deriv(y)
      k2 = self.vec_deriv(y + dt * k1 / 2)
      k3 = self.vec_deriv(y + dt * k2 / 2)
      k4 = self.vec_deriv(y + dt *k3)

      y = y + dt*(k1 + 2*k2 + 2*k3 + k4) / 6

    return y[0] if y0.ndim == 1 else y
  
  def update_params(self, c=None, sigma=None, beta=None, rho=None):
    if c is not None:
      self.c = c
    if sigma is not None:
      self.sigma = sigma
    if beta is not None:
      self.beta = beta
    if rho is not None:
      self.rho = rho
