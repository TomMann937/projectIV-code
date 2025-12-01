from math import sin
from scipy import integrate as inte
import numpy as np

class LorenzModel:
  def __init__(self, c=30, sigma=10, beta=8/3, rho=28):
    self.c = c
    self.sigma = sigma
    self.beta = beta
    self.rho = rho

  def deriv(self, t, state):
    x, y, z = state
    dxdt = self.sigma * (y - self.c*sin(x/self.c))
    dydt = self.c*sin(x/self.c) * (self.rho - z) -y
    dzdt = self.c*sin(x/self.c) * y - self.beta * z
    return [dxdt, dydt, dzdt]
  
  def predict(self, y0, t_step, t_eval=None):

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
  
  def update_params(self, c=None, sigma=None, beta=None, rho=None):
      if c is not None:
          self.c = c
      if sigma is not None:
          self.sigma = sigma
      if beta is not None:
          self.beta = beta
      if rho is not None:
          self.rho = rho
