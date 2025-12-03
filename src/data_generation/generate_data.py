import numpy as np
from scipy import integrate as inte

def derivative(t, state, sigma=10, beta=8/3, rho=28):
  x, y, z = state
  dxdt = sigma * (y - x)
  dydt = x * (rho - z) - y
  dzdt = x * y - beta * z
  return [dxdt, dydt, dzdt]

def generate_data(length: int, noise_level: float = 0, time_step: float = 0.05, trunc=True, init: list[float] = [1,1,1]) -> np.ndarray:
  """
  Generate synthetic data from the Lorenx-63 equations.

  Args:
    length (int): Length of array to be returned.
    noise_level (float): Amount of noise to be added to data as a propotion of SD.
    time_step (float): Time step inbetween observation.
    trunc (bool): Should burn in period be removed.
    init list[float]: List of point to start simulation from.

  Returns:
    np.ndarray: An array of points. 
  """

  t = time_step * length
  if trunc: t += 100

  t_eval = np.arange(0, t, time_step)

  # Solve using RK45
  solution = inte.solve_ivp(derivative, (0, t), init, t_eval=t_eval, max_step=0.01)

  if not solution.success: 
    print(f"Integration failed, {solution.message}")
    return None
  
  y = np.array(solution.y)

  # Take last observations, i.e. removing the burn in period
  y = y[:, -length:]

  # Add noise proportional to variance in each dimension
  if noise_level > 0:
    std_dev = y.std(axis=0, ddof=1)
    noise = np.random.normal(0, noise_level * std_dev, size = y.shape)
    y = y + noise

  #  Transpose to return shape (time_steps, num_features)
  y = y.T

  return y


  





  




