import numpy as np
from abc import ABC, abstractmethod

class Simulator(ABC):
    """
    Abstract class for a simulator.
    """

    @abstractmethod
    def simulate(self):
        """
        Simulate the system.

        """
        pass
class lorenz(Simulator):
    def __init__(self,iteration, dt,sigma=10, rho=28, beta=2.667):
        self.dt = dt
        self.iteration = iteration
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.xs = np.empty(iteration + 1)
        self.ys = np.empty(iteration + 1)
        self.zs = np.empty(iteration + 1)

    def lorenz_calc(self,x, y, z):
        x_dot = self.sigma*(y - x)
        y_dot = self.rho*x - y - x*z
        z_dot = x*y - self.beta*z
        return x_dot, y_dot, z_dot
    
    def get_equation(self):
        return lambda t,z : [self.sigma*(z[1] - z[0]),z[0]*(self.rho - z[2]) - z[1], z[0]*z[1] - self*z[2]]
    
    def simulate(self, x0, y0, z0):
        self.xs[0] = x0
        self.ys[0] = y0
        self.zs[0] = z0
        for i in range(self.iteration):
            x_dot, y_dot, z_dot = self.lorenz_calc(self.xs[i], self.ys[i], self.zs[i])
            self.xs[i+1] = self.xs[i] + (x_dot * self.dt)
            self.ys[i+1] = self.ys[i] + (y_dot * self.dt)
            self.zs[i+1] = self.zs[i] + (z_dot * self.dt)
        return self.xs, self.ys, self.zs
    
    def reset(self):
        self.xs = np.empty(self.iteration + 1)
        self.ys = np.empty(self.iteration + 1)
        self.zs = np.empty(self.iteration + 1)