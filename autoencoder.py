import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class autoencoder(ABC):
    def __init__(self,no_of_features):
        self.model =  ps.SINDy(feature_names=["x"+str(i) for i in range(no_of_features)])
        self.no_of_features = no_of_features

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class sindy(autoencoder):
    def __init__(self, n,t):
        super().__init__(n)
        self.timestep = t
    def fit(self, x):
        self.model.fit(x,t=self.timestep)

    def predict(self, x):
        super().predict(x)
    def print(self):
        self.model.print()