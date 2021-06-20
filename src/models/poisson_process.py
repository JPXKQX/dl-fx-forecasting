from sklearn.base import BaseEstimator
from pydantic.dataclasses import dataclass


@dataclass
class PoissonProcess(BaseEstimator):
    rate: float
    
    def fit(self, X, y):
        
        pass

    def predict(self, X):
        
        pass

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "recursive": self.recursive}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self