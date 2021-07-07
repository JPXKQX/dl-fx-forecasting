from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler


class LinearRegr(LinearRegression):
    def __init__(self, *args, **kwargs):
        super(LinearRegr, self).__init__(*args, **kwargs)
        self.scale = StandardScaler()

    def fit(self, X, y=None, sample_weight=None):
        X = self.scale.fit_transform(X)
        return super(LinearRegr, self).fit(X, y, sample_weight)

    def predict(self, X):
        X = self.scale.transform(X)
        return super(LinearRegr, self).predict(X)


class ElasticNetRegr(ElasticNet):
    def __init__(self, *args, **kwargs):
        super(ElasticNetRegr, self).__init__(*args, **kwargs)
        self.scale = StandardScaler()

    def fit(self, X, y=None, sample_weight=None):
        X = self.scale.fit_transform(X)
        return super(ElasticNetRegr, self).fit(X, y, sample_weight)

    def predict(self, X):
        X = self.scale.transform(X)
        return super(ElasticNetRegr, self).predict(X)
