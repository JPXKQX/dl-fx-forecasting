from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler


class LinearRegr(LinearRegression):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None,
                 positive=False):
        super(LinearRegr, self).__init__()
        self.scale = StandardScaler()

    def fit(self, X, y=None, sample_weight=None):
        X = self.scale.fit_transform(X)
        return super(LinearRegr, self).fit(X, y, sample_weight)

    def predict(self, X):
        X = self.scale.transform(X)
        return super(LinearRegr, self).predict(X)


class ElasticNetRegr(ElasticNet):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super(ElasticNetRegr, self).__init__(alpha=alpha, l1_ratio=l1_ratio)
        self.scale = StandardScaler()

    def fit(self, X, y=None, sample_weight=None):
        X = self.scale.fit_transform(X)
        return super(ElasticNetRegr, self).fit(X, y, sample_weight)

    def predict(self, X):
        X = self.scale.transform(X)
        return super(ElasticNetRegr, self).predict(X)
