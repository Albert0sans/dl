from sklearn.base import BaseEstimator, RegressorMixin


class sklearn3dWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, base_model, target_shape):
            self.base_model = base_model
            self.target_shape = target_shape  # e.g., (1, 1)

        def fit(self, X, y):
            X=X.reshape(X.shape[0], -1) 
            y=y.reshape(y.shape[0], -1) .ravel()

            self.base_model.fit(X, y)
            return self
        def __call__(self, *args, **kwds):
             return self.predict(X=args[0])
        def predict(self, X):

            X=X.reshape(X.shape[0], -1) 

            y_pred = self.base_model.predict(X)
            return y_pred.reshape(X.shape[0], *self.target_shape)

    # Wrap your RandomForestRegressor
    