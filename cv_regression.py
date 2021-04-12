from sklearn.linear_model import Ridge
import numpy as np

def regression_iter(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
    if validate:
        y_pred = regr.predict(x_test)
#         print(y_test.shape)
        r = correlation_iter(y_test, y_pred)
#         print(r)
        return weights,r
    else:
        return weights
    
def correlation_iter(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean(axis = 0)
    return r

