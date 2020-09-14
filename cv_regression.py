from sklearn.linear_model import Ridge
import numpy as np

def regression_iter(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
    if validate:
        y_pred = regr.predict(x_test)
#         print(y_test.shape)
        r = manual_pearson(y_test, y_pred)
#         print(r)
        return weights,r
    else:
        return weights
    
def correlation_iter(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean(axis = 0)
    return r


def manual_pearson(a,b):
# """
# Accepts two arrays of equal length, and computes correlation coefficient. 
# Numerator is the sum of product of (a - a_avg) and (b - b_avg), 
# while denominator is the product of a_std and b_std multiplied by 
# length of array. 
# """

#remove columns that only have zeros since they will give error/nan
#     print(a.shape)
#     ind_nan = ~np.all((a == 0), axis=0) or ~np.all((b == 0), axis=0) 
# #     print(ind_nan.shape)
#     a = a[:,ind_nan]
#     b = b[:,ind_nan]
    
    a_avg, b_avg = a.mean(0), b.mean(0)
    a_stdev, b_stdev = a.std(0), b.std(0)
    n = len(a)
    denominator = a_stdev * b_stdev * n
#     denominator =  b_stdev * n
    numerator = np.sum(np.multiply(a-a_avg, b-b_avg),axis=0)
    p_coef = numerator/denominator
    return p_coef