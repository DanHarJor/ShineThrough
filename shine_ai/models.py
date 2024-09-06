import numpy as np
class Exponential():
    def __init__(self, experiment_label):
        all_coefficients = {'Hplasma_HNBI':{'alpha':1.137, 'beta':0.167, 'gamma':0.818, 'D':6.07e-3}}
        self.coefficients = all_coefficients[experiment_label]


    def train(self, x, y):
        return NotImplemented
    
    def fit(self, *args, **kargs):
        self.train(*args, **kargs)
    
    def tune_hypers(self, x, y):
        raise NotImplemented
    
    def fun(self, n_avg, n_peak, E_nbi, OnOffAxis):
        n_avg = n_avg*1e-19
        a = self.coefficients['alpha']
        b = self.coefficients['beta']
        g = self.coefficients['gamma']
        d = self.coefficients['D']

        logneglog = a*np.log(n_avg) + b*np.log(n_peak) + g*np.log(E_nbi) - np.log(d)

        shine = np.exp(-np.exp(logneglog))

        return shine#np.exp((n_avg**self.coefficients['alpha'] * n_peak**self.coefficients['beta'] * E_nbi**self.coefficients['gamma'])/self.coefficients['D'])

    def predict(self, x):
        # print('x', x)
        y_predict = []
        for xi in x:
            # print(xi) 
            y_predict.append(self.fun(*xi))
            print('ans',xi, self.fun(*xi))
            
        # print('y', y_predict)
        return y_predict
    