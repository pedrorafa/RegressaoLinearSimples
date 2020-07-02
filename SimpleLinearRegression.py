import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.formula = None
        self.X = None
        self.y = None
        self.predicts_ = None
        self.residuos_ = None
        self.sqt_ = None
        self.sqe_ = None
        self.sqr_ = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.X = X
        self.y = y
        soma_xy = sum(X * y)
        soma_x_ao_quadrado = sum(X * X)
        soma_x = sum(X)
        soma_y = sum(y)
        n = len(X)
        media_x = X.mean()
        media_y = y.mean()
        
        # build formula y = ax + b
        a = ( soma_xy - n * media_x * media_y ) / ( soma_x_ao_quadrado - n * ( media_x ** 2 ) )
        b = media_y - (a * media_x)
        
        self.coef_ = np.array([ b ])
        self.intercept_ = np.array([ a ])
        
        self.formula = lambda _x : (a * _x) + b
    
    def predict(self, x):
        return np.array(list(map(self.formula, x)))
    
    def test_model(self):
        i = 0
        self.predicts_ = []
        self.sqt_ = []
        self.sqe_ = []
        self.sqr_ = []
        
        while i < len(self.X):
            predicted = self.predict([self.X[i]])[0]
            sqt = ( self.y[i] - self.y.mean() ) ** 2
            sqe = ( self.y[i] - predicted ) ** 2 
            
            self.predicts_.append(predicted)
            self.sqt_.append( sqt )
            self.sqe_.append( sqe )
            self.sqr_.append( sqt - sqe )
            
            i += 1
            
    
    # fonte: https://edisciplinas.usp.br/pluginfile.php/1479289/mod_resource/content/0/regr_lin.pdf
    def sum_total_quadratic(self):
        median = self.y.mean()
        return sum( ( self.y - median ) ** 2 )
    
    def sum_error_quadratic(self):
        predicted = self.predict(x=self.X)
        return sum( ( self.y - predicted ) ** 2 )

    def regression_quadratic_sum(self):
        return self.sum_total_quadratic() - self.sum_error_quadratic()
    
    def score(self):
        return self.regression_quadratic_sum() / self.sum_total_quadratic()