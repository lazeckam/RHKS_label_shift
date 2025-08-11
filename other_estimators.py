
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class estimator_AC():

    def __init__(self, 
                 X_source_positive, X_source_negative, X_target,
                 g='RandomForestClassifier', g_params=None):
        
        self.X_target = X_target
        self.X_source_positive = X_source_positive
        self.X_source_negative = X_source_negative

        self.g = g
        self._set_g_params(g_params)

        self.n_target = X_target.shape[0]
        self.n_source_positive = X_source_positive.shape[0]
        self.n_source_negative = X_source_negative.shape[0]
        self.n_source = self.n_source_positive + self.n_source_negative

        self.pi_ac = None

        # self.compute_lambdas()

    def _set_g_params(self, g_params):
        # matching default params used in R and params from the article [1]
        #  - randomForest from library(randomForest)
        #  - glm(formula, family = binomial, data = ...)
        if g_params is None:
            if self.g == 'RandomForestClassifier':
                self.g_params = {
                    'n_estimators': 100,
                    'max_features': 'sqrt'
                }
            if self.g == 'LogisticRegression':
                self.g_params = {
                    'penalty': None,
                    'solver': 'lbfgs', 
                    'max_iter': 25,
                    'tol': 1e-8,
                    'fit_intercept': True
                }

    def fit_and_predict(self):

        if self.g == 'RandomForestClassifier':
            model = RandomForestClassifier(**self.g_params)
        if self.g == 'LogisticRegression':
            model = LogisticRegression(**self.g_params)

        self.y_predicted_source_positive = np.zeros(self.n_source_positive)
        self.y_proba_predicted_source_positive = np.zeros(self.n_source_positive)
        for i_plus in range(self.n_source_positive):
            X_source = np.vstack((np.delete(self.X_source_positive, i_plus, axis=0), self.X_source_negative))
            Y_source = np.concat((np.ones(self.n_source_positive-1), np.zeros(self.n_source_negative)))
            model.fit(X_source, Y_source)
            self.y_predicted_source_positive[i_plus] = model.predict(self.X_source_positive[i_plus,].reshape(1, -1))[0]
            self.y_proba_predicted_source_positive[i_plus] = model.predict_proba(self.X_source_positive[i_plus,].reshape(1, -1))[0][1]
        
        self.y_predicted_source_negative = np.zeros(self.n_source_negative)
        self.y_proba_predicted_source_negative = np.zeros(self.n_source_negative)
        for i_minus in range(self.n_source_negative):
            X_source = np.vstack((self.X_source_positive, np.delete(self.X_source_negative, i_minus, axis=0)))
            Y_source = np.concat((np.ones(self.n_source_positive), np.zeros(self.n_source_negative-1)))
            model.fit(X_source, Y_source)
            self.y_predicted_source_negative[i_minus] = model.predict(self.X_source_negative[i_minus,].reshape(1, -1))[0]
            self.y_proba_predicted_source_negative[i_minus] = model.predict_proba(self.X_source_negative[i_minus,].reshape(1, -1))[0][1]

        X_source = np.vstack((self.X_source_positive, self.X_source_negative))
        Y_source = np.concat((np.ones(self.n_source_positive), np.zeros(self.n_source_negative)))
        model.fit(X_source, Y_source)
        self.y_predicted_target = model.predict(self.X_target)
        self.y_proba_predicted_target = model.predict_proba(self.X_target)[:,1]
    
    def compute_mu_Bella(self):

        self.mu_target_Bella = np.mean(self.y_predicted_target)
        self.mu_plus_Bella = np.mean(self.y_predicted_source_positive)
        self.mu_minus_Bella = np.mean(self.y_predicted_source_negative)

    def compute_mu_Forman(self):

        self.mu_target_Forman = np.mean(self.y_proba_predicted_target)
        self.mu_plus_Forman = np.mean(self.y_proba_predicted_source_positive)
        self.mu_minus_Forman = np.mean(self.y_proba_predicted_source_negative)

    def compute_var_Bella(self):

        # self.var_target_Bella = np.var(self.y_predicted_target, ddof=1)
        self.var_plus_Bella = np.var(self.y_predicted_source_positive, ddof=1)
        self.var_minus_Bella = np.var(self.y_predicted_source_negative, ddof=1)

    def compute_var_Forman(self):

        # self.var_target_Forman = np.var(self.y_proba_predicted_target, ddof=1)
        self.var_plus_Forman = np.var(self.y_proba_predicted_source_positive, ddof=1)
        self.var_minus_Forman = np.var(self.y_proba_predicted_source_negative, ddof=1)
    
    def estimate_pi_Bella(self):

        self.pi_Bella = np.clip(a=(self.mu_target_Bella - self.mu_minus_Bella)/(self.mu_plus_Bella - self.mu_minus_Bella),
                                a_min=0, a_max=1)
        
    def estimate_pi_Forman(self):

        self.pi_Forman = np.clip(a=(self.mu_target_Forman - self.mu_minus_Forman)/(self.mu_plus_Forman - self.mu_minus_Forman),
                                a_min=0, a_max=1)
        
    def estimate_variance_Bella(self):

        temp1 = 1/(self.mu_plus_Bella - self.mu_minus_Bella)**2
        temp2 = ((1/temp1)*self.pi_Bella*(1-self.pi_Bella) +self.var_plus_Bella*self.pi_Bella \
                 + self.var_minus_Bella*(1-self.pi_Bella))/self.n_target
        temp3 = (self.var_minus_Bella*((1-self.pi_Bella)**2))/self.n_source_negative\
              + (self.var_plus_Bella*(self.pi_Bella**2))/self.n_source_positive

        self.var_Bella = temp1*(temp2 + temp3)

    def estimate_variance_Forman(self):

        temp1 = 1/(self.mu_plus_Forman - self.mu_minus_Forman)**2
        temp2 = ((1/temp1)*self.pi_Forman*(1-self.pi_Forman) +self.var_plus_Forman*self.pi_Forman \
                 + self.var_minus_Forman*(1-self.pi_Forman))/self.n_target
        temp3 = (self.var_minus_Forman*((1-self.pi_Forman)**2))/self.n_source_negative\
              + (self.var_plus_Forman*(self.pi_Forman**2))/self.n_source_positive

        self.var_Forman = temp1*(temp2 + temp3)


    # def compute_lambdas(self):

    #     self.h_n = 1/self.n_target + 1/self.n_source_positive + 1/self.n_source_negative

    #     r_n_inv = 1/self.r_n

    #     self.lambda_target= r_n_inv/self.n_target
    #     self.lambda_source_positive = r_n_inv/self.n_source_positive
    #     self.lambda_source_negative = r_n_inv/self.n_source_negative

    #     self._is_lambda_computed = True

    def estimate_pi_ac(self):

        self.pi_ac = 0