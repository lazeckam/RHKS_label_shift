import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import numpy as np
import pandas as pd

robjects.r['source']('../../ratio_estimator/ratio/auxs_funcs.R')  

class estimator_Vaz():

    def __init__(self, 
                 X_target, X_source_positive, X_source_negative):
        
        # self.X_target = X_target
        # self.X_source_positive = X_source_positive
        # self.X_source_negative = X_source_negative

        self.n_target = X_target.shape[0]
        self.n_source_positive = X_source_positive.shape[0]
        self.n_source_negative = X_source_negative.shape[0]
        self.n_source = self.n_source_positive + self.n_source_negative
        self.d = X_target.shape[1]

        X_source = np.vstack((X_source_positive, X_source_negative))
        Y_source = np.concat((np.ones(X_source_positive.shape[0]), np.zeros(X_source_negative.shape[0])))

        df_source = pd.DataFrame(X_source)
        df_source.columns = ['x'+str(i) for i in range(self.d)]
        self.df_source = df_source

        df_target = pd.DataFrame(X_target)
        df_target.columns = ['x'+str(i) for i in range(self.d)]
        self.df_target = df_target
        
        # df_y_source = pd.DataFrame({'response': Y_source})
        self.Y_source = Y_source

    def run_g_kernel_leave(self):

        g_list = robjects.ListVector({
            'Vaz': robjects.ListVector({
                'func':  robjects.r['g.kernel.leave'], 
                'extra': robjects.ListVector({
                    'lambda': robjects.FloatVector([0.001]),
                    'kernel': robjects.r['gaussian.kernel']
            })
            })
        })

        quantification_prior_shift = robjects.globalenv['quantification.prior.shift']

        with localconverter(robjects.default_converter + pandas2ri.converter + numpy2ri.converter):
            results = quantification_prior_shift(self.df_source, self.Y_source, self.df_target, g_list)

        self.pi_Vaz = np.array(results)[0,0][1]
        self.mu_target = np.array(results)[0,0][2]
        self.mu_negative = np.array(results)[0,0][3]
        self.mu_positive = np.array(results)[0,0][4]
        self.var_negative = np.array(results)[0,0][5]
        self.var_positive = np.array(results)[0,0][6]
        self.var_Vaz_n = np.array(results)[0,0][7]
        self.var_Vaz = np.array(results)[0,0][7]*(self.n_source + self.n_target)

    def compute_basic_simulations(self):
        self.run_g_kernel_leave()
    