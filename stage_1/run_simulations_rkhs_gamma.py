import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
import os

sys.path.append("../estimators")
from estimators_RKHS import *

from sim_par import *
save_as = sys.argv[1]
sim_scenario = save_as
pi_target = float(sys.argv[2])
params = set_params(sim_scenario, pi_target)
locals().update(params)

# method_names = ['one_over_p', 'one_over_p', 'one_over_p', 'one_over_p',
#                 'numerical', 'grid', 
#                 'bootstrap_var', 'bootstrap_mse',
#                 'distance', 'distance']

# estimator_type = ['nrm', 'nrm', 'ipr', 'ipr',
#                   'ipr', 'ipr',
#                   'ipr', 'ipr',
#                   'ipr', 'ipr']

# UorV_type = ['U', 'V', 'U', 'V',
#              'U', 'U',
#              'U', 'U',
#              'U', 'V']

# method_names = ['numerical', 'grid',
#                 'bootstrap_var', 'bootstrap_mse'] +  method_names

# estimator_type = ['ipr', 'ipr',
#                   'ipr', 'ipr'] + estimator_type

# UorV_type = ['V', 'V', 'V', 'V'] + UorV_type

method_names = ['one_over_p', 'numerical', 'grid', 
                'bootstrap_var', 'bootstrap_mse',
                'distance', 
                'one_over_p', 'numerical', 'grid', 
                'bootstrap_var', 'bootstrap_mse',
                'distance']
method_names = method_names + method_names

estimator_type = ['nrm', 'nrm', 'nrm', 'nrm','nrm', 'nrm',
                  'nrm', 'nrm', 'nrm', 'nrm','nrm', 'nrm']
estimator_type = estimator_type + ['ipr', 'ipr', 'ipr', 'ipr','ipr', 'ipr',
                                   'ipr', 'ipr','ipr', 'ipr','ipr', 'ipr']
UorV_type = ['U', 'U', 'U', 'U', 'U', 'U',
             'V', 'V', 'V', 'V', 'V', 'V']
UorV_type = UorV_type + UorV_type

kernel = 'laplacian'

for kernel in ['rbf', 'laplacian']:

    for i in tqdm(range(100)):
        for j, n_target in enumerate(n_target_seq):

            n_plus = n_plus_seq[j]
            n_minus = n_minus_seq[j]

            p_source_plus, p_source_minus, p_target = generation_function_tmp(**gen_params,
                                                                            n_plus=n_plus, n_minus=n_minus, n_target=n_target, 
                                                                            pi_target=pi_target, seed=i)
            results = []

            for k, method in enumerate(method_names):

                start = time.time()
                mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, 
                                            how=method, estimator=estimator_type[k], UorV_statistic=UorV_type[k], kernel=kernel)
                mod.compute_basic_simulations()
                end = time.time()

                if estimator_type[k] == 'nrm':
                    est_tmp = mod.pi_nrm
                if estimator_type[k] == 'ipr':
                    est_tmp = mod.pi_ipr

                results_tmp = {
                    'sim_scenario': save_as,
                    'n_plus': n_plus,
                    'n_minus': n_minus,
                    'n_target': n_target,
                    'pi_target': pi_target,
                    'seed': i,
                    'time' : end-start,
                    'gamma_selection_name': method,
                    'estimator': estimator_type[k],
                    'type': UorV_type[k],
                    'estimator_name': estimator_type[k]+'_'+UorV_type[k],
                    'kernel': kernel,
                    'pi': est_tmp,
                    'var_n': mod.var_plug_in_n,
                    'var': mod.var_plug_in,
                    'gamma': mod.gamma_opt
                }

                results.append(results_tmp)
            
            df_tmp = pd.DataFrame(results)
            filename = "./results_25_10_06/"+kernel+"/"+save_as+'_'+pi_target_name+".csv"
            if os.path.exists(filename):
                df_tmp.to_csv(filename, mode='a', index=False, header=False)
            else:
                df_tmp.to_csv(filename, mode='w', index=False, header=True)

