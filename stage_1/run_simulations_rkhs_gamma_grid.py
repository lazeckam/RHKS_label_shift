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

method_names = ['grid', 'bootstrap_var', 'bootstrap_mse', 'distance']

# UorV_type = 'U'
# kernel = 'rbf'

# np.logspace(-6, -0.1, num=100)
gamma_seq = np.logspace(-6, np.log10(4*(1/gen_params['p'])), num=100)

for kernel in ['rbf', 'laplacian']:
    for UorV_type in ['U', 'V']:

        for i in tqdm(range(100)):
            for j, n_target in enumerate(n_target_seq):
                if (j == 0) or (j==2) or (j==4):

                    n_plus = n_plus_seq[j]
                    n_minus = n_minus_seq[j]

                    p_source_plus, p_source_minus, p_target = generation_function_tmp(**gen_params,
                                                                                    n_plus=n_plus, n_minus=n_minus, n_target=n_target, 
                                                                                    pi_target=pi_target, seed=i)
                    results_estimators = []
                    for gamma in gamma_seq:
                        
                        mod = estimator_RHKS(p_target, p_source_plus, p_source_minus, kernel=kernel, UorV_statistic=UorV_type, kernel_params={'gamma': gamma})
                        mod.compute_basic_simulations()
                        var_rbf_tmp = generation_function_tmp_rbf(**gen_params, n_plus=n_plus, n_minus=n_minus, n_target=n_target, 
                                                                pi_target=pi_target, gamma=gamma)[0]*mod.r_n

                        results_estimators_tmp = {
                            'sim_scenario': save_as,
                            'n_plus': n_plus,
                            'n_minus': n_minus,
                            'n_target': n_target,
                            'pi_target': pi_target,
                            'seed': i,
                            'pi': mod.pi_ipr,
                            'estimator': 'ipr',
                            'type': UorV_type,
                            'kernel': kernel,
                            'var_n': mod.var_plug_in_n,
                            'var': mod.var_plug_in,
                            'gamma': gamma,
                            'var_rbf': var_rbf_tmp
                        }
                        results_estimators.append(results_estimators_tmp)
                        results_estimators_tmp = {
                            'sim_scenario': save_as,
                            'n_plus': n_plus,
                            'n_minus': n_minus,
                            'n_target': n_target,
                            'pi_target': pi_target,
                            'seed': i,
                            'pi': mod.pi_nrm,
                            'estimator': 'nrm',
                            'type': UorV_type,
                            'kernel': kernel,
                            'var_n': mod.var_plug_in_n,
                            'var': mod.var_plug_in,
                            'gamma': gamma,
                            'var_rbf': var_rbf_tmp
                        }
                        results_estimators.append(results_estimators_tmp)

                    results_estimators = pd.DataFrame(results_estimators)
                    filename = "./results_grid/df_estimators_"+save_as+'_'+pi_target_name+".csv"
                    if os.path.exists(filename):
                        results_estimators.to_csv(filename, mode='a', index=False, header=False)
                    else:
                        results_estimators.to_csv(filename, mode='w', index=False, header=True)

                    results_gamma = []

                    # ONE OVER P
                    mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='one_over_p', kernel=kernel, 
                                                estimator='ipr', UorV_statistic=UorV_type)
                    mod.compute_basic_simulations()
                    results_gamma_tmp = {
                            'sim_scenario': save_as,
                            'n_plus': n_plus,
                            'n_minus': n_minus,
                            'n_target': n_target,
                            'pi_target': pi_target,
                            'seed': i,
                            'gamma_selection_name': 'one_over_p',
                            'pi': mod.pi_ipr,
                            'estimator': 'ipr',
                            'type': UorV_type,
                            'kernel': kernel,
                            'gamma_optimal': mod.gamma_opt
                        }
                    results_gamma.append(results_gamma_tmp)
                    mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='one_over_p', kernel=kernel, 
                                                estimator='nrm', UorV_statistic=UorV_type)
                    mod.compute_basic_simulations()
                    results_gamma_tmp = {
                            'sim_scenario': save_as,
                            'n_plus': n_plus,
                            'n_minus': n_minus,
                            'n_target': n_target,
                            'pi_target': pi_target,
                            'seed': i,
                            'gamma_selection_name': 'one_over_p',
                            'pi': mod.pi_nrm,
                            'estimator': 'nrm',
                            'type': UorV_type,
                            'kernel': kernel,
                            'gamma_optimal': mod.gamma_opt
                        }
                    results_gamma.append(results_gamma_tmp)

                    # NUMERICAL
                    mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='numerical', kernel=kernel, 
                                                estimator='ipr', UorV_statistic=UorV_type)
                    mod.compute_basic_simulations()
                    results_gamma_tmp = {
                            'sim_scenario': save_as,
                            'n_plus': n_plus,
                            'n_minus': n_minus,
                            'n_target': n_target,
                            'pi_target': pi_target,
                            'seed': i,
                            'gamma_selection_name': 'numerical',
                            'pi': mod.pi_ipr,
                            'estimator': 'ipr',
                            'type': UorV_type,
                            'kernel': kernel,
                            'gamma_optimal': mod.gamma_opt
                        }
                    results_gamma.append(results_gamma_tmp)
                    mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='numerical', kernel=kernel, 
                                                estimator='nrm', UorV_statistic=UorV_type)
                    mod.compute_basic_simulations()
                    results_gamma_tmp = {
                            'sim_scenario': save_as,
                            'n_plus': n_plus,
                            'n_minus': n_minus,
                            'n_target': n_target,
                            'pi_target': pi_target,
                            'seed': i,
                            'gamma_selection_name': 'numerical',
                            'pi': mod.pi_nrm,
                            'estimator': 'nrm',
                            'type': UorV_type,
                            'kernel': kernel,
                            'gamma_optimal': mod.gamma_opt
                        }
                    results_gamma.append(results_gamma_tmp)

                    # ON A GRID
                    results_gamma_grid = []
                    for method in method_names:
                        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how=method, kernel=kernel, 
                                                estimator='ipr', UorV_statistic=UorV_type)
                        mod.compute_basic_simulations()
                        results_gamma_tmp = {
                                'sim_scenario': save_as,
                                'n_plus': n_plus,
                                'n_minus': n_minus,
                                'n_target': n_target,
                                'pi_target': pi_target,
                                'seed': i,
                                'gamma_selection_name': method,
                                'pi': mod.pi_ipr,
                                'estimator': 'ipr',
                                'type': UorV_type,
                                'kernel': kernel,
                                'gamma_optimal': mod.gamma_opt
                            }
                        results_gamma.append(results_gamma_tmp)
                        results_gamma_grid_tmp = {
                                **mod.opt_values,
                                'sim_scenario': save_as,
                                'n_plus': n_plus,
                                'n_minus': n_minus,
                                'n_target': n_target,
                                'pi_target': pi_target,
                                'seed': i,
                                'gamma_selection_name': method,
                                'pi': mod.pi_ipr,
                                'estimator': 'ipr',
                                'type': UorV_type,
                                'kernel': kernel
                            }
                        results_gamma_grid.append(results_gamma_grid_tmp)
                        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how=method, kernel=kernel, 
                                                    estimator='nrm', UorV_statistic=UorV_type)
                        mod.compute_basic_simulations()
                        results_gamma_tmp = {
                                'sim_scenario': save_as,
                                'n_plus': n_plus,
                                'n_minus': n_minus,
                                'n_target': n_target,
                                'pi_target': pi_target,
                                'seed': i,
                                'gamma_selection_name': method,
                                'pi': mod.pi_nrm,
                                'estimator': 'nrm',
                                'type': UorV_type,
                                'kernel': kernel,
                                'gamma_optimal': mod.gamma_opt
                            }
                        results_gamma.append(results_gamma_tmp)
                        results_gamma_grid_tmp = {
                                **mod.opt_values,
                                'sim_scenario': save_as,
                                'n_plus': n_plus,
                                'n_minus': n_minus,
                                'n_target': n_target,
                                'pi_target': pi_target,
                                'seed': i,
                                'gamma_selection_name': method,
                                'pi': mod.pi_ipr,
                                'estimator': 'ipr',
                                'type': UorV_type,
                                'kernel': kernel
                            }
                        results_gamma_grid.append(results_gamma_grid_tmp)

                    results_gamma = pd.DataFrame(results_gamma)
                    filename = "./results_grid/df_gamma_selection_"+save_as+'_'+pi_target_name+".csv"
                    if os.path.exists(filename):
                        results_gamma.to_csv(filename, mode='a', index=False, header=False)
                    else:
                        results_gamma.to_csv(filename, mode='w', index=False, header=True)

                    results_gamma_grid = pd.DataFrame(results_gamma_grid)
                    filename = "./results_grid/df_gamma_grid_"+save_as+'_'+pi_target_name+".csv"
                    if os.path.exists(filename):
                        results_gamma_grid.to_csv(filename, mode='a', index=False, header=False)
                    else:
                        results_gamma_grid.to_csv(filename, mode='w', index=False, header=True)



