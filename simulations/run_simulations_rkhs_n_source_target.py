import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("../estimators")
from estimators_RKHS import *

from params.sim_par_2 import *
save_as = 'sim_par_2'

iter = 0

df_results_ipr = pd.DataFrame(np.zeros((N*len(n_target_seq), 8)), 
                              columns=['n_plus', 'n_minus', 'n_target', 'pi_target', 'pi', 'var_n', 'var', 'seed'])
df_results_nrm = pd.DataFrame(np.zeros((N*len(n_target_seq), 8)), 
                              columns=['n_plus', 'n_minus', 'n_target', 'pi_target', 'pi', 'var_n', 'var', 'seed'])

df_results_ipr['pi_target'] = pi_target
df_results_nrm['pi_target'] = pi_target

for i in tqdm(range(N)):
    for n_target in n_target_seq:

        n_plus = int(n_target/2)
        n_minus = int(n_target/2)

        df_results_ipr['n_plus'] = n_plus
        df_results_ipr['n_minus'] = n_minus
        df_results_nrm['n_plus'] = n_plus
        df_results_nrm['n_minus'] = n_minus

        p_source_plus, p_source_minus, p_target = generation_function_tmp(**gen_params,
                                                                          n_plus=n_plus, n_minus=n_minus, n_target=n_target, 
                                                                          pi_target=pi_target, seed=i)
        df_results_ipr.loc[iter, 'n_target'] = n_target
        df_results_nrm.loc[iter, 'n_target'] = n_target
        df_results_ipr.loc[iter, 'seed'] = i
        df_results_nrm.loc[iter, 'seed'] = i

        mod = estimator_RHKS(p_target, p_source_plus, p_source_minus)
        mod.compute_basic_simulations()

        df_results_ipr.loc[iter, 'pi'] = mod.pi_ipr
        df_results_nrm.loc[iter, 'pi'] =  mod.pi_nrm
        df_results_ipr.loc[iter, 'var_n'] = mod.var_plug_in_n
        df_results_nrm.loc[iter, 'var_n'] =  mod.var_plug_in_n
        df_results_ipr.loc[iter, 'var'] = mod.var_plug_in
        df_results_nrm.loc[iter, 'var'] =  mod.var_plug_in

        df_results_ipr.to_csv("./results/"+save_as+"/ipr_pi_target"+pi_target_name+".csv", index=False)
        df_results_nrm.to_csv("./results/"+save_as+"/nrm_pi_target"+pi_target_name+".csv", index=False)

        iter += 1

