import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("../estimators")
from estimators_AC import *

from params.sim_par_n_target import *
save_as = sys.argv[1]
pi_target = float(sys.argv[2])
params = set_params(save_as, pi_target)
locals().update(params)

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

        p_source_plus, p_source_minus, p_target = generation_function_tmp(**gen_params,
                                                                          n_plus=n_plus, n_minus=n_minus, n_target=n_target, 
                                                                          pi_target=pi_target, seed=i)
        df_results_ipr.loc[iter, 'n_target'] = n_target
        df_results_nrm.loc[iter, 'n_target'] = n_target
        df_results_ipr.loc[iter, 'n_plus'] = n_plus
        df_results_ipr.loc[iter, 'n_minus'] = n_minus
        df_results_nrm.loc[iter, 'n_plus'] = n_plus
        df_results_nrm.loc[iter, 'n_minus'] = n_minus
        df_results_ipr.loc[iter, 'seed'] = i
        df_results_nrm.loc[iter, 'seed'] = i

        mod = estimator_AC(p_target, p_source_plus, p_source_minus, g ='LogisticRegression')
        mod.compute_basic_simulations()

        df_results_ipr.loc[iter, 'pi'] = mod.pi_Bella
        df_results_nrm.loc[iter, 'pi'] =  mod.pi_Forman
        df_results_ipr.loc[iter, 'var_n'] = mod.var_Bella_n
        df_results_nrm.loc[iter, 'var_n'] =  mod.var_Forman_n
        df_results_ipr.loc[iter, 'var'] = mod.var_Bella
        df_results_nrm.loc[iter, 'var'] =  mod.var_Forman

        df_results_ipr.to_csv("./results/"+save_as+"/Bella_logreg_pi_target"+pi_target_name+".csv", index=False)
        df_results_nrm.to_csv("./results/"+save_as+"/Forman_logreg_pi_target"+pi_target_name+".csv", index=False)

        iter += 1

