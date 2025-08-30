import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("../estimators")
from estimators_AC_Vaz import *

from params.sim_par_n_target import *
save_as = sys.argv[0]
pi_target = sys.argv[1]
params = set_params(save_as, pi_target)
locals().update(params)

iter = 0

df_results = pd.DataFrame(np.zeros((N*len(n_target_seq), 8)), 
                          columns=['n_plus', 'n_minus', 'n_target', 'pi_target', 'pi', 'var_n', 'var', 'seed'])

df_results['n_plus'] = n_plus
df_results['n_minus'] = n_minus
df_results['pi_target'] = pi_target

for i in tqdm(range(N)):
    for n_target in n_target_seq:

        p_source_plus, p_source_minus, p_target = generation_function_tmp(**gen_params,
                                                                          n_plus=n_plus, n_minus=n_minus, n_target=n_target, 
                                                                          pi_target=pi_target, seed=i)
        df_results.loc[iter, 'n_target'] = n_target
        df_results.loc[iter, 'seed'] = i

        mod = estimator_AC_Vaz(p_target, p_source_plus, p_source_minus, g='g.bella.logistic')
        mod.compute_basic_simulations()

        df_results.loc[iter, 'pi'] = mod.pi_Vaz
        df_results.loc[iter, 'var_n'] = mod.var_Vaz_n
        df_results.loc[iter, 'var'] = mod.var_Vaz

        df_results.to_csv("./results/"+save_as+"/Bella_logreg_Vaz_pi_target"+pi_target_name+".csv", index=False)

        iter += 1



iter = 0

df_results = pd.DataFrame(np.zeros((N*len(n_target_seq), 8)), 
                          columns=['n_plus', 'n_minus', 'n_target', 'pi_target', 'pi', 'var_n', 'var', 'seed'])

df_results['n_plus'] = n_plus
df_results['n_minus'] = n_minus
df_results['pi_target'] = pi_target

for i in tqdm(range(N)):
    for n_target in n_target_seq:

        p_source_plus, p_source_minus, p_target = generation_function_tmp(**gen_params,
                                                                          n_plus=n_plus, n_minus=n_minus, n_target=n_target, 
                                                                          pi_target=pi_target, seed=i)
        df_results.loc[iter, 'n_target'] = n_target
        df_results.loc[iter, 'seed'] = i

        mod = estimator_AC_Vaz(p_target, p_source_plus, p_source_minus, g='g.forman.logistic')
        mod.compute_basic_simulations()

        df_results.loc[iter, 'pi'] = mod.pi_Vaz
        df_results.loc[iter, 'var_n'] = mod.var_Vaz_n
        df_results.loc[iter, 'var'] = mod.var_Vaz

        df_results.to_csv("./results/"+save_as+"/Forman_logreg_Vaz_pi_target"+pi_target_name+".csv", index=False)

        iter += 1