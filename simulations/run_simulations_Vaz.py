import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("../estimators")
from estimators_Vaz import *

from params.sim_par_1 import *
save_as = 'sim_par_1'

N = 100
n_target_seq = [100, 200, 300, 400, 500, 1000, 1500]
pi_target = 0.7
iter = 0

df_results = pd.DataFrame(np.zeros((N*len(n_target_seq), 8)), 
                              columns=['n_plus', 'n_minus', 'n_target', 'pi_target', 'pi', 'var_n', 'var', 'seed'])

df_results['n_plus'] = 150
df_results['n_minus'] = 150
df_results['pi_target'] = pi_target

for i in tqdm(range(N)):
    for n_target in n_target_seq:

        p_source_plus, p_source_minus, p_target = generation_function_tmp(**gen_params,
                                                                          n_plus=150, n_minus=150, n_target=n_target, 
                                                                          pi_target=pi_target, seed=i)
        df_results.loc[iter, 'n_target'] = n_target
        df_results.loc[iter, 'seed'] = i

        mod = estimator_Vaz(p_target, p_source_plus, p_source_minus)
        mod.compute_basic_simulations()

        df_results.loc[iter, 'pi'] = mod.pi_Vaz
        df_results.loc[iter, 'var_n'] = mod.var_Vaz_n
        df_results.loc[iter, 'var'] = mod.var_Vaz

        df_results.to_csv("./results/"+save_as+"/Vaz_pi_target07.csv", index=False)

        iter += 1



