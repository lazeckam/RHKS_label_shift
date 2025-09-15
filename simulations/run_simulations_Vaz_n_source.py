import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("../estimators")
from estimators_Vaz import *

from params.sim_par_n_target import *
save_as = sys.argv[1]
pi_target = float(sys.argv[2])
params = set_params(save_as, pi_target)
locals().update(params)

iter = 0

df_results = pd.DataFrame(np.zeros((N*len(n_source_seq), 8)), 
                          columns=['n_plus', 'n_minus', 'n_target', 'pi_target', 'pi', 'var_n', 'var', 'seed'])

df_results['pi_target'] = pi_target

for i in tqdm(range(N)):
    for n_source in n_source_seq:

        n_plus = int(n_source/2)
        n_minus = int(n_source/2)


        df_results['n_plus'] = n_plus
        df_results['n_minus'] = n_minus

        p_source_plus, p_source_minus, p_target = generation_function_tmp(**gen_params,
                                                                          n_plus=n_plus, n_minus=n_minus, n_target=n_target, 
                                                                          pi_target=pi_target, seed=i)
        df_results.loc[iter, 'n_target'] = n_target
        df_results.loc[iter, 'seed'] = i

        mod = estimator_Vaz(p_target, p_source_plus, p_source_minus)
        mod.compute_basic_simulations()

        df_results.loc[iter, 'pi'] = mod.pi_Vaz
        df_results.loc[iter, 'var_n'] = mod.var_Vaz_n
        df_results.loc[iter, 'var'] = mod.var_Vaz

        df_results.to_csv("./results/"+save_as+"/Vaz_pi_target"+pi_target_name+".csv", index=False)

        iter += 1



