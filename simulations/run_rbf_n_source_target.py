import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("../estimators")
from estimators_RKHS import *

from params.sim_par_n_target import *
save_as = sys.argv[1]
pi_target = float(sys.argv[2])
params = set_params(save_as, pi_target)
locals().update(params)

iter = 0

df_results = pd.DataFrame(np.zeros((len(n_target_seq), 8)), 
                              columns=['n_plus', 'n_minus', 'n_target', 'pi_target', 'pi', 'var_n', 'var', 'seed'])
df_results['pi_target'] = pi_target

for n_target in n_target_seq:
    n_plus = int(n_target/2)
    n_minus = int(n_target/2)

    df_results['n_plus'] = n_plus
    df_results['n_minus'] = n_minus
    r_n = 1/n_plus + 1/n_minus + 1/n_target
    var = generate_sample_Nstd_Nstd_CC_rbf(p=gen_params['p'], beta=gen_params['beta'], n_plus=n_plus, n_minus=n_minus, 
                                           n_target=n_target, pi_target=pi_target, gamma=1/gen_params['p'])[0]
    

    df_results.loc[iter, 'n_target'] = n_target
    df_results.loc[iter, 'seed'] = None


    df_results.loc[iter, 'pi'] = None
    df_results.loc[iter, 'var_n'] =  var*r_n
    df_results.loc[iter, 'var'] =  var

    df_results.to_csv("./results/"+save_as+"/rbf_pi_target"+pi_target_name+".csv", index=False)

    iter += 1

