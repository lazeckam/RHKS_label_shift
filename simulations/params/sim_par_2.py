import sys
sys.path.append("../")
from generation import *

generation_function_tmp = generate_sample_Nstd_Nstd_CC
gen_params = {
    'p': 5,
    'beta': 1
}

N = 100
n_target_seq = [100, 200, 300, 400, 500, 1000, 1500]
pi_target = 0.7
pi_target_name='07'