import sys
sys.path.append("../")
from generation import *

def set_params(sim_scenario, pi_target):

    if sim_scenario == 'sim_par_1':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_Nstd_CC,
            "gen_params": {
                "p": 5,
                "beta": 1
            },
            "n_target_seq": [100, 200, 300, 400, 500, 1000, 1500],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", ""),
            "n_plus": 150,
            "n_minus": 150
        }

    if sim_scenario == 'sim_par_2':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_Nstd_CC,
            "gen_params": {
                "p": 5,
                "beta": 0.5
            },
            "n_target_seq": [100, 200, 300, 400, 500, 1000, 1500],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", ""),
            "n_plus": 150,
            "n_minus": 150
        }

    if sim_scenario == 'sim_par_3':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_AR1_CC,
            "gen_params": {
                "p": 5,
                "beta": 1,
                'rho': 0.5
            },
            "n_target_seq": [100, 200, 300, 400, 500, 1000, 1500],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", ""),
            "n_plus": 150,
            "n_minus": 150
        }

    return params