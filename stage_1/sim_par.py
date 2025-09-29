import sys
sys.path.append("../simulations")
from generation import *

def set_params(sim_scenario, pi_target):

    list_of_strings = sim_scenario.split('-')
    p_change = None
    generation_function_change = None
    if len(list_of_strings) > 1:
        sim_scenario = list_of_strings[0]
        if list_of_strings[1] == 'p':
            p_change = int(list_of_strings[2])
        if list_of_strings[1] == 'Cauchy':
           generation_function_change = generate_sample_Cauchy_Cauchy_CC,

    if sim_scenario == 'sim_par_1':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_Nstd_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_Nstd_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 1
            },
            "n_target_seq": [100, 200, 300, 400, 500, 1000, 1500],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", ""),
            "n_plus_seq": [150]*7,
            "n_minus_seq": [150]*7
        }

    if sim_scenario == 'sim_par_2':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_Nstd_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_Nstd_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 0.5
            },
            "n_target_seq": [100, 200, 300, 400, 500, 1000, 1500],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", ""),
            "n_plus_seq": [150]*7,
            "n_minus_seq": [150]*7
        }

    if sim_scenario == 'sim_par_3':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_AR1_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_AR1_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 1,
                'rho': 0.5
            },
            "n_target_seq": [100, 200, 300, 400, 500, 1000, 1500],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", ""),
            "n_plus_seq": [150]*7,
            "n_minus_seq": [150]*7
        }

    if sim_scenario == 'sim_par_4':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_Nstd_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_Nstd_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 1
            },
            "n_target_seq": [100, 200, 300, 400, 500],
            "n_plus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "n_minus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", "")
        }

    if sim_scenario == 'sim_par_5':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_Nstd_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_Nstd_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 0.5
            },
            "n_target_seq": [100, 200, 300, 400, 500],
            "n_plus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "n_minus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", "")
        }

    if sim_scenario == 'sim_par_6':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_AR1_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_AR1_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 1,
                'rho': 0.5
            },
            "n_target_seq": [100, 200, 300, 400, 500],
            "n_plus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "n_minus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", "")
        }

    if sim_scenario == 'sim_par_7':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_Nstd_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_Nstd_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 1
            },
            "n_target_seq": [300]*5,
            "n_plus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "n_minus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", "")
        }

    if sim_scenario == 'sim_par_8':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_Nstd_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_Nstd_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 0.5
            },
            "n_target_seq": [300]*5,
            "n_plus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "n_minus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", "")
        }

    if sim_scenario == 'sim_par_9':
        params = {
            'N': 100,
            "generation_function_tmp": generate_sample_Nstd_AR1_CC,
            "generation_function_tmp_rbf": generate_sample_Nstd_AR1_CC_rbf,
            "gen_params": {
                "p": 5,
                "beta": 1,
                'rho': 0.5
            },
            "n_target_seq": [300]*5,
            "n_plus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "n_minus_seq": [int(i/2) for i in [100, 200, 300, 400, 500]],
            "pi_target": pi_target,
            "pi_target_name": str(pi_target).replace(".", "")
        }
    
    if p_change is not None:
        params["gen_params"]['p'] = p_change
    if generation_function_change is not None:
        params["generation_function_tmp"] = generation_function_change[0]
    return params