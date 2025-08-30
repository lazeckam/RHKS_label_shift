import subprocess

p1 = subprocess.Popen(["python", "run_simulations_rkhs_n_target.py", "sim_par_1", '0.6'])
p2 = subprocess.Popen(["python", "run_simulations_rkhs_n_target.py", "sim_par_1", '0.4'])