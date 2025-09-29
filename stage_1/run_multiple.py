import subprocess


# for i in range(6,9):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.25'])
#     print('computing 0.25 '+str(i))

# for i in range(3,9):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.75'])
#     print('computing 0.75 '+str(i))

# simp par 1 0.75 kill 
# laplace 0.25 dla 4-6 puszczony 2 raxy

# for i in range(3,6):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-10", '0.75'])
#     print('computing 0.75 '+str(i))

# for i in range(3,6):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-20", '0.75'])
#     print('computing 0.75 '+str(i))

# for i in range(3,6):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-20", '0.25'])
#     print('computing 0.25 '+str(i))

# for i in [0]:
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.25'])
#     print('computing 0.25 '+str(i))

for i in [1,4,7]:
    subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.25'])
    print('computing 0.25 '+str(i))

for i in [1,4,7]:
    subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.75'])
    print('computing 0.75 '+str(i))