import subprocess

# # 1:
# for i in range(6):
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.25'])
#     print('computing 0.25 '+str(i))

# # 2:
# for i in range(6,9):
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.25'])
#     print('computing 0.25 '+str(i))
# for i in range(6,9):
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.75'])
#     print('computing 0.75 '+str(i))

# # 3:
# for i in range(6):
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.75'])
#     print('computing 0.75 '+str(i))

# # 4:
# for i in [0,1,3,4,6,7]:
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.25'])
#     print('computing couchy 0.25 '+str(i))

# # 5:
# for i in [0,1,3,4,6,7]:
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.75'])
#     print('computing couchy 0.75 '+str(i))

# # 6:
# for i in range(3,6):
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-10", '0.25'])
#     print('computing p=10 0.25 '+str(i))

# # 7:
# for i in range(3,6):
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-10", '0.75'])
#     print('computing p=10 0.75 '+str(i))

# # 8:
# for i in range(3,6):
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-20", '0.25'])
#     print('computing p=20 0.25 '+str(i))

# # 9:
# for i in range(3,6):
#     subprocess.run(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-20", '0.75'])
#     print('computing p=20 0.75 '+str(i))


# # 10:
# subprocess.run(["python", "run_simulations_rkhs_gamma_grid.py", "sim_par_4", '0.25'])
# print('computing grid 0.25 ')

# # 11:
# subprocess.run(["python", "run_simulations_rkhs_gamma_grid.py", "sim_par_4", '0.75'])
# print('computing grid 0.75 ')

# # 12:
# subprocess.run(["python", "run_simulations_rkhs_gamma_grid.py", "sim_par_4-p-20", '0.25'])
# print('computing grid p=20 0.25 ')

# # 13:
# subprocess.run(["python", "run_simulations_rkhs_gamma_grid.py", "sim_par_4-Cauchy", '0.25'])
# print('computing grid cauchy 0.25')

# # 14:
# subprocess.run(["python", "run_simulations_rkhs_gamma_grid.py", "sim_par_4-p-20", '0.75'])
# print('computing grid p=20 0.75 ')

# # 15:
# subprocess.run(["python", "run_simulations_rkhs_gamma_grid.py", "sim_par_4-Cauchy", '0.75'])
# print('computing grid cauchy 75')


# RUN 1 for 4-6 again for rbf and laplace so it's faster
# screens: 2,6
# 6 for laplacian and 0.75 missing - run later 8
# for i in range(3,6):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.25'])
#     print('computing 0.25 '+str(i))
for i in range(5,6):
    subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.75'])
    print('computing 0.75 '+str(i))



# for i in range(9):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.25'])
#     print('computing 0.25 '+str(i))

# for i in range(6,9):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1), '0.75'])
#     print('computing 0.75 '+str(i))


# laplace 6-9 0.75

# for i in range(3,6):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-10", '0.25'])
#     print('computing 0.25 '+str(i))

# for i in range(3,6):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-20", '0.25'])
#     print('computing 0.25 '+str(i))

# for i in range(3,6):
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-p-20", '0.25'])
#     print('computing 0.25 '+str(i))

# for i in [0]:
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.25'])
#     print('computing 0.25 '+str(i))

# for i in [1,4,7]:#[0,1,3,4,6,7]:
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.25'])
#     print('computing 0.25 '+str(i))

# for i in [0,1,3,4,6,7]:
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.75'])
#     print('computing 0.75 '+str(i))

# # we need to run this:
# for i in [0,3,6]:#[0,1,3,4,6,7]:
#     subprocess.Popen(["python", "run_simulations_rkhs_gamma.py", "sim_par_"+str(i+1)+"-Cauchy", '0.25'])
#     print('computing 0.25 '+str(i))