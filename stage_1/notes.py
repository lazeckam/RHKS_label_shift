


def compute_results(n_target):
    # gamma_seq = np.logspace(-6, -0.1, num=100)
    gamma_seq = np.logspace(-6, -0.4, num=50)
    gen_params = {
        'p' :5,
        'beta': 1, 
        'n_plus': 150,
        'n_minus': 150,
        'n_target': n_target,
        'pi_target': 0.75
    }
    results_grid = []
    N = 100

    df_results_emp = pd.DataFrame(np.zeros((N*len(gamma_seq), 8)), 
                          columns=['pi_ipr','pi_ipr_V', 'pi_nrm','var_n', 'var', 'seed', 'gamma', 'var_rbf'])
    iter = 0

    df_results = pd.DataFrame(np.zeros((N, 10)), 
                          columns=['gamma_numerical', 'gamma_numerical_V', 'gamma_grid', 'gamma_grid_V',
                                   'gamma_bootstrap_var', 'gamma_bootstrap_var_V', 'gamma_bootstrap_mse', 'gamma_bootstrap_mse_V',
                                   'gamma_distance', 'gamma_distance_V'])
    
    for i in tqdm(range(N)):
        p_source_plus, p_source_minus, p_target = generate_sample_Nstd_Nstd_CC(**gen_params, seed=i)
        for gamma in gamma_seq:
            
            mod = estimator_RHKS(p_target, p_source_plus, p_source_minus, kernel_params={'gamma': gamma})
            mod.compute_basic_simulations()

            df_results_emp.loc[iter, 'pi_ipr'] = mod.pi_ipr
            df_results_emp.loc[iter, 'pi_nrm'] =  mod.pi_nrm
            df_results_emp.loc[iter, 'var_n'] = mod.var_plug_in_n
            df_results_emp.loc[iter, 'var'] = mod.var_plug_in
            df_results_emp.loc[iter, 'seed'] = i
            df_results_emp.loc[iter, 'gamma'] =  gamma
            df_results_emp.loc[iter, 'var_rbf'] = generate_sample_Nstd_Nstd_CC_rbf(**gen_params, gamma=gamma)[0]*mod.r_n

            mod = estimator_RHKS(p_target, p_source_plus, p_source_minus, kernel_params={'gamma': gamma}, UorV_statistic='V')
            mod.compute_basic_simulations()
            df_results_emp.loc[iter, 'pi_ipr_V'] = mod.pi_ipr


            iter = iter + 1
        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='numerical')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_numerical'] = mod.gamma_opt

        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='numerical_V')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_numerical_V'] = mod.gamma_opt
        
        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='grid')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_grid'] = mod.gamma_opt
        results_grid.append({**mod.opt_values,
                             'model': 'grid',
                             'seed': i})

        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='grid_V')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_grid_V'] = mod.gamma_opt
        results_grid.append({**mod.opt_values,
                             'model': 'grid_V',
                             'seed': i})

        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='bootstrap_var')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_bootstrap_var'] = mod.gamma_opt
        results_grid.append({**mod.opt_values,
                             'model': 'bootstrap_var',
                             'seed': i})
        
        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='bootstrap_var_V')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_bootstrap_var_V'] = mod.gamma_opt
        results_grid.append({**mod.opt_values,
                             'model': 'bootstrap_var_V',
                             'seed': i})
        
        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='bootstrap_mse')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_bootstrap_mse'] = mod.gamma_opt
        results_grid.append({**mod.opt_values,
                             'model': 'bootstrap_mse',
                             'seed': i})
        
        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='bootstrap_mse_V')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_bootstrap_mse_V'] = mod.gamma_opt
        results_grid.append({**mod.opt_values,
                             'model': 'bootstrap_mse_V',
                             'seed': i})
        
        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='distance')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_distance'] = mod.gamma_opt
        results_grid.append({**mod.opt_values,
                             'model': 'distance',
                             'seed': i})
        
        mod = estimator_RHKS_rbf_gamma(p_target, p_source_plus, p_source_minus, how='distance_V')
        mod.compute_basic_simulations()
        df_results.loc[i, 'gamma_distance_V'] = mod.gamma_opt
        results_grid.append({**mod.opt_values,
                             'model': 'distance_V',
                             'seed': i})

    df_results_emp['MSE'] = (df_results_emp['pi_ipr'] - gen_params['pi_target'])**2
    df_results_emp['MSE_V'] = (df_results_emp['pi_ipr_V'] - gen_params['pi_target'])**2

    df1 = df_results_emp[['gamma', 'MSE']].groupby('gamma').aggregate('mean').reset_index()
    df1V = df_results_emp[['gamma', 'MSE_V']].groupby('gamma').aggregate('mean').reset_index()
    # df1['RMSE_div20'] = np.sqrt(df1['MSE'])/20
    df2 = df_results_emp[['gamma', 'pi_ipr']].groupby('gamma').aggregate(lambda s: s.var(ddof=0)).reset_index()
    df2.rename(columns={df2.columns[1]: "var_emp"}, inplace=True)
    df3 = df_results_emp[['gamma', 'var_n']].groupby('gamma').aggregate('mean').reset_index()
    df3.rename(columns={df3.columns[1]: "est_var_as"}, inplace=True)
    df4 = df_results_emp[['gamma', 'var_rbf']].groupby('gamma').aggregate('mean').reset_index()
    df1 = df1.merge(df1V, on='gamma')
    df = df1.merge(df2, on='gamma')
    df = df.merge(df3, on='gamma')
    df = df.merge(df4, on='gamma')

    return df_results_emp, df_results, df, results_grid