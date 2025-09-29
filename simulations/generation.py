import numpy as np
import scipy as scp
import pandas as pd
from tqdm import tqdm
from copy import copy
import sys
from rbf_kernel import *
sys.path.append("../estimators")
from estimators_RKHS import *

N_max = 5000


def AR1(p, rho):
    """Generate AR(1) covariance matrix with parameter rho and size n x n."""
    return scp.linalg.toeplitz(rho**np.arange(p))

def generate_sample_normal_distribution_CC(mu_plus, Sigma_plus, mu_minus, Sigma_minus,
                                           n_plus, n_minus, n_target, pi_target, seed):
    """Generate sample"""

    np.random.seed(int(seed))
    

    n_target_plus = np.random.binomial(n=n_target, p=pi_target)
    n_target_minus = n_target - n_target_plus

    p_plus = np.random.multivariate_normal(mean=mu_plus, cov=Sigma_plus, size=N_max)[:n_plus,:]
    p_minus = np.random.multivariate_normal(mean=mu_minus, cov=Sigma_minus, size=N_max)[:n_minus,:]

    p_target_plus = np.random.multivariate_normal(mean=mu_plus, cov=Sigma_plus, size=N_max)[:n_target_plus,:]
    p_target_minus = np.random.multivariate_normal(mean=mu_minus, cov=Sigma_minus, size=N_max)[:n_target_minus,:]
    p_target = np.vstack((p_target_plus, p_target_minus))

    return p_plus, p_minus, p_target

def generate_sample_tstudent_distribution_CC(mu_plus, mu_minus, df,
                                             n_plus, n_minus, n_target, pi_target, seed):
    """Generate sample"""

    np.random.seed(int(seed))
    
    n_target_plus = np.random.binomial(n=n_target, p=pi_target)
    n_target_minus = n_target - n_target_plus

    p_plus = np.random.standard_t(df=df, size=(N_max, mu_plus.shape[0]))[:n_plus,:] + mu_plus
    p_minus = np.random.standard_t(df=df, size=(N_max, mu_minus.shape[0]))[:n_minus,:] + mu_minus

    p_target_plus = np.random.standard_t(df=df, size=(N_max, mu_plus.shape[0]))[:n_target_plus,:] + mu_plus
    p_target_minus = np.random.standard_t(df=df, size=(N_max, mu_minus.shape[0]))[:n_target_minus,:] + mu_minus
    p_target = np.vstack((p_target_plus, p_target_minus))

    return p_plus, p_minus, p_target

def generate_sample_Cauchy_Cauchy_CC(p, beta, 
                                     n_plus, n_minus, n_target, pi_target, seed):
    
    
    p = int(p)
    n_plus = int(n_plus)
    n_minus = int(n_minus)
    n_target = int(n_target)
    
    mu_plus = np.zeros(p)
    mu_minus = beta*np.ones(p)
    
    p_plus, p_minus, p_target = generate_sample_tstudent_distribution_CC(mu_plus, mu_minus, 1,
                                                                         n_plus, n_minus, n_target, pi_target, seed)
    
    return p_plus, p_minus, p_target

def generate_sample_Nstd_AR1_CC(p, beta, rho,
                                n_plus, n_minus, n_target, pi_target, seed):
    
    
    p = int(p)
    n_plus = int(n_plus)
    n_minus = int(n_minus)
    n_target = int(n_target)
    
    mu_plus = np.zeros(p)
    Sigma_plus = np.eye(p)
    mu_minus = beta*np.ones(p)
    Sigma_minus = AR1(p, rho)
    
    p_plus, p_minus, p_target = generate_sample_normal_distribution_CC(mu_plus, Sigma_plus, mu_minus, Sigma_minus,
                                                                       n_plus, n_minus, n_target, pi_target, seed)
    
    return p_plus, p_minus, p_target

def generate_sample_Nstd_Nstd_CC(p, beta, 
                                 n_plus, n_minus, n_target, pi_target, seed):
    
    
    p = int(p)
    n_plus = int(n_plus)
    n_minus = int(n_minus)
    n_target = int(n_target)
    
    mu_plus = np.zeros(p)
    Sigma_plus = np.eye(p)
    mu_minus = beta*np.ones(p)
    Sigma_minus = np.eye(p)
    
    p_plus, p_minus, p_target = generate_sample_normal_distribution_CC(mu_plus, Sigma_plus, mu_minus, Sigma_minus,
                                                                       n_plus, n_minus, n_target, pi_target, seed)
    
    return p_plus, p_minus, p_target

def generate_sample_Nstd_Nstd_CC_rbf(p, beta, n_plus, n_minus, n_target, pi_target, gamma, seed=None):
    p = int(p)
    n_plus = int(n_plus)
    n_minus = int(n_minus)
    n_target = int(n_target)
    
    mu_plus = np.zeros(p)
    Sigma_plus = np.eye(p)
    mu_minus = beta*np.ones(p)
    Sigma_minus = np.eye(p)

    r_n = 1/n_target + 1/n_plus + 1/n_minus
    r_n_inv = 1/r_n

    lambda_target= r_n_inv/n_target
    lambda_plus = r_n_inv/n_plus
    lambda_minus = r_n_inv/n_minus

    return variance_for_rbf(mu_plus, Sigma_plus, mu_minus, Sigma_minus, pi_target, 
                     lambda_target, lambda_plus, lambda_minus, gamma=gamma)

def generate_sample_Nstd_AR1_CC_rbf(p, beta, rho,
                                n_plus, n_minus, n_target, pi_target, gamma, seed=None):
    p = int(p)
    n_plus = int(n_plus)
    n_minus = int(n_minus)
    n_target = int(n_target)
    
    mu_plus = np.zeros(p)
    Sigma_plus = np.eye(p)
    mu_minus = beta*np.ones(p)
    Sigma_minus = AR1(p, rho)

    r_n = 1/n_target + 1/n_plus + 1/n_minus
    r_n_inv = 1/r_n

    lambda_target= r_n_inv/n_target
    lambda_plus = r_n_inv/n_plus
    lambda_minus = r_n_inv/n_minus

    return variance_for_rbf(mu_plus, Sigma_plus, mu_minus, Sigma_minus, pi_target, 
                     lambda_target, lambda_plus, lambda_minus, gamma=gamma)

def simulation(df_params, generation_function, param_names,
               what_to_compute=None):
    
    N = df_params.shape[0]
    df_results = pd.DataFrame(np.zeros((N, 6)), 
                              columns=['pi_nmr_u', 'pi_ipr_u', 
                                       'pi_nmr_v', 'pi_ipr_v', 
                                       'var_plug-in', 'var_explicit'])
    df = pd.concat((df_params, df_results), axis=1)
    
    for i in tqdm(range(N)):
        p_plus, p_minus, p_target = generation_function(**df_params.loc[i, param_names].to_dict())

        est_tmp = estimator_RHKS(p_target, p_plus, p_minus, kernel_params={'gamma': 1/(2*df_params.loc[i, 'p']**2)})
        est_tmp.estimate_pi_nrm()
        est_tmp.estimate_pi_ipr()

        if 'pi_nmr_u' in what_to_compute:
            df.loc[i, 'pi_nmr_u'] = est_tmp.pi_nrm
        if 'pi_ipr_u' in what_to_compute:
            df.loc[i, 'pi_ipr_u'] = est_tmp.pi_ipr

        if ('pi_nmr_v' in what_to_compute )and ( 'pi_ipr_v' in what_to_compute ):
            est_tmp2 = estimator_RHKS(p_target, p_plus, p_minus, kernel_params={'gamma': 1/(2*df_params.loc[i, 'p']**2)},
                                      UorV_statistic = 'V')
            est_tmp2.estimate_pi_nrm()
            est_tmp2.estimate_pi_ipr()
            df.loc[i, 'pi_nmr_v'] = est_tmp2.pi_nrm
            df.loc[i, 'pi_ipr_v'] = est_tmp2.pi_ipr

        if 'var_plug-in' in what_to_compute:
            est_tmp.compute_K2()
            est_tmp.compute_tau_plug_in()
            est_tmp.estimate_variance_plug_in()
            df.loc[i, 'var_plug-in'] = est_tmp.var_plug_in
            df.loc[i, 'var_plug-in_target'] = est_tmp.tau_target_plug_in
            df.loc[i, 'var_plug-in_plus'] = est_tmp.tau_positive_plug_in
            df.loc[i, 'var_plug-in_minus'] = est_tmp.tau_negative_plug_in

            df.loc[i,'r_n'] = est_tmp.r_n
            df.loc[i,'lambda_target'] = 1/(est_tmp.r_n*est_tmp.n_target)
            df.loc[i,'lambda_plus']  = 1/(est_tmp.r_n*est_tmp.n_source_positive)
            df.loc[i,'lambda_minus'] = 1/(est_tmp.r_n*est_tmp.n_source_negative)
        
        if 'var_explicit' in what_to_compute:
            est_tmp.compute_tau_explicit()
            est_tmp.estimate_variance_explicit()
            df.loc[i, 'var_explicit'] = est_tmp.var_explicit
            df.loc[i, 'var_explicit_target'] = est_tmp.tau_target_explicit
            df.loc[i, 'var_explicit_plus'] = est_tmp.tau_positive_explicit
            df.loc[i, 'var_explicit_minus'] = est_tmp.tau_negative_explicit

    
    if 'var_rbf' in what_to_compute:
        df_tmp = copy(df_params)
        df_tmp = df_tmp.drop(columns=['seed'])
        df_tmp = df_tmp.drop_duplicates()
        keys_join = df_tmp.columns
        df_tmp = df_tmp.reset_index()

        df_tmp['var_rbf'] = np.nan
        df_tmp['var_target_rbf'] = np.nan
        df_tmp['var_plus_rbf'] = np.nan
        df_tmp['var_minus_rbf'] = np.nan

        for i in range(df_tmp.shape[0]):
            var_tmp,var_target,var_plus,var_minus = generate_sample_Nstd_Nstd_CC_rbf(**df_tmp.loc[i, keys_join].to_dict(), gamma = 1/(2*df_tmp.loc[i, 'p']**2))
            df_tmp.loc[i, 'var_rbf'] = var_tmp
            df_tmp.loc[i, 'var_target_rbf'] = var_target
            df_tmp.loc[i, 'var_plus_rbf'] = var_plus
            df_tmp.loc[i, 'var_minus_rbf'] = var_minus

        df = pd.merge(df, df_tmp, on=keys_join.to_list(), how='left') 

    return df


def simulation_gamma(df_params, generation_function, param_names, gamma_seq,
               what_to_compute=None):
    
    df_final = []
    
    for gamma in gamma_seq:
    
        N = df_params.shape[0]
        df_results = pd.DataFrame(np.zeros((N, 6)), 
                                columns=['pi_nmr_u', 'pi_ipr_u', 
                                        'pi_nmr_v', 'pi_ipr_v', 
                                        'var_plug-in', 'var_explicit'])
        df = pd.concat((df_params, df_results), axis=1)
        
        for i in tqdm(range(N)):
            p_plus, p_minus, p_target = generation_function(**df_params.loc[i, param_names].to_dict())

            est_tmp = estimator_RHKS(p_target, p_plus, p_minus, kernel_params={'gamma': gamma})
            est_tmp.estimate_pi_nrm()
            est_tmp.estimate_pi_ipr()

            if 'pi_nmr_u' in what_to_compute:
                df.loc[i, 'pi_nmr_u'] = est_tmp.pi_nrm
            if 'pi_ipr_u' in what_to_compute:
                df.loc[i, 'pi_ipr_u'] = est_tmp.pi_ipr

            if ('pi_nmr_v' in what_to_compute )and ( 'pi_ipr_v' in what_to_compute ):
                est_tmp2 = estimator_RHKS(p_target, p_plus, p_minus, kernel_params={'gamma': gamma},
                                        UorV_statistic = 'V')
                est_tmp2.estimate_pi_nrm()
                est_tmp2.estimate_pi_ipr()
                df.loc[i, 'pi_nmr_v'] = est_tmp2.pi_nrm
                df.loc[i, 'pi_ipr_v'] = est_tmp2.pi_ipr

            if 'var_plug-in' in what_to_compute:
                est_tmp.compute_K2()
                est_tmp.compute_tau_plug_in()
                est_tmp.estimate_variance_plug_in()
                df.loc[i, 'var_plug-in'] = est_tmp.var_plug_in
                df.loc[i, 'var_plug-in_target'] = est_tmp.tau_target_plug_in
                df.loc[i, 'var_plug-in_plus'] = est_tmp.tau_positive_plug_in
                df.loc[i, 'var_plug-in_minus'] = est_tmp.tau_negative_plug_in

                df.loc[i,'r_n'] = est_tmp.r_n
                df.loc[i,'lambda_target'] = 1/(est_tmp.r_n*est_tmp.n_target)
                df.loc[i,'lambda_plus']  = 1/(est_tmp.r_n*est_tmp.n_source_positive)
                df.loc[i,'lambda_minus'] = 1/(est_tmp.r_n*est_tmp.n_source_negative)
            
            if 'var_explicit' in what_to_compute:
                est_tmp.compute_tau_explicit()
                est_tmp.estimate_variance_explicit()
                df.loc[i, 'var_explicit'] = est_tmp.var_explicit
                df.loc[i, 'var_explicit_target'] = est_tmp.tau_target_explicit
                df.loc[i, 'var_explicit_plus'] = est_tmp.tau_positive_explicit
                df.loc[i, 'var_explicit_minus'] = est_tmp.tau_negative_explicit

            if 'var_rbf_estimate' in what_to_compute:
                mu0 = np.mean(p_plus, 0)
                cov0 = np.cov(p_plus.T)
                mu1 = np.mean(p_minus, 0)
                cov1 = np.cov(p_plus.T)
                est1 = est_tmp.pi_nrm
                est2 = est_tmp.pi_ipr

                lam_target = 1/(est_tmp.r_n*est_tmp.n_target)
                lam_plus = 1/(est_tmp.r_n*est_tmp.n_source_positive)
                lam_minus = 1/(est_tmp.r_n*est_tmp.n_source_negative)

                var_tmp,_,_,_ = variance_for_rbf(mu0, cov0, mu1, cov1,
                                                                 est1,
                                                                 lam_target, lam_plus, lam_minus, 
                                                                 gamma = gamma)
                df.loc[i, 'var_rbf_nrm'] = var_tmp

                var_tmp,_,_,_ = variance_for_rbf(mu0, cov0, mu1, cov1,
                                                                 est2,
                                                                 lam_target, lam_plus, lam_minus, 
                                                                 gamma = gamma)
                df.loc[i, 'var_rbf_ipr'] = var_tmp


        if 'var_rbf' in what_to_compute:
            df_tmp = copy(df_params)
            df_tmp = df_tmp.drop(columns=['seed'])
            df_tmp = df_tmp.drop_duplicates()
            keys_join = df_tmp.columns
            df_tmp = df_tmp.reset_index()

            df_tmp['var_rbf'] = np.nan
            df_tmp['var_target_rbf'] = np.nan
            df_tmp['var_plus_rbf'] = np.nan
            df_tmp['var_minus_rbf'] = np.nan

            for i in range(df_tmp.shape[0]):
                var_tmp,var_target,var_plus,var_minus = generate_sample_Nstd_Nstd_CC_rbf(**df_tmp.loc[i, keys_join].to_dict(), gamma = gamma)
                df_tmp.loc[i, 'var_rbf'] = var_tmp
                df_tmp.loc[i, 'var_target_rbf'] = var_target
                df_tmp.loc[i, 'var_plus_rbf'] = var_plus
                df_tmp.loc[i, 'var_minus_rbf'] = var_minus

            df = pd.merge(df, df_tmp, on=keys_join.to_list(), how='left') 
        df['gamma'] = gamma
        
        df_final.append(df)

    df_final = pd.concat(df_final)

    return df_final
      