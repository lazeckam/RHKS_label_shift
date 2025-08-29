import numpy as np


def E_K_X1_X2_rbf(mu_1, Sigma_1, mu_2, Sigma_2, gamma):
    sig = np.sqrt(1/(2*gamma))
    p = Sigma_1.shape[0]
    M = Sigma_1 + Sigma_2 + sig**2*np.eye(p)
    mu_diff = mu_1 - mu_2
    return (sig**p)/np.sqrt(np.linalg.det(M))*np.exp(-np.dot(mu_diff, np.dot(mu_diff, np.linalg.inv(M)))/2)

# Computes \| \Phi(P_1) - \Phi(P_2) \|^2
# = E [ K(X_1, X_1) ] - 2 E[K(X_1, X_2)] + E [K(X_2, X_2)]
# for independent X_1, X_2
# X_j is p-variate Gaussian with mean \mu_j and variance matrix \Sigma_j
# def fun(mu_1, mu_2, Sigma_1, Sigma_2, gamma=0.5):
#     return E_K_X1_X2_rbf(mu_1, mu_1, Sigma_1, Sigma_1, gamma) - 2*E_K_X1_X2_rbf(mu_1, mu_2, Sigma_1, Sigma_2, gamma) + E_K_X1_X2_rbf(mu_2, mu_2, Sigma_2, Sigma_2, gamma) 
    

def E_K_X1_X2_K_X1_X3_rbf(mu1, Sigma1, mu2, Sigma2, mu3, Sigma3, gamma):
    sig = np.sqrt(1/(2*gamma))
    p = Sigma1.shape[0]
    pp = 2*p
    M = np.zeros((pp,pp))
    M[:p, :p] = Sigma1 + Sigma2 + sig**2 * np.eye(p)
    M[:p, p:] = Sigma1
    M[p:, :p] = Sigma1
    M[p:, p:] = Sigma1 + Sigma3 + sig**2 * np.eye(p)

    mudiff = np.concatenate((mu1-mu2, mu1-mu3))
    return (sig**pp / np.sqrt(np.linalg.det(M))) * np.exp(-np.dot(mudiff.T, np.dot(np.linalg.inv(M), mudiff))/2)

def Var_Phi_den1_den2_at_den3_rbf(mu1, Sigma1, mu2, Sigma2, mu3, Sigma3, gamma):
    mean1 = E_K_X1_X2_rbf(mu3, Sigma3, mu1, Sigma1, gamma)
    var1 = E_K_X1_X2_K_X1_X3_rbf(mu3, Sigma3, mu1, Sigma1, mu1, Sigma1, gamma) - mean1**2

    mean2 = E_K_X1_X2_rbf(mu3, Sigma3, mu2, Sigma2, gamma)
    var2 = E_K_X1_X2_K_X1_X3_rbf(mu3, Sigma3, mu2, Sigma2, mu2, Sigma2, gamma) - mean2**2

    cov12 = E_K_X1_X2_K_X1_X3_rbf(mu3, Sigma3, mu1, Sigma1, mu2, Sigma2, gamma) - mean1*mean2

    return var1 - 2*cov12 + var2

def variance_for_rbf(mu_plus, Sigma_plus, mu_minus, Sigma_minus, pi_target, 
                     lambda_target, lambda_plus, lambda_minus, gamma=2, *args, **kwargs):
    Var_Phi_plus_minus_at_plus = Var_Phi_den1_den2_at_den3_rbf(mu_plus, Sigma_plus, mu_minus, Sigma_minus, mu_plus, Sigma_plus, gamma)
    Var_Phi_plus_minus_at_minus = Var_Phi_den1_den2_at_den3_rbf(mu_plus, Sigma_plus, mu_minus, Sigma_minus, mu_minus, Sigma_minus, gamma)
    Delta_plus_minus = E_K_X1_X2_rbf(mu_plus, Sigma_plus, mu_plus, Sigma_plus, gamma) - 2*E_K_X1_X2_rbf(mu_plus, Sigma_plus, mu_minus, Sigma_minus, gamma) + E_K_X1_X2_rbf(mu_minus, Sigma_minus, mu_minus, Sigma_minus, gamma)
    Delta2_plus_minus = Delta_plus_minus**2

    term_target = pi_target*Var_Phi_plus_minus_at_plus + (1 - pi_target)*Var_Phi_plus_minus_at_minus + pi_target*(1 - pi_target)*Delta2_plus_minus
    term_plus = (pi_target**2)*Var_Phi_plus_minus_at_plus
    term_minus = ((1 - pi_target)**2)*Var_Phi_plus_minus_at_minus

    var_gamma = (lambda_target*term_target + lambda_plus*term_plus + lambda_minus*term_minus)/Delta2_plus_minus

    return var_gamma, term_target, term_plus, term_minus