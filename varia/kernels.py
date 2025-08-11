import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_kernels







    


def RBF_kernel(x, y, sigma=1.0, gamma=None):
    """Evaluates the radial basis function (RBF) kernel on input arrays.

    Parameters
    ----------
    x, y : array like
        Input values.

    sigma : float
        Param

    gamma : float
        Param

    Returns
    -------
    ndarray
        RBF kernel evaluated on x and y.
    """ 
    RBF_result = None

    if sigma is not None:
        RBF_result = np.exp(-1/(sigma**2) * cdist(x, y, metric="sqeuclidean"))

    if sigma is None and gamma is not None:
        RBF_result = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))

    return RBF_result

def U_kernel(x, y, kernel='RBF', params_kernel = {'sigma': 1.0}):
    """Computes U estimator on input arrays.

    Parameters
    ----------
    x, y : array like
        Input values.

    kernel : string
        Param

    params_kernel : dict
        Params

    Returns
    -------
    ndarray
        U estimator.
    """ 
    U_estimator = np.mean(RBF_kernel(x,y))

    return U_estimator

def V_kernel(x, y, kernel='RBF', params_kernel = {'sigma': 1.0}):
    """Computes U estimator on input arrays.

    Parameters
    ----------
    x, y : array like
        Input values.

    kernel : string
        Param

    params_kernel : dict
        Params

    Returns
    -------
    ndarray
        U estimator.
    """ 
    U_estimator = (np.sum(RBF_kernel(x,y)) - np.sum(np.diag(RBF_kernel(x,y))))/(x.shape[0]**2 - x.shape[0])

    return U_estimator

def delta_P_Q_U(x, y):
    return U_kernel(x, x) - 2*U_kernel(x, y) + U_kernel(y, y)

def delta_P_Q_V(x, y):
    return V_kernel(x, x) - 2*U_kernel(x, y) + V_kernel(y, y)

def pi_ratio_u(p_target, p_source_positive, p_source_negative):
    return np.sqrt(delta_P_Q_U(p_target, p_source_negative)/delta_P_Q_U(p_source_positive, p_source_negative))

def pi_ratio_v(p_target, p_source_positive, p_source_negative):
    return np.sqrt(delta_P_Q_V(p_target, p_source_negative)/delta_P_Q_V(p_source_positive, p_source_negative))

def E_Phi(x, z=None):
    if z is not None:
        return U_kernel(x, z)
    else:
        return V_kernel(x, x)

def E_Phi_squared(x, z=None):
    if z is not None:
        k_x_z = RBF_kernel(x,z)
        res = 0
        den = 0
        for i in range(z.shape[0]):
            for j in range(x.shape[0]-1):
                for k in range(j, x.shape[0]):
                    res += k_x_z[j,i]*k_x_z[k,i]
                    den += 1
        res = res/den
        return res
    else:
        k_x_x = RBF_kernel(x,x)
        res = 0
        den = 0
        for i in range(x.shape[0]-2):
            for j in range(i, x.shape[0]-1):
                for k in range(j, x.shape[0]):
                    res += k_x_x[i,j]*k_x_x[i,k]
                    den += 1
        res = res/den
        return res

def E_Phi_X_Phi_Y(x, y, z=None):
    if z is not None:
        k_x_z = RBF_kernel(x,z)
        k_y_z = RBF_kernel(y,z)
        res = 0
        den = 0
        for i in range(z.shape[0]):
            for j in range(i, x.shape[0]):
                for k in range(0, y.shape[0]):
                    res += k_x_z[j,i]*k_y_z[k,i]
                    den += 1
        res = res/den
        return res
    else:
        k_x_x = RBF_kernel(x,x)
        k_x_y = RBF_kernel(x,y)
        res = 0
        den = 0
        for i in range(x.shape[0]-1):
            for j in range(i, x.shape[0]):
                for k in range(0, y.shape[0]):
                    res += k_x_x[i,j]*k_x_y[i,k]
                    den += 1
        res = res/den
        return res

def var_Phi(x, z=None):
    if z is not None:
        return E_Phi_squared(x, z=z) - E_Phi(x, z=z)**2
    else:
        return E_Phi_squared(x) - E_Phi(x)**2

def cov_Phi(x, y, z=None):
    if z is not None:
        return E_Phi_squared(x, z=z) -E_Phi(x, z=z)*E_Phi(y, z=z)
    else:
        return E_Phi_X_Phi_Y(x, y) - E_Phi(x)*E_Phi(y)
    

def sigma_prime(p_target, p_source_positive, p_source_negative):
    return var_Phi(p_target) - 2*cov_Phi(p_target, p_source_negative) + var_Phi(p_source_negative, p_target)

def sigma_plus(p_target, p_source_positive, p_source_negative, pi_target):
    return pi_target**4*(var_Phi(p_source_positive) - 2*cov_Phi(p_source_positive, p_source_negative) + var_Phi(p_source_negative, p_source_positive))


def sigma_minus(p_target, p_source_positive, p_source_negative, pi_target):
    return pi_target**2*(1-pi_target)**2*(var_Phi(p_source_positive, p_source_negative) - 2*cov_Phi(p_source_negative, p_source_positive) + var_Phi(p_source_negative))