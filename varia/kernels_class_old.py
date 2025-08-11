import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_kernels

def U_X_X(K_X_X):
    return (np.sum(K_X_X) - np.sum(np.diag(K_X_X)))/(K_X_X.shape[0]*(K_X_X.shape[0] - 1))

def U_X_Y(K_X_Y):
    return np.mean(K_X_Y)

def E_K_X1_X2_K_X1_X3(K_X_X):

    n = K_X_X.shape[0]

    K_X_X_no_diag = K_X_X.copy()
    np.fill_diagonal(K_X_X_no_diag, 0)
    
    squared_sum_X2_K_X1_X2 = np.sum(K_X_X_no_diag, axis=1)**2
    sum_X2_squared_K_X1_X2 = np.sum(K_X_X_no_diag**2, axis=1)
    
    sum_X1 = np.sum(squared_sum_X2_K_X1_X2 - sum_X2_squared_K_X1_X2)

    return sum_X1/(n*(n-1)*(n-2))

def E_K_X1_Y1_K_X1_Y2(K_X_Y):

    n_X = K_X_Y.shape[0]
    n_Y = K_X_Y.shape[1]
    
    squared_sum_Y1_K_X1_Y1 = np.sum(K_X_Y, axis=1)**2
    sum_Y1_squared_K_X1_Y1 = np.sum(K_X_Y**2, axis=1)
    
    sum_X1 = np.sum(squared_sum_Y1_K_X1_Y1 - sum_Y1_squared_K_X1_Y1)

    return sum_X1/(n_X*n_Y*(n_Y-1))

def E_K_X1_X2_K_X1_Y1(K_X_X, K_X_Y):

    n_X = K_X_Y.shape[0]
    n_Y = K_X_Y.shape[1]
    
    sum_Y1_K_X1_Y1 = np.sum(K_X_Y, axis=1)

    K_X_X_no_diag = K_X_X.copy()
    np.fill_diagonal(K_X_X_no_diag, 0)
    sum_X2_K_X1_X2 = np.sum(K_X_X_no_diag, axis=1)

    sum_X1 = np.sum(sum_Y1_K_X1_Y1*sum_X2_K_X1_X2)

    return sum_X1/(n_X*n_Y*(n_X-1))

class estimator_RHKS():

    def __init__(self, 
                 X_target, X_source_positive, X_source_negative,
                 UorV_statistic = 'U',
                 kernel='rbf', kernel_params={}):
        
        self.X_target = X_target
        self.X_source_positive = X_source_positive
        self.X_source_negative = X_source_negative

        self.kernel = kernel
        self.kernel_params = kernel_params

        self.UorV_statistic = UorV_statistic

        self.n_target = X_target.shape[0]
        self.n_source_positive = X_source_positive.shape[0]
        self.n_source_negative = X_source_negative.shape[0]
        self.n_source = self.n_source_positive + self.n_source_negative

        self.pi_nrm = None
        self.pi_ipr = None

        self._is_K_computed = False
        self._is_U_computed = False
        self._is_D_computed = False

        self.compute_lambdas()

    def compute_K(self):

        self.K_target_target = pairwise_kernels(self.X_target, self.X_target, metric=self.kernel, **self.kernel_params)
        self.K_source_positive_source_positive = pairwise_kernels(self.X_source_positive, self.X_source_positive, metric=self.kernel, **self.kernel_params)
        self.K_source_negative_source_negative = pairwise_kernels(self.X_source_negative, self.X_source_negative, metric=self.kernel, **self.kernel_params)
        
        self.K_target_source_positive = pairwise_kernels(self.X_target, self.X_source_positive, metric=self.kernel, **self.kernel_params)
        self.K_target_source_negative = pairwise_kernels(self.X_target, self.X_source_negative, metric=self.kernel, **self.kernel_params)
        self.K_source_positive_source_negative = pairwise_kernels(self.X_source_positive, self.X_source_negative, metric=self.kernel, **self.kernel_params)

        self._is_K_computed = True

    def compute_U(self):
        
        if self.UorV_statistic == 'U':
            self.U_target_target = U_X_X(self.K_target_target)
            self.U_source_positive_source_positive = U_X_X(self.K_source_positive_source_positive)
            self.U_source_negative_source_negative = U_X_X(self.K_source_negative_source_negative)
            # self.V_target_target = U_X_Y(self.K_target_target)
            # self.V_source_positive_source_positive = U_X_Y(self.K_source_positive_source_positive)
            # self.V_source_negative_source_negative = U_X_Y(self.K_source_negative_source_negative)
        if self.UorV_statistic == 'V':
            self.U_target_target = U_X_Y(self.K_target_target)
            self.U_source_positive_source_positive = U_X_Y(self.K_source_positive_source_positive)
            self.U_source_negative_source_negative = U_X_Y(self.K_source_negative_source_negative)

        self.U_target_source_positive = U_X_Y(self.K_target_source_positive)
        self.U_target_source_negative = U_X_Y(self.K_target_source_negative)
        self.U_source_positive_source_negative = U_X_Y(self.K_source_positive_source_negative)

        self._is_U_computed = True

    def compute_D(self):

        self.D_target_source_positive = self.U_target_target - 2*self.U_target_source_positive + self.U_source_positive_source_positive
        self.D_target_source_negative = self.U_target_target - 2*self.U_target_source_negative + self.U_source_negative_source_negative
        self.D_source_positive_source_negative = self.U_source_positive_source_positive - 2*self.U_source_positive_source_negative + self.U_source_negative_source_negative

        self._is_D_computed = True

    def compute_K_U_D(self):

        if not self._is_K_computed:
            self.compute_K()
        if not self._is_U_computed:
            self.compute_U()
        if not self._is_D_computed:
            self.compute_D()

    def compute_lambdas(self):

        self.s_n = 1/self.n_target + 1/self.n_source_positive + 1/self.n_source_negative

        s_n_inv = 1/self.s_n

        self.lambda_target= s_n_inv/self.n_target
        self.lambda_source_positive = s_n_inv/self.n_source_positive
        self.lambda_source_negative = s_n_inv/self.n_source_negative

    def estimate_pi_nrm(self):

        self.compute_K_U_D()

        self.pi_nrm = np.sqrt(np.clip(a=self.D_target_source_negative/self.D_source_positive_source_negative,
                                      a_min=0, a_max=1))
        
    def estimate_pi_ipr(self):

        self.compute_K_U_D()

        self.pi_ipr = np.clip(a=((self.D_target_source_negative - self.D_target_source_positive)/self.D_source_positive_source_negative + 1)/2,
                              a_min=0, a_max=1)
        
    def compute_sigmas2_nrm_1(self):

        if self.pi_nrm is None:
            self.estimate_pi_nrm()

        self.sigma2_target_1 = np.var(np.mean(self.K_target_target, axis=1) - np.mean(self.K_target_source_negative, axis=1), ddof=1)
        self.sigma2_positive_1 = (self.pi_nrm**4)*np.var(np.mean(self.K_source_positive_source_positive, axis=1) - np.mean(self.K_source_positive_source_negative, axis=1), ddof=1)
        self.sigma2_negative_1 = ((self.pi_nrm*(1 - self.pi_nrm))**2)*np.var(np.mean(self.K_source_positive_source_negative, axis=0) - np.mean(self.K_source_negative_source_negative, axis=1), ddof=1)

    def compute_variance_nrm_1(self):

        nominator = self.lambda_target*self.sigma2_target_1 + self.lambda_source_positive*self.sigma2_positive_1 + self.lambda_source_negative*self.sigma2_negative_1
        denominator = self.pi_nrm**2*self.D_source_positive_source_negative**2
        
        self.var_nrm_1 = nominator/denominator
    
    def compute_variance_nrm_1_known_pi(self, pi):

        sigma2_target_1 = np.var(np.mean(self.K_target_target, axis=1) - np.mean(self.K_target_source_negative, axis=1))
        sigma2_positive_1 = (pi**4)*np.var(np.mean(self.K_source_positive_source_positive, axis=1) - np.mean(self.K_source_positive_source_negative, axis=1))
        sigma2_negative_1 = ((pi*(1 - pi))**2)*np.var(np.mean(self.K_source_positive_source_negative, axis=0) - np.mean(self.K_source_negative_source_negative, axis=1))

        nominator = self.lambda_target*sigma2_target_1 + self.lambda_source_positive*sigma2_positive_1 + + self.lambda_source_negative*sigma2_negative_1
        denominator = pi**2*self.D_source_positive_source_negative**2

        return nominator/denominator
    
    def compute_K2(self):

        self.E2_K_target_at_target = E_K_X1_X2_K_X1_X3(self.K_target_target)
        self.E2_K_source_positive_at_source_positive = E_K_X1_X2_K_X1_X3(self.K_source_positive_source_positive)
        self.E2_K_source_negative_at_source_negative = E_K_X1_X2_K_X1_X3(self.K_source_negative_source_negative)

        # self.E2_K_source_positive_at_target = E_K_X1_X2_K_X1_X3(self.K_target_source_positive)
        self.E2_K_source_negative_at_target = E_K_X1_X2_K_X1_X3(self.K_target_source_negative)
        self.E2_K_source_negative_at_positive = E_K_X1_X2_K_X1_X3(self.K_source_positive_source_negative)
        self.E2_K_source_positive_at_negative = E_K_X1_X2_K_X1_X3(self.K_source_positive_source_negative.T)

        self.E_K_target_K_source_negative_at_target = E_K_X1_X2_K_X1_Y1(self.K_target_target, self.K_target_source_negative)
        self.E_K_source_positive_K_source_negative_at_source_positive = E_K_X1_X2_K_X1_Y1(self.K_source_positive_source_positive, self.K_source_positive_source_negative)
        self.E_K_source_positive_K_source_negative_at_source_negative = E_K_X1_X2_K_X1_Y1(self.K_source_negative_source_negative, self.K_source_positive_source_negative.T)

    def compute_sigmas2_nrm_2(self):

        var1_tmp = self.E2_K_target_at_target - self.U_target_target**2
        cov_tmp = self.E_K_target_K_source_negative_at_target - self.U_target_target*self.U_target_source_negative
        var2_tmp = self.E2_K_source_negative_at_target - self.U_target_source_negative**2
        self.sigma2_target_2 = var1_tmp - 2*cov_tmp + var2_tmp
        print(var1_tmp)
        print(var2_tmp)

        var1_tmp = self.E2_K_source_positive_at_source_positive - self.U_source_positive_source_positive**2
        cov_tmp = self.E_K_source_positive_K_source_negative_at_source_positive - self.U_source_positive_source_positive*self.U_target_source_positive
        var2_tmp = self.E2_K_source_negative_at_positive - self.U_source_positive_source_negative**2
        self.sigma2_positive_2 = self.pi_nrm**4*(var1_tmp - 2*cov_tmp + var2_tmp)

        var1_tmp = self.E2_K_source_negative_at_source_negative - self.U_source_negative_source_negative**2
        cov_tmp = self.E_K_source_positive_K_source_negative_at_source_negative - self.U_source_positive_source_negative*self.U_source_negative_source_negative
        var2_tmp = self.E2_K_source_positive_at_negative - self.U_source_positive_source_negative**2
        self.sigma2_negative_2 = ((self.pi_nrm*(1 - self.pi_nrm))**2)*(var1_tmp - 2*cov_tmp + var2_tmp)
        
    def compute_variance_nrm_2(self):

        nominator = self.lambda_target*self.sigma2_target_2 + self.lambda_source_positive*self.sigma2_positive_2 + + self.lambda_source_negative*self.sigma2_negative_2
        denominator = self.pi_nrm**2*self.D_source_positive_source_negative**2
        
        self.var_nrm_2 = nominator/denominator

    # def compute_variance_nrm_2_known_pi(self, pi):

    #     sigma2_target_2 = self.E2_K_target_at_target - 2*self.E_K_target_K_source_negative_at_target + self.E2_K_source_negative_at_target
    #     sigma2_positive_2 = self.pi_nrm**4*(self.E2_K_source_positive_at_source_positive - 2*self.E_K_source_positive_K_source_negative_at_source_positive + self.E2_K_source_negative_at_positive)
    #     sigma2_negative_2 = ((self.pi_nrm*(1 - self.pi_nrm))**2)*(self.E2_K_source_negative_at_source_negative - 2*self.E_K_source_positive_K_source_negative_at_source_negative + self.E2_K_source_positive_at_negative)

    #     nominator = self.lambda_target*sigma2_target_2 + self.lambda_source_positive*sigma2_positive_2 + + self.lambda_source_negative*sigma2_negative_2
    #     denominator = pi**2*self.D_source_positive_source_negative**2
        
    #     return nominator/denominator

    def compute_taus2_ipr_1(self):

        if self.pi_ipr is None:
            self.estimate_pi_ipr()

        self.tau2_target_1 = np.var(np.mean(self.K_target_source_positive, axis=1) - np.mean(self.K_target_source_negative, axis=1), ddof=1)
        self.tau2_positive_1 = np.var(np.mean(self.K_target_source_positive, axis=0) - np.mean(self.K_source_positive_source_negative, axis=1), ddof=1)
        self.tau2_negative_1 = np.var(np.mean(self.K_target_source_negative, axis=0) - np.mean(self.K_source_positive_source_negative, axis=0), ddof=1)
            
    def compute_variance_ipr_1(self):

        nominator = self.lambda_target*self.tau2_target_1 + self.lambda_source_positive*self.tau2_positive_1 + self.lambda_source_negative*self.tau2_negative_1
        denominator = self.D_source_positive_source_negative**2
        
        self.var_ipr_1 = nominator/denominator


    

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