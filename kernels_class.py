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

def E_K_X1_Y1_K_X1_Z1(K_X_Y, K_X_Z):
    #check

    n_X = K_X_Y.shape[0]
    n_Y = K_X_Y.shape[1]
    n_Z = K_X_Z.shape[1]
    
    sum_Y1 = np.sum(K_X_Y, axis=1)
    sum_Z1 = np.sum(K_X_Z, axis=1)
    
    sum_X1 = np.sum(sum_Y1*sum_Z1)

    return sum_X1/(n_X*n_Y*n_Z)

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

        self._is_lambda_computed = False
        self._is_K2_computed = False

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

        self.r_n = 1/self.n_target + 1/self.n_source_positive + 1/self.n_source_negative

        r_n_inv = 1/self.r_n

        self.lambda_target= r_n_inv/self.n_target
        self.lambda_source_positive = r_n_inv/self.n_source_positive
        self.lambda_source_negative = r_n_inv/self.n_source_negative

        self._is_lambda_computed = True

    def estimate_pi_nrm(self):

        self.compute_K_U_D()

        self.pi_nrm = np.sqrt(np.clip(a=self.D_target_source_negative/self.D_source_positive_source_negative,
                                      a_min=0, a_max=1))
        
    def estimate_pi_ipr(self):

        self.compute_K_U_D()

        self.pi_ipr = np.clip(a=((self.D_target_source_negative - self.D_target_source_positive)/self.D_source_positive_source_negative + 1)/2,
                              a_min=0, a_max=1)
    
    def compute_K2(self):

        self.E2_K_target_at_target = E_K_X1_X2_K_X1_X3(self.K_target_target)
        self.E2_K_source_positive_at_source_positive = E_K_X1_X2_K_X1_X3(self.K_source_positive_source_positive)
        self.E2_K_source_negative_at_source_negative = E_K_X1_X2_K_X1_X3(self.K_source_negative_source_negative)

        self.E2_K_source_negative_at_target = E_K_X1_Y1_K_X1_Y2(self.K_target_source_negative)
        self.E2_K_source_positive_at_target = E_K_X1_Y1_K_X1_Y2(self.K_target_source_positive)
        self.E2_K_target_at_source_negative = E_K_X1_Y1_K_X1_Y2(self.K_target_source_negative.T)
        self.E2_K_target_at_source_positive = E_K_X1_Y1_K_X1_Y2(self.K_target_source_positive.T)
        self.E2_K_source_negative_at_positive = E_K_X1_Y1_K_X1_Y2(self.K_source_positive_source_negative)
        self.E2_K_source_positive_at_negative = E_K_X1_Y1_K_X1_Y2(self.K_source_positive_source_negative.T)

        self.E_K_target_K_source_negative_at_target = E_K_X1_X2_K_X1_Y1(self.K_target_target, self.K_target_source_negative)
        self.E_K_source_positive_K_source_negative_at_source_positive = E_K_X1_X2_K_X1_Y1(self.K_source_positive_source_positive, self.K_source_positive_source_negative)
        self.E_K_source_positive_K_source_negative_at_source_negative = E_K_X1_X2_K_X1_Y1(self.K_source_negative_source_negative, self.K_source_positive_source_negative.T)

        self.E_K_target_K_source_positive_at_source_negative = E_K_X1_Y1_K_X1_Z1(self.K_source_positive_source_negative.T, self.K_target_source_negative.T)
        self.E_K_target_K_source_negative_at_source_positive = E_K_X1_Y1_K_X1_Z1(self.K_source_positive_source_negative, self.K_target_source_positive.T)
        self.E_K_source_positive_K_source_negative_at_target = E_K_X1_Y1_K_X1_Z1(self.K_target_source_negative, self.K_target_source_positive)

        self._is_K2_computed = True

    def compute_tau_plug_in(self):

        self.compute_K_U_D()

        self.tau_target_plug_in = np.var(np.mean(self.K_target_source_positive, axis=1) - np.mean(self.K_target_source_negative, axis=1), ddof=1)
        self.tau_positive_plug_in = np.var(np.mean(self.K_target_source_positive, axis=0) - np.mean(self.K_source_positive_source_negative, axis=1), ddof=1)
        self.tau_negative_plug_in = np.var(np.mean(self.K_target_source_negative, axis=0) - np.mean(self.K_source_positive_source_negative, axis=0), ddof=1)
            
    def estimate_variance_plug_in(self):

        if not self._is_lambda_computed:
            self.compute_lambdas()

        nominator = self.lambda_target*self.tau_target_plug_in + self.lambda_source_positive*self.tau_positive_plug_in + self.lambda_source_negative*self.tau_negative_plug_in
        denominator = self.D_source_positive_source_negative**2
        
        self.var_plug_in = nominator/denominator

    def compute_tau_explicit(self):

        self.compute_K_U_D()

        if not self._is_K2_computed:
            self.compute_K2()

        var1_tmp = self.E2_K_source_positive_at_target - self.U_target_source_positive**2
        cov_tmp = self.E_K_source_positive_K_source_negative_at_target - self.U_target_source_positive*self.U_target_source_negative
        var2_tmp = self.E2_K_source_negative_at_target - self.U_target_source_negative**2
        self.tau_target_explicit = var1_tmp - 2*cov_tmp + var2_tmp

        var1_tmp = self.E2_K_target_at_source_positive - self.U_target_source_positive**2
        cov_tmp = self.E_K_target_K_source_negative_at_source_positive - self.U_target_source_positive*self.U_source_positive_source_negative
        var2_tmp = self.E2_K_source_negative_at_positive - self.U_source_positive_source_negative**2
        self.tau_positive_explicit = var1_tmp - 2*cov_tmp + var2_tmp

        var1_tmp = self.E2_K_source_positive_at_negative - self.U_source_positive_source_negative**2
        cov_tmp = self.E_K_target_K_source_positive_at_source_negative - self.U_source_positive_source_negative*self.U_target_source_negative
        var2_tmp = self.E2_K_target_at_source_negative - self.U_target_source_negative**2
        self.tau_negative_explicit = var1_tmp - 2*cov_tmp + var2_tmp

    def estimate_variance_explicit(self):

        if not self._is_lambda_computed:
            self.compute_lambdas()

        nominator = self.lambda_target*self.tau_target_explicit + self.lambda_source_positive*self.tau_positive_explicit + self.lambda_source_negative*self.tau_negative_explicit
        denominator = self.D_source_positive_source_negative**2
        
        self.var_explicit = nominator/denominator
