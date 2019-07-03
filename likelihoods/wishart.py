# import numpy as np
import autograd.numpy as np
from autograd import grad
from autograd.scipy.special import multigammaln
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from scipy.stats import wishart
from scipy.special import logsumexp


class Wishart(Likelihood):
    """
    Wishart likelihood with a latent function over its mean parameter
    """

    def __init__(self, df=5, D=3, gp_link=link_functions.Identity()):
        self.df = df
        self.D = D
        super(Wishart, self).__init__(gp_link, name='Wishart')

    def pdf(self, F, Y, Y_metadata=None):
        N = len(F)
        S = self._construct_symm(Y)
        scale = self._construct_wishart(F) / self.df
        pdf = np.zeros(N)
        for i in range(N):
            pdf[i] = wishart.pdf(S[i], self.df, scale[i])
        return pdf

    def logpdf(self, F, Y, Y_metadata=None):
        N = len(F)
        S = self._construct_symm(Y)
        scale = self._construct_wishart(F) / self.df
        logpdf = np.zeros(N)
        for i in range(N):
            logpdf[i] = wishart.logpdf(S[i], self.df, scale[i])
        return logpdf

    def _logpdf(self, Fi, Yi):
        """Compute logpdf for one sample"""
        F = Fi.reshape(self.df, -1)
        W = np.sum(F[..., None] @ F[:, None], 0)
        Si = self._construct_symm(Yi[None])[0]
        scale = W / self.df
        numerator = (self.df - self.D - 1) / 2 * np.log(np.linalg.det(Si))
        denominator = self.df * self.D / 2 * np.log(2) + self.df / 2 * np.log(
            np.linalg.det(scale)) + multigammaln(self.df / 2, self.D)
        expo = -0.5 * np.trace(np.linalg.inv(scale) @ Si)
        return numerator - denominator + expo

    def samples(self, F, num_samples, Y_metadata=None):
        N = len(F)
        scale = self._construct_wishart(F) / self.df
        rvs = np.zeros((N, self.D, self.D))
        for i in range(N):
            rvs[i] = wishart.rvs(self.df, scale[i], size=num_samples)
        tril_ind = np.tril_indices(self.D)
        return rvs[:, tril_ind[0], tril_ind[1]]

    def _construct_wishart(self, F):
        # Construct positive semi-definite matrix
        N = len(F)
        W = F.reshape(N, self.df, -1)
        W = W[..., None] @ W[:, :, None]
        W = np.sum(W, 1)
        return W

    def _construct_symm(self, Y):
        L = np.zeros((len(Y), self.D, self.D))
        U = np.zeros((len(Y), self.D, self.D))
        tril_ind = np.tril_indices(self.D)
        stril_ind = np.triu_indices(self.D, 1)
        L[:, tril_ind[0], tril_ind[1]] = Y
        U[:, stril_ind[0], stril_ind[1]] = L[:, stril_ind[0], stril_ind[1]]
        return L + np.swapaxes(U, 1, 2)

    def var_exp(self, Y, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=10)
        else:
            gh_f, gh_w = gh_points

        gh_w = gh_w / np.sqrt(np.pi)
        D = M.shape[1]
        # grid-size and fd tuples
        expanded_F_tuples = []
        grid_tuple = [M.shape[0]]
        for d in range(D):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1] * (D + 1)
            expanded_fd_tuple[d + 1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # mean-variance tuple
        mv_tuple = [1] * (D + 1)
        mv_tuple[0] = M.shape[0]
        mv_tuple = tuple(mv_tuple)

        # building, normalizing and reshaping the grids
        F = np.zeros((np.prod(grid_tuple), D))
        for d in range(D):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d]) * np.sqrt(
                2 * np.reshape(V[:, d], mv_tuple)) + np.reshape(
                    M[:, d], mv_tuple)
            F[:, d, None] = fd.reshape(np.prod(grid_tuple), -1, order='C')

        # function evaluation
        Y_full = np.repeat(Y, gh_f.shape[0]**D, axis=0)
        logp = self.logpdf(F, Y_full)
        logp = logp.reshape(tuple(grid_tuple))

        # calculating quadrature
        var_exp = logp.dot(gh_w)  # / np.sqrt(np.pi)
        for d in range(D - 1):
            var_exp = var_exp.dot(gh_w)  # / np.sqrt(np.pi)
        return var_exp[:, None]

    def var_exp_derivatives(self, Y, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=10)
        else:
            gh_f, gh_w = gh_points

        gh_w = gh_w / np.sqrt(np.pi)
        N = M.shape[0]
        D = M.shape[1]
        # grid-size and fd tuples
        expanded_F_tuples = []
        grid_tuple = [M.shape[0]]
        for d in range(D):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1] * (D + 1)
            expanded_fd_tuple[d + 1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # mean-variance tuple
        mv_tuple = [1] * (D + 1)
        mv_tuple[0] = M.shape[0]
        mv_tuple = tuple(mv_tuple)

        # building, normalizing and reshaping the grids
        F = np.zeros((np.prod(grid_tuple), D))
        for d in range(D):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d]) * np.sqrt(
                2 * np.reshape(V[:, d], mv_tuple)) + np.reshape(
                    M[:, d], mv_tuple)
            F[:, d, None] = fd.reshape(np.prod(grid_tuple), -1, order='C')

        # function evaluation
        Y_full = np.repeat(Y, gh_f.shape[0]**D, axis=0)
        var_exp_dm = np.empty((N, D))
        var_exp_dv = np.empty((N, D))
        dlogp_df = np.zeros_like(F)
        d2logp_df2 = np.zeros_like(F)
        for i in range(len(F)):
            dlogp_df[i] = grad(self._logpdf)(F[i], Y_full[i])
        for d in range(D):
            # wrt to the mean
            dlogp = dlogp_df[:, d, None]
            dlogp = dlogp.reshape(tuple(grid_tuple))
            ve_dm = dlogp.dot(gh_w)  # / np.sqrt(np.pi)
            # wrt to the variance
            d2logp = d2logp_df2[:, d, None]
            d2logp = d2logp.reshape(tuple(grid_tuple))
            ve_dv = d2logp.dot(gh_w)  # / np.sqrt(np.pi)
            for fd in range(D - 1):
                ve_dm = ve_dm.dot(gh_w)  # / np.sqrt(np.pi)
                ve_dv = ve_dv.dot(gh_w)  # / np.sqrt(np.pi)

            var_exp_dm[:, d] = ve_dm
            var_exp_dv[:, d] = 0.5 * ve_dv
        return var_exp_dm, var_exp_dv

    def predictive(self, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        # if gh_points is None:
        #     gh_f, gh_w = self._gh_points(T=10)
        # else:
        #     gh_f, gh_w = gh_points

        # gh_w = gh_w / np.sqrt(np.pi)
        # N = M.shape[0]
        # D = M.shape[1]
        # # grid-size and fd tuples
        # expanded_F_tuples = []
        # grid_tuple = [M.shape[0]]
        # for d in range(D):
        #     grid_tuple.append(gh_f.shape[0])
        #     expanded_fd_tuple = [1] * (D + 1)
        #     expanded_fd_tuple[d + 1] = gh_f.shape[0]
        #     expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # # mean-variance tuple
        # mv_tuple = [1] * (D + 1)
        # mv_tuple[0] = M.shape[0]
        # mv_tuple = tuple(mv_tuple)

        # grid_tuple_prod = np.prod(grid_tuple)

        # # building, normalizing and reshaping the grids
        # F = np.zeros((grid_tuple_prod, D))
        # for d in range(D):
        #     fd = np.zeros(tuple(grid_tuple))
        #     fd[:] = np.reshape(gh_f, expanded_F_tuples[d]) * np.sqrt(
        #         2 * np.reshape(V[:, d], mv_tuple)) + np.reshape(
        #             M[:, d], mv_tuple)
        #     F[:, d, None] = fd.reshape(grid_tuple_prod, -1, order='C')

        # # function evaluation
        # mean_pred = np.empty((N, D))
        # var_pred = np.zeros((N, D))
        # for d in range(D):
        #     # wrt to the mean
        #     mean_k = self.rho_k(F, d)
        #     mean_k = mean_k.reshape(tuple(grid_tuple))
        #     mean_pred_k = mean_k.dot(gh_w)  # / np.sqrt(np.pi)
        #     # wrt to the variance
        #     # NOT IMPLEMENTED
        #     for fd in range(D - 1):
        #         mean_pred_k = mean_pred_k.dot(gh_w)  # / np.sqrt(np.pi)

        #     mean_pred[:, d] = mean_pred_k
        # return mean_pred, var_pred
        return self._construct_wishart(M), None

    def log_predictive(self, Ytest, mu_F_star, v_F_star, num_samples):
        Ntest, D = mu_F_star.shape
        F_samples = np.empty((Ntest, D, num_samples))
        # function samples:
        for d in range(D):
            mu_fd_star = mu_F_star[:, d, None]
            var_fd_star = v_F_star[:, d, None]
            F_samples[:, d, :] = np.random.normal(mu_fd_star,
                                                  np.sqrt(var_fd_star),
                                                  size=(Ntest, num_samples))

        # monte-carlo:
        log_pred = -np.log(num_samples) + logsumexp(
            self.logpdf_sampling(F_samples, Ytest), axis=-1)
        log_pred = np.array(log_pred).reshape(*Ytest.shape)
        # log_predictive = (1/num_samples)*log_pred.sum()
        return log_pred

    def get_metadata(self):
        dim_y = self.D * (self.D + 1) // 2
        dim_f = self.df * self.D
        dim_p = 1
        return dim_y, dim_f, dim_p

    def ismulti(self):
        # Returns if the distribution is multivariate
        return True
