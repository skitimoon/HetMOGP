import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from scipy.stats import wishart
# from scipy.misc import logsumexp


class Wishart(Likelihood):
    """
    Wishart likelihood with a latent function over its mean parameter
    """

    def __init__(self, df=3, gp_link=link_functions.Identity()):

        super(Wishart, self).__init__(gp_link, name='Wishart')

    def pdf(self, W, Y, Y_metadata=None):
        pdf = wishart.pdf(Y, df=self.df, scale=W/self.df)
        return pdf

    def logpdf(self, W, Y, Y_metadata=None):
        logpdf = wishart.logpdf(Y, df=self.df, scale=W/self.df)
        return logpdf

    def samples(self, W, num_samples, Y_metadata=None):
        # samples = np.random.normal(loc=f, scale=self.df)
        samples = wishart.rvs(self.df, W/self.df, num_samples)
        return samples

    def var_exp(self, Y, M, V, gh_points=None, Y_metadata=None):
        # Variational expectation using Gauss–Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points

        var_exp = 1

        return var_exp

    def var_exp_derivatives(self, Y, m, v, gh_points=None, Y_metadata=None):
        # Variational Expectations of derivatives
        lik_v = np.square(self.df)
        m, v, Y = m.flatten(), v.flatten(), Y.flatten()
        m = m[:,None]
        v = v[:,None]
        Y = Y[:,None]
        var_exp_dm = - (m - Y) / lik_v
        var_exp_dv = - 0.5 * (1 / np.tile(lik_v, (m.shape[0],1)))
        return var_exp_dm, var_exp_dv

    def predictive(self, m, v, Y_metadata):
        mean_pred = m
        var_pred = np.square(self.df) + v
        return mean_pred, var_pred

    def log_predictive(self, Ytest, mu_F_star, v_F_star, num_samples):
        Ntest, D = mu_F_star.shape
        F_samples = np.empty((Ntest, num_samples, D))
        # function samples:
        for d in range(D):
            mu_fd_star = mu_F_star[:, d][:, None]
            var_fd_star = v_F_star[:, d][:, None]
            F_samples[:, :, d] = np.random.normal(mu_fd_star, np.sqrt(var_fd_star), size=(Ntest, num_samples))

        # monte-carlo:
        log_pred = -np.log(num_samples) + logsumexp(self.logpdf(F_samples[:,:,0], Ytest), axis=-1)
        log_pred = np.array(log_pred).reshape(*Ytest.shape)
        log_predictive = (1/num_samples)*log_pred.sum()
        return log_predictive

    def get_metadata(self):
        dim_y = 1
        dim_f = 1
        dim_p = 1
        return dim_y, dim_f, dim_p

    def ismulti(self):
        # Returns if the distribution is multivariate
        return True
