import sys
# import os
from pprint import pprint
import numpy as np
import numpy.random as npr
import climin

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
from likelihoods.gaussian import Gaussian
from likelihoods.categorical import Categorical
from hetmogp.het_likelihood import HetLikelihood
from hetmogp.svmogp import SVMOGP
from hetmogp.util import vem_algorithm as VEM
from hetmogp.util import latent_functions_prior

import matplotlib.pyplot as plt

M = 40
Q = 3

likelihoods_list = [Gaussian(sigma=1), Categorical(2)]
likelihood = HetLikelihood(likelihoods_list)
Y_metadata = likelihood.generate_metadata()
T = len(Y_metadata['task_index'])
D = likelihood.num_output_functions(Y_metadata)
npr.seed(1)

pprint((Y_metadata, T, D))

X1 = np.sort(npr.rand(600))[:, None]
X2 = np.sort(npr.rand(500))[:, None]
X = [X1, X2]

# kern = GPy.kern.RBF(1, lengthscale=0.1)
# pprint(kern.K(X1))

U = []
U_params = npr.uniform(-7, 7, (Q, 9))
for Xt in X:
    Ut = np.zeros((Xt.shape[0], Q))
    for q in range(Q):
        Ut[:, q, None] = U_params[q, 0] * np.cos(
            U_params[q, 1] * np.pi * Xt +
            U_params[q, 2] * np.pi) - U_params[q, 3] * np.sin(
                U_params[q, 4] * np.pi * Xt + U_params[q, 5] *
                np.pi) + U_params[q, 6] * np.cos(U_params[q, 7] * np.pi * Xt +
                                                 U_params[q, 8] * np.pi)
    #     plt.plot(Xt, Ut[:, q], alpha=0.2)
    # plt.show()
    U.append(Ut)

W = npr.uniform(-1, 1, (D, Q))
W = [W[Y_metadata['function_index'] == t] for t in range(T)]
pprint(W)

F = []
for t in range(T):
    F.append(U[t] @ W[t].T)
    # plt.plot(X[t], U[t], alpha=0.2)
    plt.plot(X[t], F[t], alpha=0.2)
    # plt.show()

Y = likelihood.samples(F=F, Y_metadata=Y_metadata)
for t in range(T):
    plt.scatter(X[t], Y[t])
plt.show()

idx_test = np.s_[350:450]
X2train = np.delete(X[1], idx_test, axis=0)
Y2train = np.delete(Y[1], idx_test, axis=0)
X2test = X[1][idx_test]
Y2test = Y[1][idx_test]

Xtrain = [X[0], X2train]
Ytrain = [Y[0], Y2train]
# Xtest = [X[0], X2test]
# Ytest = [Y[0], Y2test]

# print([Xt.shape for Xt in X])
# print([Yt.shape for Yt in Y])
# print([Xt.shape for Xt in Xtrain])
# print([Yt.shape for Yt in Ytrain])
# print([Xt.shape for Xt in Xtest])
# print([Yt.shape for Yt in Ytest])

# Plot training and testing data
for t in range(T):
    plt.scatter(Xtrain[t], Ytrain[t])
plt.scatter(X2test, Y2test)
plt.show()

# Initialisation
ls_q = np.array([0.05] * Q)
var_q = np.array([0.5] * Q)
kern_list = latent_functions_prior(Q,
                                   lenghtscale=ls_q,
                                   variance=var_q,
                                   input_dim=1)

Z = X[0][npr.choice(len(X[0]), M, replace=False)]

# Model and Inference
model = SVMOGP(X=Xtrain,
               Y=Ytrain,
               Z=Z,
               kern_list=kern_list,
               likelihood=likelihood,
               Y_metadata=Y_metadata,
               batch_size=100)

# model = VEM(model, vem_iters=7, maxIter_perVEM=100)

max_iter = 2200
loglikelihood = np.zeros(max_iter)


def callback(info):
    if info['n_iter'] % 100 == 0:
        print(info['n_iter'], model.log_likelihood())
    if info['n_iter'] > max_iter:
        return True
    loglikelihood[info['n_iter'] - 1] = model.log_likelihood()
    return False


# def f(x):
#     model.optimizer_array = x
#     return model.objective_function()


# def f_grad(x):
#     model.optimizer_array = x
#     return model.objective_function_gradients()


# opt = climin.Adam(model.optimizer_array,
#                   model.stochastic_grad,
#                   step_rate=2)

opt = climin.Adadelta(model.optimizer_array,
                      model.stochastic_grad,
                      step_rate=0.1,
                      decay=0.8)

opt.minimize_until(callback)

# from scipy.optimize import minimize

# model.optimizer_array[model.optimizer_array < -1000] = -1000
# res = minimize(f,
#                model.optimizer_array,
#                method='BFGS',
#                jac=f_grad,
#                options={'disp': True})


# print(res)
# print(res.x)

plt.plot(loglikelihood)
plt.ylim([-2e4, 1e3])
plt.show()

Fpred = model.posteriors_F(X[0])
Ypred, Vpred = model.predict(X[0])
pprint(model.Z)

# plt.scatter(X[0], Y[0], label='Y_0')
# plt.scatter(X[1], Y[1], label='Y_1')
for t in range(T):
    plt.scatter(Xtrain[t], Ytrain[t], label=f'Y_{t}')
plt.scatter(X2test, Y2test, label=f'Ytest')
plt.plot(X[0], Ypred[0], label='Ypred_0')
plt.plot(X[0], Ypred[1], label='Ypred_1')
plt.legend()
plt.show()

plt.plot(X[0], Fpred[0].mean, label='1')
plt.plot(X[0], Fpred[1].mean, label='2')
plt.legend()
plt.show()
