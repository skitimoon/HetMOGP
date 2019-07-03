import sys
sys.path.append('..')
# import numpy as np
from scipy.stats import wishart, multivariate_normal
from likelihoods.wishart import Wishart
import GPy
import autograd.numpy as np
from autograd.scipy.special import multigammaln
from autograd import grad
import matplotlib.pyplot as plt

N = 4
df = 6
D = 3
scale = np.eye(D)

# Plot different scale parameters
ls = np.linspace(1e-7, 12)
foo = np.zeros_like(ls)
for i, l in enumerate(ls):
    foo[i] = wishart.pdf(l * scale, df, scale)
    print(l, foo[i])
print(np.mean(foo))
print(ls[np.argmax(foo)], foo[np.argmax(foo)])
plt.plot(ls, foo)
plt.show()

Y = wishart.rvs(df, scale=scale, size=N)
print(Y)
# print(x.T)
# ws = wishart(df, scale)
# print(ws.pdf(x.T))
# print(ws.logpdf(x.T))

kern = GPy.kern.RBF(1)
X = np.random.rand(N)[:, None]
K = kern.K(X)
# print(K)
# print(np.linalg.eigvalsh(K))
F = np.zeros((N, df * D))
for i in range(df * D):
    F[:, i] = multivariate_normal.rvs(np.zeros(N), K)
print(F.shape)
ws = Wishart(df)
W = ws._construct_wishart(F)
print(W)
# print(ws.pdf(F, W))


def logp(Fi, Yi, df, D):
    F = Fi.reshape(df, -1)
    W = np.sum(F[..., None] @ F[:, None], 0)
    scale = W / df
    numerator = (df - D - 1) / 2 * np.log(np.linalg.det(Yi))
    denominator = df * D / 2 * np.log(2) + df / 2 * np.log(
        np.linalg.det(scale)) + multigammaln(df / 2, D)
    expo = -0.5 * np.trace(np.linalg.inv(scale) @ Yi)
    return numerator - denominator + expo


def numer_grad(fun, x, eps=1e-8):
    D = len(x)
    x_grad = np.zeros_like(x)
    for i in range(D):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] -= eps
        x2[i] += eps
        x_grad[i] = (fun(x2) - fun(x1)) / (2 * eps)
        # print((fun(x2) - fun(x1)) / (2 * eps), end=' ')
    return x_grad


def fun(x):
    return logp(x, Y[0], df, D)


# check_grad(fun, 2 * foo)

print('F[0]:', F[0])
print('logp:', logp(F[0], Y[0], df, D))
Fi = F[0].reshape(df, -1)
W = np.sum(Fi[..., None] @ Fi[:, None], 0)
scale = W / df
print('logp (scipy):', wishart.logpdf(Y[0], df, scale))
print('logp (HetMOGP)', ws._logpdf(F[0], Y[0]))
print('logp close:',
      np.allclose(wishart.logpdf(Y[0], df, scale), logp(F[0], Y[0], df, D)))
print('grad:', grad(logp)(F[0], Y[0], df, D))
print('numerical grad:', numer_grad(fun, F[0]))
print('grad close:',
      np.allclose(grad(logp)(F[0], Y[0], df, D), numer_grad(fun, F[0])))

print('HetMOGP')

# # print(grad(logp)(F, Y))
# print(wishart.logpdf(Y[0], df, scale))
# print(grad(wishart.logpdf)(Y[0], df, scale))
