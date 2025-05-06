import numpy as np
from itertools import product

def gen_conditional_entropy(X, *Y, nbins=10, base=2):

    b = np.log(base)

    Y = np.array(Y)
    X, Y = X.flatten(), Y.reshape(Y.shape[0], -1)

    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y, axis=1), np.max(Y, axis=1)
    X_bins = np.linspace(x_min, x_max, nbins+1)
    Y_bins = np.linspace(y_min, y_max, nbins+1).T

    X_bins[-1] += 1
    Y_bins[:,-1] += 1

    joint_dims = tuple(Y.shape[0] * [nbins])
    joint_dimsp1 = tuple((Y.shape[0] + 1) * [nbins])

    Py = np.empty(joint_dims)
    Pxy = np.empty(joint_dimsp1)
    CE = 0

    joint_ranges = [range(x) for x in joint_dims]
    for indices in product(*joint_ranges):
        condition = np.ones_like(Y[0], dtype=np.bool_)
        for ii in range(len(joint_ranges)):
            condition = condition & (Y[ii] >= Y_bins[ii,indices[ii]]) &\
                        (Y[ii] < Y_bins[ii,indices[ii]+1])
        Py[*indices] = np.where(condition, 1, 0).sum() / Y.shape[1]

    for k, indices in product(range(nbins), product(*joint_ranges)):
        condition = np.ones_like(Y[0], dtype=np.bool_)
        for ii in range(len(joint_ranges)):
            condition = condition & (Y[ii] >= Y_bins[ii,indices[ii]]) &\
                        (Y[ii] < Y_bins[ii,indices[ii]+1])
        condition = condition & (X >= X_bins[k]) & (X < X_bins[k+1])
        Pxy[k, *indices] = np.where(condition, 1, 0).sum() / X.shape[0]

        if Pxy[k, *indices] and Py[*indices]:
            CE += Pxy[k, *indices] * np.log(Py[*indices] / Pxy[k, *indices] ) / b

    return CE


X = np.random.rand(100)
Y = np.random.rand(100)

ce = gen_conditional_entropy(X, Y)
cesp = scipy_conditional_entropy(X,Y)
print(f"   my conditional entropy = {ce}")
print(f"scipy conditional entropy = {cesp}")