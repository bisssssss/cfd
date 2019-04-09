import numpy as np

# Computes Lagrange basis function values
def basis(xn, x):
    N = len(xn)
    phi = np.zeros(N)
    for j in range(N):
        pj = 1.0
        for i in range(N):
            if (i==j): continue
            pj *= (x-xn[i])/(xn[j]-xn[i])
        phi[j] = pj
    return phi

# Computes Lagrange basis gradients
def gbasis(xn, x):
    N = len(xn)
    gphi = np.zeros(N)
    for j in range(N):
        gphi[j] = 0.
        for k in range(N):
            if (k==j): continue
            gj = 1.0/(xn[j]-xn[k])
            for i in range(N):
                if (i==j) or (i==k): continue
                gj *= (x-xn[i])/(xn[j]-xn[i])
            gphi[j] += gj
    return gphi


