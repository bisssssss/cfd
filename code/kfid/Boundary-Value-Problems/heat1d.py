import numpy as np
from scipy import sparse
from scipy.sparse import linalg

def heat1d(T0, TN, L, kappa, N):
    dx = float(L)/N             # spacing
    x = np.linspace(0, L, N+1)  # all nodes
    data = np.ones((3,N-1))     # diagonals for making a sparse matrix
    data[0,:] *= -1; data[1,:] *= 2; data[2,:] *= -1
    diags = np.array([-1, 0, 1])
    A = sparse.spdiags(data, diags, N-1, N-1, 'csr')
    q = np.sin(np.pi*x/L)       # source term
    b = q[1:N]*dx*dx/kappa      # right-hand side
    b[0] += T0; b[N-2] += TN    # Dirichlet boundary conditions
    Tt = linalg.spsolve(A,b)    # solution at interior points
    T = np.zeros(N+1)           # solution at all points
    T[0] = T0; T[1:N] = Tt; T[N] = TN
    return x, T
    
def main():
    x, T = heat1d(3, 4, 2.0, .5, 10)
    
if __name__ == "__main__":
    main()
