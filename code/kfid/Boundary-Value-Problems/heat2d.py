import numpy as np
from scipy import sparse
from scipy.sparse import linalg

def source(x, y, Lx, Ly, kappa):
    return np.sin(np.pi*x/Lx)*(-(np.pi/Lx)**2*y/Ly*(1-y/Ly) - 2/Ly**2)*(-kappa)

def heat2d(Lx, Ly, Nx, Ny, kappa):
    dx = float(Lx)/Nx;  dy = float(Ly)/Ny;   # spacing
    x = np.linspace(0, Lx, Nx+1)  # x nodes
    y = np.linspace(0, Ly, Ny+1)  # y nodes
    Y, X = np.meshgrid(y, x)      # matrices of all nodes
    N = (Nx+1)*(Ny+1) # total number of unknowns
    nnz = (4+2*(Nx-1)+2*(Ny-1))*1 + (Nx-1)*(Ny-1)*5 # number of nonzeros
    data = np.zeros(nnz, dtype=np.float);
    irow = np.zeros(nnz, dtype=np.int); icol = np.zeros(nnz, dtype=np.int)
    q = np.zeros(N)               # empty rhs vector

    # fill in interior contributions
    inz = 0
    for iy in range(1,Ny):
        for ix in range(1,Nx):
            i = iy*(Nx+1)+ix; x = ix*dx; y = iy*dy
            iL = i-1; iR = i+1; iD = i-(Nx+1); iU = i+(Nx+1)
            q[i] = source(x,y,Lx,Ly,kappa)/kappa
            I = range(inz+0,inz+5)
            irow[I] = i
            icol[I] = [i, iL, iR, iD, iU]
            data[I] = [2./dx**2 + 2./dy**2, -1./dx**2, -1./dx**2, -1./dy**2, -1./dy**2]
            inz += 5
            
    # enforce Dirichlet boundary conditions
    for ix in range(Nx+1):
        irow[inz] = icol[inz] = ix; data[inz] = 1.; inz+=1
        irow[inz] = icol[inz] = Ny*(Nx+1)+ix; data[inz] = 1.; inz+=1
    for iy in range(1,Ny):
        irow[inz] = icol[inz] = iy*(Nx+1); data[inz] = 1.; inz+=1
        irow[inz] = icol[inz] = iy*(Nx+1)+Nx; data[inz] = 1.; inz+=1
        
    # build sparse matrix
    A = sparse.csr_matrix((data, (irow, icol)), shape=(N,N))
        
    # solve system
    Tv = linalg.spsolve(A,q)    # solution at all points
    T = np.reshape(Tv, (Nx+1,Ny+1), order='F') # reshape into matrix

    return X, Y, T
    
def main():
    X, Y, T = heat2d(2.0, 1.0, 4, 3, 0.5)
    
if __name__ == "__main__":
    main()
