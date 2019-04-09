import numpy as np
import matplotlib.pyplot as plt

# exact solve via the Thomas algorithm, N = # intervals
def exactsolve(N, dx, f, ue):
    n = N-1 # number of unknowns
    a = c = -1./dx**2; b = 2./dx**2
    y = np.zeros(n); bp = np.zeros(n); y[0] = f[0]; bp[0] = b
    for i in range(n-1):
        bp[i+1] = b - a*c/bp[i]
        y[i+1] = f[i+1] - a/bp[i]*y[i]
    u = np.zeros(n); u[n-1] = y[n-1]/bp[n-1] 
    for i in range(n-2, -1, -1):
        u[i] = (y[i] - c*u[i+1])/bp[i]
    ue[0] = 0; ue[1:N] = u; ue[N] = 0
    
# Jacobi iterations: modify u in place
def Jacobi(N, dx, u, f, niter, omega):
    for ii in range(niter):
        uL = u[0]
        for i in range(1, N):
            ut = u[i]; u[i] = (1-omega)*u[i] + omega*0.5*(uL + u[i+1] + dx*dx*f[i]); uL = ut

# residual calculation
def residual(N, dx, u, f, r):
    for i in range(1, N): r[i] = f[i] - (-u[i-1] + 2.*u[i] - u[i+1])/(dx*dx)

# residual restriction
def restrict(rh, rH):
    for i in range(1,rH.size-1): rH[i] = 0.25*rh[2*i-1] + 0.5*rh[2*i] + 0.25*rh[2*i+1]
    
# state prolongation
def prolongate(eH, uh):
    for i in range(1,eH.size-1):
        uh[2*i-1] += 0.5*eH[i]; uh[2*i] += eH[i]; uh[2*i+1] += 0.5*eH[i]

def printfig(fname):
    plt.xlabel(r'$x$', fontsize=32)
    plt.legend(fontsize=24, borderaxespad=0.1, loc=8)
    plt.grid(); plt.tick_params(axis='both', labelsize=24)
    plt.tight_layout(pad=0); plt.show(block=False); plt.savefig(fname)
            
def main():
    colors = ['black', 'blue', 'red', 'green', 'magenta', 'cyan'];

    fig = plt.figure(figsize=(16,12))
    
    for inl in range(1):
        '''
        # allocate space for vectors
        uv = []; fv = []; dxv = []; Nv = []; N = NH
        for ilevel in range(nlevel):
            uv.append(np.zeros(N+1)); fv.append(np.zeros(N+1))
            Nv.append(N); dxv.append(L/N)
            N *= 2
        r = np.zeros(Nv[nlevel-1]+1)
            
        # initial condition on finest level
        xh = np.linspace(0,L,Nv[nlevel-1]+1)
        uv[nlevel-1] = np.sin(4*np.pi*xh) + np.sin(8*np.pi*xh)
        # error history
        ehist = np.zeros(niter+1);
        ehist[0] = np.sqrt(dxv[nlevel-1])*np.linalg.norm(uv[nlevel-1]);
        # multigrid iterations
        for iiter in range(niter):
            # pre-smoothing, residual calc, restriction
            for i in range(nlevel-1, 0, -1):
                Jacobi(Nv[i], dxv[i], uv[i], fv[i], nu1, omega)
                residual(Nv[i], dxv[i], uv[i], fv[i], r)
                restrict(r, fv[i-1])

                uv[i-1] *= 0  # initialize solution (error) on coarse mesh

            # coarse-space solve
            exactsolve(Nv[0], dxv[0], fv[0], uv[0])

            # prolongation + post-smoothing
            for i in range(1, nlevel, 1):
                prolongate(uv[i-1], uv[i])
                Jacobi(Nv[i], dxv[i], uv[i], fv[i], nu2, omega)

            # error calculation
            ehist[iiter+1] = np.sqrt(dxv[nlevel-1])*np.linalg.norm(uv[nlevel-1]);
		'''
        L = 1.0
        niter = 20
        xh = np.linspace(0,L,8+1)
        u = np.sin(4*np.pi*xh) + np.sin(8*np.pi*xh)
        ehist = []
        ehist.append(np.sqrt(1/8)*np.linalg.norm(u))
        f = np.zeros(9)
        for i in range(niter):
            Jacobi(8, 1/8, u, f, 1,1 )
            ehist.append(np.sqrt(1/8)*np.linalg.norm(u))
        # make plot
        plt.semilogy(range(niter+1),ehist, '-', linewidth=2, color=colors[inl], label='N=8')
        
    # print figure
    plt.ylabel(r'$|u_h|_{L_2}$', fontsize=32)
    plt.xlabel(r'V-cycle iteration', fontsize=32)
    plt.xticks(range(niter+1))
    plt.grid(); plt.tick_params(axis='both', labelsize=24)
    plt.legend(fontsize=24, borderaxespad=0.1, loc=1)
    plt.tight_layout(pad=0); plt.show(block=True)
    plt.savefig('Vconv.pdf')
    plt.close(fig)
        
if __name__ == "__main__":
    main()
