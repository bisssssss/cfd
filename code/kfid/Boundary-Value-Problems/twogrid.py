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
    
# state/error prolongation
def prolongate(eH, uh):
    for i in range(1,eH.size-1):
        uh[2*i-1] += 0.5*eH[i]; uh[2*i] += eH[i]; uh[2*i+1] += 0.5*eH[i]

def printfig(fname):
    plt.xlabel(r'$x$', fontsize=32)
    plt.legend(fontsize=24, borderaxespad=0.1, loc=8)
    plt.grid(); plt.tick_params(axis='both', labelsize=24)
    plt.tight_layout(pad=0); plt.show(block=True);# plt.savefig(fname)
            
def main():
    nu1 = 2; nu2 = 2; niter = 3; L = 1.0; omega = 2./3.
    Nh = 64; dxh = L/Nh; NH = int(Nh/2); dxH = L/NH
    xh = np.linspace(0,L,Nh+1); xH = np.linspace(0,L,NH+1)
    uh = np.sin(4*np.pi*xh) + np.sin(8*np.pi*xh)
    fh = np.zeros(Nh+1); rh = np.zeros(Nh+1); #eh = np.zeros(Nh+1)
    rH = np.zeros(NH+1); eH = np.zeros(NH+1)
    ev = np.zeros(niter);
    for iiter in range(niter):

        # pre-smoothing
        fig = plt.figure(figsize=(16,12))
        plt.plot(xh,uh, '-', linewidth=4, color='blue', label='before smoothing')
        Jacobi(Nh, dxh, uh, fh, nu1, omega)  # pre-smooth
        plt.plot(xh,uh, '--', linewidth=2, color='red', label='after smoothing')
        plt.ylabel(r'$u_h$', fontsize=32)
        plt.title('MG cycle %d: pre-smoothing'%(iiter), fontsize=32)
        printfig('TwoGridSteps%d_h1.pdf'%(iiter))
        plt.close(fig)
        
        # restriction + coarse-space solve
        residual(Nh, dxh, uh, fh, rh)
        restrict(rh, rH)
        exactsolve(NH, dxH, rH[1:NH], eH)
        fig = plt.figure(figsize=(16,12))
        plt.plot(xH,eH, '-', linewidth=4, color='blue', label='coarse solution')
        plt.title('MG cycle %d: coarse solve'%(iiter), fontsize=32)
        plt.ylabel(r'$e_{2h}$', fontsize=32)
        printfig('TwoGridSteps%d_H.pdf'%(iiter))
        plt.close(fig)
        
        # prolongation + post-smoothing
        prolongate(eH, uh)
        fig = plt.figure(figsize=(16,12))
        plt.plot(xh,uh, '-', linewidth=4, color='blue', label='before smoothing')
        Jacobi(Nh, dxh, uh, fh, nu2, omega)  # post-smooth
        plt.plot(xh,uh, '--', linewidth=4, color='red', label='after smoothing')
        plt.ylabel(r'$u_h$', fontsize=32)
        plt.title('MG cycle %d: post-smoothing'%(iiter), fontsize=32)
        printfig('TwoGridSteps%d_h2.pdf'%(iiter))
        plt.close(fig)

        ev[iiter] = np.sqrt(dxh)*np.linalg.norm(uh);

    fig = plt.figure(figsize=(16,12))
    plt.semilogy(range(niter),ev, 'o-', linewidth=4, markersize=20,color='blue')
    plt.ylabel(r'$|u_h|_{L_2}$', fontsize=32)
    plt.xlabel(r'Two-grid cycle number', fontsize=32)
    plt.xticks(range(niter))
    plt.grid(); plt.tick_params(axis='both', labelsize=24)
    plt.tight_layout(pad=0); plt.show(block=True)
    plt.savefig('TwoGridStepsConv.pdf')
    plt.close(fig)
        
if __name__ == "__main__":
    main()
