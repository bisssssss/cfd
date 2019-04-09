import numpy as np
import matplotlib.pyplot as plt

def uexact(x):
    return 1 + (np.sin(2*x))**3

def guexact(x):
    return 3*(np.sin(2*x))**2 * np.cos(2*x) * 2

def printfig(f, fname):
    plt.figure(f.number); plt.grid()
    plt.tick_params(axis='both', labelsize=16)
    f.tight_layout(); plt.show(block=True); plt.savefig(fname)

def main():
    x0 = 0.5  # point at which derivative is approximated
    dx = 0.1  # initial delta x
    N = 6     # number of dx values to consider
    E = np.zeros((3,N))
    dxv = np.zeros((N,1));
    tauB = np.zeros((N,1));
    tauF = np.zeros((N,1))
    tauC = np.zeros((N,1))
    gu0 = guexact(x0)
    for n in range(N):
        dxv[n] = dx
        x = np.array([x0-dx, x0, x0+dx])
        u = uexact(x)
        tauB[n] = (u[1]-u[0])/(x[1]-x[0]) - gu0
        tauF[n] = (u[2]-u[1])/(x[2]-x[1]) - gu0
        tauC[n] = (u[2]-u[0])/(x[2]-x[0]) - gu0
        dx /= 2
    f = plt.figure(figsize=(8,6))
    rate = np.log2(tauF[N-2]/tauF[N-1])
    plt.loglog(dxv, abs(tauF), 'o-', linewidth=2, color='black', label='Forward: %.2f'%(rate))
    rate = np.log2(tauB[N-2]/tauB[N-1])
    plt.loglog(dxv, abs(tauB), 's-', linewidth=2, color='blue', label='Backward: %.2f'%(rate))
    rate = np.log2(tauC[N-2]/tauC[N-1])
    plt.loglog(dxv, abs(tauC), '^-', linewidth=2, color='red', label='Central: %.2f'%(rate))
    plt.legend(fontsize=16, loc=2, borderaxespad=0.1)
    plt.xlabel(r'$\Delta x$', fontsize=20)
    plt.ylabel(r'truncation error, $\tau$', fontsize=20)
    printfig(f, 'pverify.pdf')
    plt.close(f)
        
if __name__ == "__main__":
    main()
