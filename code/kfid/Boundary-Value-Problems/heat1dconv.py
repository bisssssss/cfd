import numpy as np
import matplotlib.pyplot as plt
from heat1d import heat1d
    
def main():
    T0, TN, L, kappa, N = 1, 4, 2, 0.5, 4
    Nref = 5
    dxv = np.zeros(Nref); ev = np.zeros(Nref)
    for i in range(Nref):
        x, T = heat1d(T0, TN, L, kappa, N)
        dxv[i] = x[2]-x[1]
        Te = 1./(kappa*(np.pi/L)**2)*np.sin(np.pi*x/L) + T0 + (TN-T0)*x/L
        ev[i] = np.sqrt(dxv[i]*np.sum((T-Te)**2))
        N *= 2
    f = plt.figure(figsize=(8,4))
    rate = np.log2(ev[Nref-2]/ev[Nref-1])
    plt.loglog(dxv, ev, 'o-', linewidth=2, color='blue', label='rate = %.2f'%(rate))
    plt.xlabel(r'mesh spacing, $\Delta x$', fontsize=16)
    plt.ylabel(r'$L_2$ error norm', fontsize=16)
    plt.legend(fontsize=16, borderaxespad=0.1, loc=2)
    plt.figure(f.number); plt.grid()
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout(); plt.show(block=True)
    plt.close(f)
        
if __name__ == "__main__":
    main()
