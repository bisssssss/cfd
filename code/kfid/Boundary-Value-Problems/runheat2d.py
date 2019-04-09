import numpy as np
import matplotlib.pyplot as plt
from heat2d import heat2d

def plotsol(X, Y, T, fname):
    f = plt.figure(figsize=(8,6))
    plt.contourf(X, Y, T)
    plt.xlabel(r'$x$', fontsize=16)
    plt.ylabel(r'$y$', fontsize=16)
    plt.figure(f.number); plt.grid()
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout(); plt.show(block=True);
    plt.close(f)
    
def main():
    Lx, Ly, Nx, Ny, kappa = 2., 1., 40, 20, 0.5
    X, Y, T = heat2d(Lx, Ly, Nx, Ny, kappa)
    plotsol(X, Y, T, 'heat2d_N861.pdf')
        
if __name__ == "__main__":
    main()
