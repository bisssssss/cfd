import numpy as np
import matplotlib.pyplot as plt
from heat1d import heat1d

def plotsol(x, T, fname):
    f = plt.figure(figsize=(8,3))
    plt.plot(x, T, 'o-', linewidth=2, color='blue')
    plt.xlabel(r'position, $x$', fontsize=16)
    plt.ylabel(r'Temperature, $T$', fontsize=16)
    plt.figure(f.number); plt.grid()
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout(); plt.show(block=True);
    plt.close(f)
    
def main():
    T0, TN, L, kappa = 1, 1, 2, 0.5
    x, T = heat1d(T0, TN, L, kappa, 10)
    plotsol(x, T, 'heat1d_N10.pdf')
        
if __name__ == "__main__":
    main()
