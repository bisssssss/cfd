import numpy as np
import matplotlib.pyplot as plt
from lagrange import basis,gbasis

def printfig(f, fname):
    plt.figure(f.number); plt.grid()
    plt.tick_params(axis='both', labelsize=18)
    f.tight_layout(); plt.show(block=True); #plt.savefig(fname)

def main():
    L = 1; N = 4; Np = 200  # length of domain, num nodes and plot nodes
    xn = np.linspace(0,L,N)  # node locations
    xn[1] = .4
    print xn
    xp = np.linspace(0,L,Np) # plotting points
    Phi = np.zeros((Np,N))
    for n in range(Np): Phi[n,:] = basis(xn, xp[n]) # basis values
    f = plt.figure(figsize=(8,6))
    colors = ['black', 'red', 'blue', 'green', 'magenta', 'cyan']
    for i in range(N): plt.plot(xp, Phi[:,i], linewidth=4, color=colors[i%6], label=r'$L_{%d}(x)$'%(i))
    plt.xlabel('$x$', fontsize=24)
    plt.ylabel('$L(x)$', fontsize=24)
    plt.legend(fontsize=24, borderaxespad=0.1, loc=4)
    printfig(f, 'plotlagrange.pdf')
    plt.close(f)
        
if __name__ == "__main__":
    main()
