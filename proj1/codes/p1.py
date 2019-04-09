import numpy as np 
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt

def MsnapshotVisual(data, clas, numb, V, E):
    u = data[:,1]/data[:,0]
    v = data[:,2]/data[:,0]
    vvec = np.sqrt(u**2 + v**2)
    gamma = 1.4
    c = np.sqrt(gamma*(gamma-1)*(data[:,3]/data[:,0]-1/2*vvec**2))
    M = vvec/c
    pltcontour(clas, numb, V, E, M, 'Mach number', 'M')

    return [M[5:9]] # return boundary values

def decomposition(V, E):
    ntrain = 20
    S = np.zeros((5 * V.shape[0], 20))
    for i in range (ntrain):
        data = np.loadtxt(fngen("train", i+1))
        data = np.reshape(data, (5 * V.shape[0], 1))
        S[:,i] = data[:,0]
    U, D, V = np.linalg.svd(S, False)

    return U, D, V

def pltcontour(clas, numb, V, E, value, title, shorttitle):

    fig, ax = plt.subplots(figsize = (8,4))
    ax.set_aspect('equal')
    tcf = ax.tricontourf(V[:,0], V[:,1], E, value, 30, cmap = 'jet')
    fig.colorbar(tcf)

    plt.xlabel(r'$x$', fontsize=16)
    plt.ylabel(r'$y$', fontsize=16)
    plt.xlim(-.3, 1.5)
    plt.ylim(-.5, .5)
    plt.title(title + ' contour of ' + clas + ' ' + str(numb))

    #plt.show();
    fig.savefig('../figs/' + shorttitle + '_'+clas+'_'+str(numb)+'.jpg')
    plt.close()

def fngen(c, i):
    return c + '_' + str(i) + '_state.txt'

def main():
    #############################################################
    # 
    # Note: plots have been auto saved to the figs directory.
    #      If you want to see it directly, uncomment the
    #      plt.show() in the last of main().
    
    # read V, E
    V = np.loadtxt('V.txt')
    E = np.loadtxt('E.txt')
    testMa = np.loadtxt('testMalpha.txt')
    trainMa = np.loadtxt('Malpha.txt')
    E[:,:] = E[:,:]-1 # python index
    
    ##############################################################
    #   
    # ************************* pt 1 *****************************
    cases = [['train', 1], 
             ['train', 20],
             ['test', 55],
             ['test', 90]]
    # plot
    print('pt1\n')
    for i in range(len(cases)):
        data = np.loadtxt(fngen(cases[i][0], cases[i][1]))
        M = MsnapshotVisual(data, cases[i][0], cases[i][1], V, E)
        print(cases[i])
        print('The corner values:')
        print(M)
        print('The exact Mach number:')
        if cases[i][0] == 'train':
            print(trainMa[cases[i][1]-1, 0],'\n')
        elif cases[i][0] == 'test':
            print(testMa[cases[i][1]-1, 0],'\n')
        else:
            print('Error. Check the input.\n')
    
    ##############################################################
    #   
    # ************************* pt 2 *****************************
    U, D, _ = decomposition(V, E)
    f = plt.figure(figsize = (8,4))
    plt.plot(D, linewidth = 2, color = 'blue')
    plt.yscale('log')
    plt.grid()
    plt.xlabel('index')
    plt.ylabel('singlular values')
    plt.title('Decay of sigular values')
    #plt.show(block=False)
    plt.savefig('../figs/singularDecay.png')
    plt.close()

    for i in range(4):
        col = np.reshape(U[:,i], (V.shape[0], 5))
        pltcontour('basis', i+1, V, E, col[:,0], 'Density', 'den')
        pltcontour('basis', i+1, V, E, col[:,1], 'x-momentum', 'xm')
    
    ##############################################################
    #   
    # ************************* pt 3 *****************************
    nbset = np.array([2, 4, 8, 16])
    testset = np.array([55, 90])
    
    print('pt3\n')
    for nb in nbset:
        print('nb =', nb, '\n')
        L2error = np.zeros(100)
        for i in range(100):
            data = np.loadtxt(fngen("test", i+1))
            data = np.reshape(data, (5 * V.shape[0], 1))
            projss = np.zeros(data.shape)
            for t in range(nb):
                basis = U[:,t].reshape(data.shape)
                projss += np.dot(data.T, basis) * basis
            L2error[i] = np.linalg.norm((data - projss))
            # plot M for test snapshorts 55 and 90
            for j in range(len(testset)):
                if i == testset[j]-1:
                    projss = np.reshape(projss, (V.shape[0], 5))
                    projM = MsnapshotVisual(projss, 'projectedTest(n='+str(nb)+')', testset[j], V, E)
                    print('projTest', testset[j])
                    print('The corner values:')
                    print(projM)
                    print('The exact Mach number:')
                    print(testMa[testset[j]-1, 0],'\n')
        # plot (M, a) contour
        f = plt.figure(figsize = (7, 5))
        Mspan = testMa[:10,0]
        aspan = np.zeros((1,10))
        for i in range(10): 
            aspan[0,i] = testMa[i*10, 1]
        MM, AA = np.meshgrid(Mspan, aspan)
        plt.contourf(MM, AA, L2error.reshape((10, 10)))
        plt.xlabel(r'Mach number, $M$')
        plt.ylabel(r'angle of attack, $alpha$, (degrees)')
        plt.title('L2 error in (M, a) space, nb = ' + str(nb))
        plt.colorbar()
        #plt.show()
        plt.savefig('../figs/MA_nb' + str(nb) + '.png')
        plt.close()
    
    ##############################################################
    #   
    # ************************* pt 4 *****************************
    print('pt4\n')
    for nb in nbset:
        print('nb =', nb, '\n')

        # calculate exact c_i for tranining data
        projpar = np.zeros((20, nb))
        for i in range(20):
            data = np.loadtxt(fngen("train", i+1))
            data = np.reshape(data, (5 * V.shape[0], 1))
            for t in range(nb):
                basis = U[:,t].reshape(data.shape)
                par = np.dot(data.T, basis)
                projpar[i, t] = par
        projpar = np.reshape(projpar, (20 * nb, 1))

        # Linear coefficient
        print('Linear')
        nc = 3
        bd = []
        X = sparse.lil_matrix((nb*20, nb*nc))
        for i in range(20):
            for j in range(nb):
                X[nb*i+j, j*nc] = 1
                X[nb*i+j, j*nc+1] = trainMa[i,0]
                X[nb*i+j, j*nc+2] = trainMa[i,1]

        coefflr = sparse.linalg.lsqr(X, projpar)
        coefflr = np.reshape(coefflr[0], (nb, nc))
        print('coeff:')
        print(coefflr,'\n')

        # Test cases interperation
        L2errorlr = np.zeros(100)
        for i in range(100):
            data = np.loadtxt(fngen("test", i+1))
            data = np.reshape(data, (5 * V.shape[0], 1))
            projss = np.zeros(data.shape)
            for t in range(nb):
                basis = U[:,t].reshape(data.shape)
                projss += (coefflr[t, 0] + coefflr[t, 1]*testMa[i,0] \
                            + coefflr[t, 2]*testMa[i,1]) * basis
            L2errorlr[i] = np.linalg.norm((data - projss))/np.sqrt(7045*5)
            
            # plot M for test snapshorts 55 and 90
            for j in range(len(testset)):
                if i == testset[j]-1:
                    print(i, L2errorlr[i])
                    projss = np.reshape(projss, (V.shape[0], 5))
                    projM = MsnapshotVisual(projss, 'approxLrTest(n='+str(nb)+')', testset[j], V, E)
                    print('approxLrTest', testset[j])
                    print('The corner values:')
                    print(projM)
                    print('The exact Mach number:')
                    print(testMa[testset[j]-1, 0],'\n')

        # plot (M, a) contour
        f = plt.figure(figsize = (7, 5))
        Mspan = testMa[:10,0]
        aspan = np.zeros((1,10))
        for i in range(10): 
            aspan[0,i] = testMa[i*10, 1]
        MM, AA = np.meshgrid(Mspan, aspan)
        plt.contourf(MM, AA, L2errorlr.reshape((10, 10)))
        plt.xlabel(r'Mach number, $M$')
        plt.ylabel(r'angle of attack, $alpha$, (degrees)')
        plt.title('L2 error in (M, a) space using linear approximation, nb = ' + str(nb))
        plt.colorbar()
        #plt.show()
        plt.savefig('../figs/MAapproxLr_nb' + str(nb) + '.png')
        plt.close()

        # Quadratic coefficient
        print('Quadratic')
        bd = []
        nc = 6
        X = sparse.lil_matrix((nb*20, nb*nc))
        for i in range(20):
            for j in range(nb):
                X[nb*i+j, j*nc] = 1
                X[nb*i+j, j*nc+1] = trainMa[i,0]
                X[nb*i+j, j*nc+2] = trainMa[i,1]
                X[nb*i+j, j*nc+3] = trainMa[i,0]**2
                X[nb*i+j, j*nc+4] = trainMa[i,0]*trainMa[i,1]
                X[nb*i+j, j*nc+5] = trainMa[i,1]**2

        coeffqua = sparse.linalg.lsqr(X, projpar)
        coeffqua = np.reshape(coeffqua[0], (nb, nc))
        print('coeff:')
        print(coeffqua,'\n')

        # Test cases interperation
        L2errorqua = np.zeros(100)
        for i in range(100):
            data = np.loadtxt(fngen("test", i+1))
            data = np.reshape(data, (5 * V.shape[0], 1))
            projss = np.zeros(data.shape)
            for t in range(nb):
                basis = U[:,t].reshape(data.shape)
                projss += (coeffqua[t, 0] + coeffqua[t, 1]*testMa[i,0] \
                         + coeffqua[t, 2]*testMa[i,1] + coeffqua[t, 3]*testMa[i,0]**2 \
                         + coeffqua[t, 4]*testMa[i,0]*testMa[i,1] \
                         +coeffqua[t, 5]*testMa[i,1]**2) * basis
            L2errorqua[i] = np.linalg.norm((data - projss))/np.sqrt(7045*5)
            
            # plot M for test snapshorts 55 and 90
            for j in range(len(testset)):
                if i == testset[j]-1:
                    print(L2errorqua[i])
                    projss = np.reshape(projss, (V.shape[0], 5))
                    projM = MsnapshotVisual(projss, 'approxQuadTest_n='+str(nb), testset[j], V, E)
                    print('approxQuaTest', testset[j])
                    print('The corner values:')
                    print(projM)
                    print('The exact Mach number:')
                    print(testMa[testset[j]-1, 0],'\n')

        # plot (M, a) contour
        f = plt.figure(figsize = (7, 5))
        Mspan = testMa[:10,0]
        aspan = np.zeros((1,10))
        for i in range(10): 
            aspan[0,i] = testMa[i*10, 1]
        MM, AA = np.meshgrid(Mspan, aspan)
        plt.contourf(MM, AA, L2errorqua.reshape((10, 10)))
        plt.xlabel(r'Mach number, $M$')
        plt.ylabel(r'angle of attack, $alpha$, (degrees)')
        plt.title('L2 error in (M, a) space using quadratic approximation, nb = ' + str(nb))
        plt.colorbar()
        #plt.show()
        plt.savefig('../figs/MAapproxQua_nb' + str(nb) + '.png')
        plt.close()

if __name__ == "__main__":
    main()
