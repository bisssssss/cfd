import numpy as np
import matplotlib.pyplot as plt

def plotstate (V, E, U, sfield , ftitle , fname):
	# plots state contours using tricontourf
	f = plt.figure(figsize =(8,4))
	F = getField(U, sfield );
	plt.tricontourf(V [:,0], V [:,1], E, F, 20)
	dosave = not not fname
	plt.axis('equal'); plt.axis([-0.3, 1.5,-0.5, 0.5])
	plt.colorbar()
	if(not not ftitle): plt.title(ftitle, fontsize =16)
	plt.tick_params(axis='both', labelsize =12)
	f.tight_layout(); #plt.show(block=(not dosave));
	if(dosave): plt.savefig(fname)
	plt.close(f)

def getField(U, field ):
	r, ru, rv, rE = [U[:, i ] for i in range(4)]
	g = 1.4; s = field.lower()
	V = np.sqrt(ru**2 + rv**2)/r # speed
	p = (g - 1.)*(rE - 0.5*r*V**2) # pressure
	if (s == 'mach'):
		c = np.sqrt(g*p/r)
		return V/c
	elif (s == 'pressure'):
		return p
	elif (s == 'density'):
		return r
	elif (s == 'xmomentum'):
		return ru

def m1():
	nsnap = 20 # number of training snapshots
	nnode = 7045 # number of mesh nodes
	sr = 5 # state rank
	ndof = nnode * sr # total number of degrees of freedom
	S = np.zeros((ndof,nsnap), dtype=np.float); # snapshots
	# loop over snapshots and load them in
	for isnap in range(nsnap):
		v = np.loadtxt('../runs/train_%d_state.txt'%(isnap+1))
		S[:, isnap] = v.reshape(ndof);
	# call built−in SVD calculation (not with full matrices)
	U,s,V = np.linalg.svd(S, full_matrices =False);
	# save basis functions to a file
	np.savetxt('Basis.txt', U, fmt='%.10E')
	# make a plot of the singular values
	f = plt.figure(figsize =(8,6))
	plt.semilogy(range(nsnap), s, 'o-', linewidth=1, markersize=8, color='blue')
	plt.xlabel(r'mode number', fontsize=16)
	plt.ylabel(r'singular value', fontsize =16)
	plt.figure (f.number); plt.grid()
	plt.tick_params(axis='both', labelsize =12)
	f.tight_layout();
	#plt.show(block=False);
	plt.savefig('singvals.pdf')
	plt.close(f)

	# plot basis vectors
	V = np.loadtxt('V.txt')
	E = np.loadtxt('E.txt'); E = E - 1; # make 0−based
	for i in range(4):
		u = U[:,i ]
		plotstate (V, E, u.reshape((nnode,sr)), "density",
		"mode_%d: density"%(i), "mode_%d_density.pdf"%(i))
		plotstate (V, E, u.reshape((nnode,sr)), "xmomentum",
		"mode_%d: $x$−momentum"%(i), "mode_%d_xmom.pdf"%(i))

def m2():
  m1();
  nsnap = 20 # number of snapshots
  nnode = 7045 # number of mesh nodes
  ntest = 100 # number of test points
  sr = 5 # state rank
  ndof = nnode * sr # total number of dofs
  B = np.loadtxt('Basis.txt') # load basis vectors
  nbv = [2, 4, 8, 16] # numbers of basis vectors
  mag = np.zeros(16) # for storing dot products
  v = np.zeros(ndof) # for reconstructing the state
  N = 10 # number of mach/alpha points (1d)
  mv = np.linspace(.5, .8, N) # Mach points
  av = np.linspace(0, 4, N) # alpha points
  M, A = np.meshgrid(mv, av) # grid of points
  meshV = np.loadtxt('V.txt') # load verices, and elements
  meshE = np.loadtxt('E.txt'); meshE = meshE - 1; # make 0−based
  # loop over runs ( different numbers of basis vectors)
  for inbv in range(4):
    nb = nbv[inbv]
    E = np.zeros((N,N)); # for storing the error
    print("\nrunning with nb = %d\n"%(nb))
    for itest in range(ntest):
      s = np.loadtxt('../runs/test_%d_state.txt'%(itest+1))
      t = s.reshape(ndof)
      for j in range(nb):
        mag[j] = np.dot(B[:,j ], t) # dot products with basis vectors
      v *= 0;
      for j in range(nb):
        v = v + B[:,j]*mag[j] # reconstruction
      err = np.sqrt(np.sum((v-t)**2)/ndof) # L2 error calculation
      E[itest // N, itest%N] = err; # store error
      # plot requested state
      if (( itest == 54) or (itest == 89)):
        plotstate (meshV, meshE, v.reshape(nnode,sr), "mach",
                   "nb=%d,_itest=%d:_Mach"%(nb, itest+1),
                   "recon_nb%d_test%d.pdf"%(nb, itest+1))

    # save error matrix
    np.savetxt('E B%d.txt'%(nb), E, fmt='%.10E');
    # make a contour plot of the error
    f = plt.figure(figsize =(8,4))
    plt.contourf(M, A, E, 30)
    plt.title (r'%d basis vectors'%(nb), fontsize=20)
    plt.xlabel(r'Mach number', fontsize=16)
    plt.ylabel(r'angle of attack (deg)', fontsize =16)
    plt.clim(0, 0.15)
    plt.colorbar()
    f.tight_layout(); plt.show(block=False); plt.savefig('E B%d.pdf'%(nb))
    plt.close(f)

def main():
	m2();
	nsnap = 20 # number of snapshots
	nnode = 7045 # number of mesh nodes
	ntest = 100 # number of test points
	sr = 5 # state rank
	ndof = nnode * sr # total number of dofs
	B = np.loadtxt('Basis.txt') # load basis vectors
	Malpha = np.loadtxt('Malpha.txt') # load training point info
	nbv = [2, 4, 8, 16] # number of basis vectors to use
	smodel = ['Linear', 'Quadratic'] # names of models
	meshV = np.loadtxt('V.txt') # vertices , elements of mesh
	meshE = np.loadtxt('E.txt'); meshE = meshE - 1; # make 0−based
	for imodel in range(2): # loop over models
		nm = 3 * (imodel+1) # number of unknowns per coefficient
		for inbv in range(4): # loop over different numbers of basis vectors
			nb = nbv[inbv] # number of basis vectors

			# construct a least−squares system for the coefficients
			RHS = np.zeros((nsnap, nb));
			LHS = np.zeros((nsnap, nm));
			for isnap in range(nsnap): # build system one row at a time
				s = np.loadtxt('../runs/train_%d_state.txt'%(isnap+1))
				t = s.reshape(ndof)
				M = Malpha[isnap,0]; alpha = Malpha[isnap,1]
				if (imodel == 0):
					LHS[isnap,:] = [1, M, alpha]
				else:
					LHS[isnap,:] = [1, M, alpha, M*M, M*alpha, alpha*alpha]
				for j in range(nb):
					RHS[isnap,j] = np.dot(B[:,j ], t) # RHS is actual projection

			# solve least−squares system, and save data
			C = np.linalg.lstsq(LHS, RHS)[0]
			np.savetxt('coeff_model%d_nb%d.txt'%(imodel, nb), np.transpose(C), fmt='%13.5E')
			# reconstruct states at test points
			N = 10
			mv = np.linspace(.5, .8, N)
			av = np.linspace(0, 3, N)
			M, A = np.meshgrid(mv, av)
			E = np.zeros((N,N));
			v = np.zeros(ndof)
			for itest in range(ntest):
				s = np.loadtxt('../runs/test_%d_state.txt'%(itest+1))
				t = s.reshape(ndof)
				mach = mv[itest%N]
				alpha = av[itest//N]
				if (imodel == 0):
					w = [1, mach, alpha]
				else:
					w = [1, mach, alpha, mach*mach, mach*alpha, alpha*alpha]
				v *= 0.
				for j in range(nb):
					mag = 0.
					for k in range(nm):
						mag += w[k]*C[k,j]
					v += B[:,j]*mag
				err = np.sqrt(np.sum((v - t)**2)/ndof)
				E[itest//N, itest%N] = err;
				if (( itest == 54) or (itest == 89)):
					print(itest, " ", err)
					plotstate (meshV, meshE, v.reshape(nnode,sr), "mach",
						"%s_model:_nb=%d,_itest=%d_(Mach)"%(smodel[imodel],
						nb, itest +1), "model%d_nb%d_test%d.pdf"%(imodel,nb,
						itest +1))
				# save errors , make a contour plot
			np.savetxt('ME%d_B%d.txt'%(imodel, nb), E, fmt='%.10E');
			f = plt.figure ( figsize =(8,4))
			plt.contourf(M, A, E, 30)
			plt.title (r'%s model: %d basis vectors'%(smodel[imodel],nb), fontsize=20)
			plt.xlabel(r'Mach number', fontsize=16)
			plt.ylabel(r'angle of attack (deg)', fontsize =16)
			plt.clim(0, 0.15); plt.colorbar()
			f.tight_layout(); plt.show(block=False); plt. savefig ('ME%d B%d.pdf'%(imodel,nb))
			plt.close(f)






if __name__ == "__main__":
  main()
