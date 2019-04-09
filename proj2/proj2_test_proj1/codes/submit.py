# A.1
def HLLEflux(elemL, lk, elemR):

	hflux = np.empty(4, dtype=float);
	ul = np.dot(np.array([elemL.u[1]/elemL.u[0], elemL.u[2]/elemL.u[0]]), elemL.nvec[lk])
	ur = np.dot(np.array([elemR.u[1]/elemR.u[0], elemR.u[2]/elemR.u[0]]), elemL.nvec[lk])
	sLmin = np.minimum(0., ul - elemL.get_c())
	sLmax = np.maximum(0., ul + elemL.get_c())
	sRmin = np.minimum(0., ur - elemR.get_c())
	sRmax = np.maximum(0., ur + elemR.get_c())
	smin = np.minimum(sLmin, sRmin)
	smax = np.maximum(sLmax, sRmax)

	for i in range(4):
		FL = np.dot(elemL.flux[i], elemL.nvec[lk])
		FR = np.dot(elemR.flux[i], elemL.nvec[lk])
		hflux[i] = 1./2.* (FL + FR) - 1./2.*(smax+smin)/(smax-smin)*(FR-FL)+smax*smin/(smax-smin)*(elemR.u[i] - elemL.u[i])
	return hflux

# A.2
def test_HLLE_flux():

	uexp = np.array([0.7648,   -0.1605,    0.0184,    2.5855])
	u0 = np.array([0.7705,   0.0008,   0.0013,    2.5500])
	usub1 = np.array([0.9526,    0.3888,    0.0075,    3.2496])
	usub2 = np.array([0.6988,    0.3349,    0.0222,    2.3786])
	utras = np.array([0.9583,    1.0133,    0.1550,   3.4105])
	usup1 = np.array([0.5731,    0.9746,  -0.0237,   2.1847])
	usup2 = np.array([0.5815,    0.9832,   -0.0130,    2.2183])

	# Node idx = -1 means test nodes (not exists in a CFD mesh
	elem1 = Element(-1, Node(-1, 0, 0),  Node(-1, 1, 0),  Node(-1, 0, 1))
	elem2 = Element(-1, Node(-1, 0, 0),  Node(-1, 0, 1),  Node(-1, -1, 0))
	elem3 = Element(-1, Node(-1, 1, 0),  Node(-1, 2, 2.5),  Node(-1, 0, 1))
	elem4 = Element(-1, Node(-1, 0, 1),  Node(-1, 2, 2.5),  Node(-1, 0, 2))

	elem1.update_state(uexp) 
	elem2.update_state(u0)
	print(HLLEflux(elem1, 1, elem2)) # test1
	print(HLLEflux(elem2, 2, elem1)) # test2 

	elem3.update_state(usub1)
	elem4.update_state(usub2)
	print(HLLEflux(elem3, 0, elem4)) # test3

	elem4.update_state(utras)
	print(HLLEflux(elem3, 0, elem4)) # test4

	elem3.update_state(usup1)
	elem4.update_state(usup2)
	print(HLLEflux(elem3, 0, elem4)) # test5
	print(HLLEflux(elem1, 0, elem3)) # test6

	elem2.update_state(uexp)
	print(HLLEflux(elem1, 1, elem2)) # test5
	for i in range(4):
		print(np.dot(elem1.flux[i], elem1.nvec[1]))

	print_elem(elem1)

# A.3
def test_proj1():

	# Read mesh
	V = np.loadtxt('../mesh/proj1/V.txt')
	E = np.loadtxt('../mesh/proj1/E.txt')
	E[:,:] = E[:,:]-1
	nV = len(V)
	nE = len(E)
	data = np.loadtxt('../mesh/proj1/test_90_state.txt')

	# Data structure construction
	Elements = []
	for i in range(nE):
		Elements.append(Element(i, Node(E[i][0], V[E[i][0]][0], V[E[i][0]][1]), \
			 					Node(E[i][1], V[E[i][1]][0], V[E[i][1]][1]), \
			 					Node(E[i][2], V[E[i][2]][0], V[E[i][2]][1])))
		u = np.empty(4, dtype=float)
		utemp = (data[[E[i][0]],0:4].T +data[[E[i][1]],0:4].T+data[[E[i][2]],0:4].T)/3
		for j in range(4):
			u[j] = utemp[j][0]
		Elements[i].update_state(u)

	# Build edge table, find neighbors of each element
	Edges = []
	EdgeTable = np.zeros((nV, nV, 4), dtype=int); EdgeTable[:,:] = -1

	for i in range(nE):
		for k in range(3):
			elemv2 = [E[i,k-2], E[i,k-1]]
			if elemv2[0] > elemv2[1]:
				elemv2 = [elemv2[1], elemv2[0]]
			check = EdgeTable[elemv2[0], elemv2[1]]
			if check[0] != -1:
				existi = check[0]
				existk = check[1]
				Edges.append([existi, existk, i, k])
				EdgeTable[elemv2[0], elemv2[1]] = [existi, existk, i, k]
				Elements[i].neighbor[k] = existi
				Elements[existi].neighbor[existk] = i
			else:
				EdgeTable[elemv2[0], elemv2[1], 0] = i
				EdgeTable[elemv2[0], elemv2[1], 1] = k

	nD = len(Edges)

	for i in range(nE):
		Elements[i].clear_residusl()

	for i in range(nD):
		hflux = HLLEflux(Elements[Edges[i][0]], Edges[i][1], Elements[Edges[i][2]])
		Elements[Edges[i][0]].add_residual(hflux, Edges[i][1])
		Elements[Edges[i][2]].add_residual(-hflux, Edges[i][3])

	for i in range(nE):
		if i == 1000:
			print(Elements[i].residual)








































def test_HLLE_flux():

	uexp = np.array([0.7648,   -0.1605,    0.0184,    2.5855])
	u0 = np.array([0.7705,   0.0008,   0.0013,    2.5500])
	usub1 = np.array([0.9526,    0.3888,    0.0075,    3.2496])
	usub2 = np.array([0.6988,    0.3349,    0.0222,    2.3786])
	utras = np.array([0.9583,    1.0133,    0.1550,   3.4105])
	usup1 = np.array([0.5731,    0.9746,  -0.0237,   2.1847])
	usup2 = np.array([0.5815,    0.9832,   -0.0130,    2.2183])

	# Node idx = -1 means test nodes (not exists in a CFD mesh
	elem1 = Element(-1, Node(-1, [0, 0]),  Node(-1, [1, 0]),  Node(-1, [0, 1]))
	elem2 = Element(-1, Node(-1, [0, 0]),  Node(-1, [0, 1]),  Node(-1, [-1, 0]))
	elem3 = Element(-1, Node(-1, [1, 0]),  Node(-1, [2, 2.5]),  Node(-1, [0, 1]))
	elem4 = Element(-1, Node(-1, [0, 1]),  Node(-1, [2, 2.5]),  Node(-1, [0, 2]))

	elem1.update_state(uexp) 
	elem2.update_state(u0)
	print(HLLEflux(elem1, 1, elem2)) # test1
	print(HLLEflux(elem2, 2, elem1)) # test2 

	elem3.update_state(usub1)
	elem4.update_state(usub2)
	print(HLLEflux(elem3, 0, elem4)) # test3

	elem4.update_state(utras)
	print(HLLEflux(elem3, 0, elem4)) # test4

	elem3.update_state(usup1)
	elem4.update_state(usup2)
	print(HLLEflux(elem3, 0, elem4)) # test5
	print(HLLEflux(elem1, 0, elem3)) # test6

	elem2.update_state(uexp)
	print(HLLEflux(elem1, 1, elem2)) # test5
	for i in range(4):
		print(np.dot(elem1.flux[i], elem1.nvec[1]))

	print_elem(elem1)

	#test_proj1()

def test_proj1():

	# Constants
	tolerence = 1e-5
	Resdnorm = np.inf
	niter = 10
	CFL = 0.6
	gamma = 1.4
	M_inf = 8.000000000000000e-01
	alpha = 3.555555555555555e+00 * np.pi/180
	u_inf = np.array([1.00001056e+00, 9.98090339e-01, 6.20025107e-02, 3.29022887e+00])

	# Read mesh
	V = np.loadtxt('../mesh/proj1/V.txt')
	E = np.loadtxt('../mesh/proj1/E.txt')
	E[:,:] = E[:,:]-1
	nV = len(V)
	nE = len(E)
	data = np.loadtxt('../mesh/proj1/test_90_state.txt')

	# Data structure construction
	Elements = []
	for i in range(nE):
		Elements.append(Element(i, Node(E[i][0], V[E[i][0],:]), Node(E[i][1], V[E[i][1],:]), Node(E[i][2], V[E[i][2],:])))
		u = np.empty(4, dtype=float)
		utemp = (data[[E[i][0]],0:4].T +data[[E[i][1]],0:4].T+data[[E[i][2]],0:4].T)/3
		for j in range(4):
			u[j] = utemp[j][0]
		Elements[i].update_state(u)
		Elements[i].gamma = gamma
		Elements[i].constR = 287.

	########################### Edge & Neighbor & Boundary ##########################

	# Build edge table, find neighbors of each element
	Edges = []
	EdgeTable = np.zeros((nV, nV, 4), dtype=int); EdgeTable[:,:] = -1

	for i in range(nE):
		for k in range(3):
			elemv2 = [E[i,k-2], E[i,k-1]]
			if elemv2[0] > elemv2[1]:
				elemv2 = [elemv2[1], elemv2[0]]
			check = EdgeTable[elemv2[0], elemv2[1]]
			if check[0] != -1:
				existi = check[0]
				existk = check[1]
				Edges.append([existi, existk, i, k])
				EdgeTable[elemv2[0], elemv2[1]] = [existi, existk, i, k]
				Elements[i].neighbor[k] = existi
				Elements[existi].neighbor[existk] = i
			else:
				EdgeTable[elemv2[0], elemv2[1], 0] = i
				EdgeTable[elemv2[0], elemv2[1], 1] = k

	nD = len(Edges)

	B = []
	B0 = []
	B1 = []
	B2 = []
	B3 = []
	B4 = []

	for i in range(nE):
		for k in range(3):
			if Elements[i].neighbor[k] < 0:
				if np.abs(Elements[i].node[k-1].x + 2000) < 1e-5:
					B3.append([Elements[i].node[k-1].idx, Elements[i].node[k-2].idx])
				elif np.abs(Elements[i].node[k-1].y + 2000) < 1e-5:
					B0.append([Elements[i].node[k-1].idx, Elements[i].node[k-2].idx])
				elif np.abs(Elements[i].node[k-1].y - 2000) < 1e-5:
					B2.append([Elements[i].node[k-1].idx, Elements[i].node[k-2].idx])
				elif np.abs(Elements[i].node[k-1].x - 2000) < 1e-5:
					B1.append([Elements[i].node[k-1].idx, Elements[i].node[k-2].idx])
				else:
					B4.append([Elements[i].node[k-1].idx, Elements[i].node[k-2].idx])
	print(len(B0), len(B1), len(B2), len(B3), len(B4))
	print(nE, nV, nD)
	B.append(B0)
	B.append(B1)
	B.append(B2)
	B.append(B3)
	B.append(B4)
	nB = len(B)


	Boundaries = []
	for bi in range(nB):
		bd = B[bi]
		bdsub = np.empty((len(bd),2), dtype=int)
		for j in range(len(bd)):
			elemv2 = [bd[j][0], bd[j][1]]
			if elemv2[0] > elemv2[1]:
				elemv2 = [elemv2[1], elemv2[0]]
			check = EdgeTable[elemv2[0], elemv2[1]]
			if check[0] == -1:
				print("Error: boundary not specified before %r" %a)
			if check[2] != -1:
				print("Error: boudnary checked twice %r" %a)
			bdsub[j,0] = check[0]; bdsub[j,1] = check[1]
			Elements[check[0]].ifbd[check[1]] = bi
		Boundaries.append(bdsub)

	###########################################################################

	'''
	Mset = np.zeros((nE,1))
	for i in range(nE):
		Mset[i] = Elements[i].M
	pltcontour(V, E, Mset, 'Mach number', 0)
	'''

	############################ run CFD Iteration #################################
	
	f = plt.figure(figsize=(16,12))
	plt.xlabel("iteration")
	plt.ylabel("Residual norm")
	plt.title("Convergence plot")
	Resdnormvec = []

	for iteri in range(niter):

		for i in range(nE):
			Elements[i].clear_residusl()

		for i in range(nD):
			hflux = HLLEflux(Elements[Edges[i][0]], Edges[i][1], Elements[Edges[i][2]])
			Elements[Edges[i][0]].add_residual(hflux, Edges[i][1])
			Elements[Edges[i][2]].add_residual(-hflux, Edges[i][3])

		'''
		for i in range(5):
			print(Elements[Boundaries[4][i][0]].residual)
		'''

		for bdi in range(5):
			for i in range(len(Boundaries[bdi])):
				hflux = boundary_flux(bdi, Elements[Boundaries[bdi][i][0]], Boundaries[bdi][i][1], u_inf)
				Elements[Boundaries[bdi][i][0]].add_residual(hflux, Boundaries[bdi][i][1])
		'''
		for i in range(5):
			print(Elements[Boundaries[4][i][0]].residual)
		'''

		# calculate residual norm and store in a list
		Resdnorm = 0.0
		for i in range(nE):
			for k in range(4):
				Resdnorm += Elements[i].residual[k]**2
		Resdnorm = np.sqrt(Resdnorm/(4*nE))
		Resdnormvec.append(Resdnorm)
		'''
		for i in range(nE):
			for k in range(4):
				if (Elements[i].residual[k] > 1e-3):
					for j in range(3):
						if Elements[i].ifbd[j] >= 0:
							print_elem(Elements[i])
					break
		'''
		# monitor and log the Residual norm at current iteration
		print("Iter - %d" %iteri)
		print("  Residual: %r" %Resdnorm)
		plt.plot(Resdnormvec, color = 'blue')
		#plt.pause(0.5) #edit

		if Resdnorm < tolerence:
			break

		# update state EDIT
		for i in range(nE):
			di = 2. * Elements[i].area/Elements[i].perimeter
			elemu = np.array([Elements[i].u[1]/Elements[i].u[0], Elements[i].u[2]/Elements[i].u[0]])
			si = 0.0
			for k in range(3):
				if Elements[i].ifbd[k] < 0:
					sie = max_wave_speed(Elements[i], k, Elements[Elements[i].neighbor[k]])
					si += sie * Elements[i].el[k]/Elements[i].perimeter
			
			localdt = CFL * di / si
			Elements[i].update_state(Elements[i].u -localdt/Elements[i].area * Elements[i].residual)
			if(Elements[i].get_T() < 0):
				print_elem(Elements[i])
		
	
	plt.show(block = False)