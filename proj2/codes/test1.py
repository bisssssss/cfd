import numpy as np
from readgri import readgri
import copy
import math
import matplotlib.pyplot as plt
from printplot import print_elem, pltcontour, pltmesh
from time import gmtime, strftime

class Node():
	def __init__(self, i, x_in, y_in):
		self.x = x_in
		self.y = y_in
		self.idx = i

class Element():

	gamma = 1.3
	constR = 8314./44.

	def __init__(self, idx_in, n1_in, n2_in, n3_in):
		self.node = [n1_in, n2_in, n3_in]
		self.idx = idx_in
		self.edge = np.empty((3,2), dtype=float)
		self.nvec = np.empty((3,2), dtype=float)
		for i in range(3):
			self.edge[i,:] = np.array([self.node[i-1].x-self.node[i-2].x, self.node[i-1].y-self.node[i-2].y])
		self.el = np.array([np.linalg.norm(self.edge[0]), np.linalg.norm(self.edge[1]), np.linalg.norm(self.edge[2])])
		for i in range(3):
			self.nvec[i,:] = np.array([self.edge[i][1]/self.el[i], -self.edge[i][0]/self.el[i]])
		self.area = np.abs(np.cross(self.edge[0], self.edge[1]) / 2)
		self.perimeter = np.sum(self.el)
		self.neighbor = np.empty(3, dtype=int);	self.neighbor[:] = np.nan
		self.residual = np.zeros(4, dtype=float)
		self.u = np.empty((4,1), dtype=float); self.u[:] = np.nan
		self.flux = np.empty((4,1,2), dtype=float); self.flux[:,:] = np.nan
		self.M = np.nan
		self.ifbd = np.empty(3, dtype=int);	self.ifbd[:] = -1
		self.ifadap = np.empty(3, dtype=int);	self.ifadap[:] = -1

	def update_state(self, u_in):
		self.u = copy.deepcopy(u_in)
		self.M = self.get_speed()/self.get_c()
		self.set_flux()

	def add_residual(self, hflux, k):
		for i in range(4):
			self.residual[i] += hflux[i]*self.el[k]

	def set_flux(self):
		self.flux[0] = np.array([self.u[1],self.u[2]])
		self.flux[1] = np.array([self.u[1]**2/self.u[0]+self.get_p(), self.u[1]*self.u[2]/self.u[0]])
		self.flux[2] = np.array([self.u[1]*self.u[2]/self.u[0], self.u[2]**2/self.u[0]+self.get_p(),])
		self.flux[3] = np.array([(self.u[3]+self.get_p())*self.u[1]/self.u[0], (self.u[3]+self.get_p())*self.u[2]/self.u[0]])

	def clear_residusl(self):
		self.residual = np.zeros(4, dtype=float)

	def get_T(self):
		return self.get_p()/self.u[0]/self.constR

	def get_speed(self):
		return np.sqrt(self.u[1]**2 +self.u[2]**2)/self.u[0]

	def get_c(self):
		return np.sqrt(self.gamma*self.constR*self.get_T())

	def get_p(self):
		return (self.gamma-1.)*(self.u[3] - 1./2. * (self.u[1]**2 +self.u[2]**2)/self.u[0])


# return HLLEflux
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

def full_state_flux(idx, elem, lk, u_in):

	bdelem = Element(-idx, Node(-1, 0, 1),  Node(-1, 2, 3),  Node(-1, 4, 5))
	bdelem.update_state(u_in)
	return HLLEflux(elem, lk, bdelem)

def ivsc_wall_flux(elem, lk):

	elemu = np.array([elem.u[1]/elem.u[0], elem.u[2]/elem.u[0]])
	bdu = np.array(elemu) - np.dot(elemu, elem.nvec[lk])*np.array(elem.nvec[lk])
	bdunorm = np.linalg.norm(bdu)
	pb = (elem.gamma-1.)*(elem.u[3] - 1./2. * elem.u[0]*(bdu[0]**2 + bdu[1]**2))
	flux = np.array([0, pb*elem.nvec[lk][0], pb*elem.nvec[lk][1], 0])
	return flux

def sups_flux(elem, lk):

	hflux = np.empty(4, dtype=float);
	for i in range(4):
		hflux[i] = np.dot(elem.flux[i], elem.nvec[lk])
	return hflux

def boundary_flux(bdi, elem, lk, u_in):
	if bdi == 1:
		return sups_flux(elem, lk)
	elif bdi == 4:
		return ivsc_wall_flux(elem, lk)
	else:
		return full_state_flux(bdi, elem, lk, u_in)

def max_wave_speed(elemL, lk, elemR):
	ul = np.dot(np.array([elemL.u[1]/elemL.u[0], elemL.u[2]/elemL.u[0]]), elemL.nvec[lk])
	ur = np.dot(np.array([elemR.u[1]/elemR.u[0], elemR.u[2]/elemR.u[0]]), elemL.nvec[lk])
	return np.maximum(abs(ul) + elemL.get_c(), abs(ur)+elemR.get_c())

def max_wave_speed_boundary(bdi, elemL, lk, u_in):

	if bdi == 1: # supersonic boundary
		ul = np.dot(np.array([elemL.u[1]/elemL.u[0], elemL.u[2]/elemL.u[0]]), elemL.nvec[lk])
		return abs(ul) + elemL.get_c()

	if bdi == 4: # capsule boundary
		elemu = np.array([elemL.u[1]/elemL.u[0], elemL.u[2]/elemL.u[0]])
		bdu = np.array(elemu) - np.dot(elemu, elemL.nvec[lk])*np.array(elemL.nvec[lk])
		bdunorm = np.linalg.norm(bdu)
		pb = (elemL.gamma-1.)*(elemL.u[3] - 1./2. * elemL.u[0]*(bdu[0]**2 + bdu[1]**2))
		cR = np.sqrt(elemL.gamma * pb/elemL.u[0])
		ul = np.dot(np.array([elemL.u[1]/elemL.u[0], elemL.u[2]/elemL.u[0]]), elemL.nvec[lk])
		return np.maximum(abs(ul) + elemL.get_c(), cR)

	else: # full state boundary
		elemR = Element(-1, Node(-1, 0, 1),  Node(-1, 2, 3),  Node(-1, 4, 5))
		elemR.update_state(u_in)
		ul = np.dot(np.array([elemL.u[1]/elemL.u[0], elemL.u[2]/elemL.u[0]]), elemL.nvec[lk])
		ur = np.dot(np.array([elemR.u[1]/elemR.u[0], elemR.u[2]/elemR.u[0]]), elemL.nvec[lk])
		return np.maximum(abs(ul) + elemL.get_c(), abs(ur)+elemR.get_c())

def init_Elements(ifread, V, E, alpha, fname):

	################################## Initialization #############################
	
	nE = len(E)
	nV = len(V)

	# Constants
	gamma = 1.3
	M_init = 1.3
	u_init = np.array([1., M_init*math.cos(alpha), M_init*math.sin(alpha), 1./(gamma-1.)/gamma + M_init*M_init/2.])
	if ifread:
		uexist = np.loadtxt(fname)

	# Data structure construction
	Elements = []
	for i in range(nE):
		Elements.append(Element(i, Node(E[i][0], V[E[i][0]][0], V[E[i][0]][1]), \
			 					Node(E[i][1], V[E[i][1]][0], V[E[i][1]][1]), \
			 					Node(E[i][2], V[E[i][2]][0], V[E[i][2]][1])))
		if ifread:
			Elements[i].update_state(uexist[i,:])
		else:
			Elements[i].update_state(u_init)

	return Elements

def init_Edges_Boundaries(V, E, B, Elements):

	########################### Edge & Neighbor & Boundary ##########################

	nE = len(E)
	nB = len(B)
	nV = len(V)

	# Build edge table [nV x nV]
	# Find neighbors of each element
	Edges = []
	EdgeTable = np.zeros((nV, nV, 4), dtype=int); EdgeTable[:,:] = -1

	for i in range(nE):
		for k in range(3):
			elemv2 = [E[i][k-2], E[i][k-1]]
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

	# Build Boundaries connectivity, similar to Edges
	# Update boundary condition for each element
	Boundaries = []
	for bi in range(nB):
		bd = B[bi]
		boundary = []
		for j in range(len(bd)):
			elemv2 = [bd[j][0], bd[j][1]]
			if elemv2[0] > elemv2[1]:
				elemv2 = [elemv2[1], elemv2[0]]
			check = EdgeTable[elemv2[0], elemv2[1]]
			if check[0] == -1:
				print("Error: boundary not detected before")
				return 0
			if check[2] != -1:
				print("Error: boundnary checked twice")
				return 0
			boundary.append([check[0], check[1]])
			Elements[check[0]].ifbd[check[1]] = bi
		Boundaries.append(boundary)

	return Edges, Boundaries	

def aero_dyn(Elements, boundary, u_inf):

	gamma = 1.3
	freestreamv = np.array([u_inf[1]/u_inf[0], u_inf[2]/u_inf[0]])
	pernorm = [-freestreamv[1], freestreamv[0]]
	v_inf = np.linalg.norm(freestreamv); rho_inf = u_inf[0]
	p_inf = (gamma-1.)*(u_inf[3] - 1./2. * (u_inf[1]**2 +u_inf[2]**2)/u_inf[0])
	d = 1.2
	center = [0.8, 0.0]

	Force = np.zeros(2); Moment = 0.0
	cp = np.empty((len(boundary), 2), dtype=float); cp[:,:] = np.nan
	for i in range(len(boundary)):
		elem = Elements[boundary[i][0]]
		pnvec = ivsc_wall_flux(elem, boundary[i][1])[1:3]
		pb = np.linalg.norm(pnvec)
		Force += pnvec*elem.el[boundary[i][1]]
		rvec = np.array([(elem.node[boundary[i][1]-1].x + elem.node[boundary[i][1]-2].x)/2., \
							(elem.node[boundary[i][1]-1].y + elem.node[boundary[i][1]-2].y)/2.])
		Moment += np.cross(rvec, pnvec)*elem.el[boundary[i][1]]
		cp[i,0] = np.arctan2(rvec[1]-center[1], -(rvec[0]-center[0])) / np.pi * 180.
		cp[i,1] = (pb-p_inf)/(1./2.*rho_inf*v_inf**2)

	Drag = np.dot(Force, freestreamv)/np.linalg.norm(freestreamv)
	Lift = np.dot(Force, pernorm)/np.linalg.norm(pernorm)

	cl = Lift/(1./2.*rho_inf*v_inf**2*d)
	cd = Drag/(1./2.*rho_inf*v_inf**2*d)
	cm = Moment/(1./2.*rho_inf*v_inf**2*d**2)
	cpsort_arg = np.argsort(cp[:,0])
	cp = cp[cpsort_arg,:]

	return cl, cd, cm, cp

def runCFD(ifsave, V, E, Elements, Edges, Boundaries, niter, alpha, CFL, tolerence, fname):

	############################ run CFD Iteration #################################

	nE = len(Elements)
	nD = len(Edges)
	nB = len(Boundaries)
	Resdnormvec = []
	clstack = []; cdstack = []; cmstack = []

	# Constants
	M_inf = 8.0
	gamma = 1.3
	u_inf = np.array([1., M_inf*math.cos(alpha), M_inf*math.sin(alpha), 1./(gamma-1.)/gamma + M_inf*M_inf/2.])

	# Convergence plot
	f = plt.figure(figsize=(10,6))
	plt.xlabel("iteration")
	plt.ylabel("Residual norm")
	plt.title("Residual Convergence plot"+fname)
	plt.grid()

	for iteri in range(niter):

		# Clear existing residuals
		for i in range(nE):
			Elements[i].clear_residusl()

		# Add residuals to elements on both side of each interior edge
		for i in range(nD):
			hflux = HLLEflux(Elements[Edges[i][0]], Edges[i][1], Elements[Edges[i][2]])
			Elements[Edges[i][0]].add_residual(hflux, Edges[i][1])
			Elements[Edges[i][2]].add_residual(-hflux, Edges[i][3])

		# Add residuals to elements next to each boundary edge
		for bdi in range(nB):
			boundary = Boundaries[bdi]
			for i in range(len(boundary)):
				hflux = boundary_flux(bdi, Elements[boundary[i][0]], boundary[i][1], u_inf)
				Elements[boundary[i][0]].add_residual(hflux, boundary[i][1])

		# Calculate residual norm and aerodyn coefficients, store values in lists
		Resd = np.empty((nE, 4))
		for i in range(nE):
			for k in range(4):
				Resd[i,k] = Elements[i].residual[k]
		Resdnorm = np.sqrt(sum(sum(Resd * Resd)))
		Resdnormvec.append(Resdnorm)
		cl, cd, cm, cp = aero_dyn(Elements, Boundaries[4], u_inf)
		clstack.append(cl); cdstack.append(cd); cmstack.append(cm)

		# Monitor and log the Residual norm at current iteration
		print("Iter - %d" %iteri)
		print("  Residual: %r" %Resdnorm)
		plt.semilogy(Resdnormvec, color = 'blue')
		#plt.pause(0.05)

		# Check if residual is smaller than tolerence
		if Resdnorm < tolerence:
			break

		# Calculate local time stepping
		localdtvec = np.zeros(nE)

		for i in range(nE):
			di = 2. * Elements[i].area/Elements[i].perimeter
			elemu = np.array([Elements[i].u[1]/Elements[i].u[0], Elements[i].u[2]/Elements[i].u[0]])
			si = 0.0
			for k in range(3):
				if Elements[i].ifbd[k] < 0:
					sie = max_wave_speed(Elements[i], k, Elements[Elements[i].neighbor[k]])
				else:
					sie = max_wave_speed_boundary(Elements[i].ifbd[k], Elements[i], k, u_inf)
				si += sie * Elements[i].el[k]/Elements[i].perimeter
			
			if iteri <= 10:
				localdt = CFL * di / si / 2
			else:
				localdt = CFL * di / si

			localdtvec[i] = localdt

		# Update state
		for i in range(nE):
			uold = copy.deepcopy(Elements[i].u)
			Elements[i].update_state(Elements[i].u - localdtvec[i]/Elements[i].area * Elements[i].residual)
			# Error check
			gett = Elements[i].get_T()
			if gett < 0:
				print("large CFL")
			while (gett < 0):
				localdtvec[i] /= 2
				Elements[i].update_state(uold - localdtvec[i]/Elements[i].area * Elements[i].residual)
				gett = Elements[i].get_T()

	if ifsave:
		usave = np.empty((nE, 4))
		for i in range(nE):
			usave[i,:] = Elements[i].u
		np.savetxt('../cache/last' + fname + '.txt', usave)
		np.savetxt('../cache/'+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'.txt', usave)
		if Resdnorm < tolerence:
			np.savetxt('../cache/steady' + fname + '.txt', usave)

	plt.pause(0.5)
	#plt.show(block = False)
	plt.savefig('../figs/resd_converg.jpg')
	plt.close(f)

	fl = plt.figure(figsize = (10,6))
	plt.plot(clstack);
	plt.xlabel(r'$iteration$', fontsize=16)
	plt.ylabel(r'$c_l$', fontsize=16)
	plt.title('Lift coefficient convergence plot')
	plt.savefig('../figs/cl_converg' + fname + '.jpg')
	plt.show(block = False)
	plt.close(fl)

	fd = plt.figure(figsize = (10,6))
	plt.plot(cdstack);
	plt.xlabel(r'$iteration$', fontsize=16)
	plt.ylabel(r'$c_d$', fontsize=16)
	plt.title('Drag coefficient convergence plot')
	plt.savefig('../figs/cd_converg' + fname + '.jpg')	
	plt.show(block = False)
	plt.close(fd)

	fm = plt.figure(figsize = (10,6))
	plt.plot(cmstack);
	plt.xlabel(r'$iteration$', fontsize=16)
	plt.ylabel(r'$c_m$', fontsize=16)
	plt.title('Moment coefficient convergence plot')
	plt.savefig('../figs/cm_converg' + fname + '.jpg')
	plt.show(block = False)
	plt.close(fm)

	if Resdnorm < tolerence:
		print("cd %r" %cd)
		print("cl %r" %cl)
		print("cm %r" %cm)
		pltcontour(V, E, Elements, 'Converged Mach contour'+fname, 1)
		fp = plt.figure(figsize = (12,6))
		plt.plot(cp[:,0], cp[:,1]);
		plt.xlabel('theta [degree]', fontsize=16)
		plt.ylabel(r'$c_p$', fontsize=16)
		plt.title('Converged pressure coefficient line plot')
		plt.savefig('../figs/cp' + fname + '.jpg')
		plt.show(block = False)
		plt.close(fp)
	return Resdnorm, cd, cl, cm, cp

def update_Elements(V,E,B,Elements,Edges,Boundaries, uexist):

	nE = len(E); nB = len(B); nV = len(V); nD = len(Edges)
	bderr = []
	totledge = []
	refpt = np.zeros((nE,1)); refpt[:,:] = 0

	V1 = []; V2 = []
	for i in range(len(V)):
		V1.append(V[i][0])
		V2.append(V[i][1])
	
	for i in range(nD):
		totledge.append(Edges[i])
		bderr.append(np.abs(Elements[Edges[i][0]].M - Elements[Edges[i][2]].M) * Elements[Edges[i][0]].el[Edges[i][1]])

	for i in range(4):
		boundary = Boundaries[i]
		for j in range(len(boundary)):
			totledge.append(boundary[j])
			bderr.append(0)

	boundary = Boundaries[4]
	for j in range(len(boundary)):
		totledge.append(boundary[j])
		elem =  Elements[boundary[j][0]]
		elemu = np.array([elem.u[1]/elem.u[0], elem.u[2]/elem.u[0]])
		normM = np.dot(elemu, elem.nvec[boundary[j][1]])/elem.get_c()
		bderr.append(np.abs(normM)* elem.el[boundary[j][1]])	

	bderr = np.array(bderr)
	bderrsort_arg = np.argsort(-bderr)


	# flag element edges

	frac = 0.03
	BDifadap = np.empty(len(bderr), dtype=int); BDifadap[:] = -1

	newu = []
	newE = []
	newB = [[],[],[],[],[]]

	for i in range(int(frac*len(bderr))):
		edgeconv = totledge[bderrsort_arg[i]]

		if(len(edgeconv) == 2):
			elem = Elements[edgeconv[0]]
			for k in range(3):
				if elem.ifadap[k] < 0:
					Node1 = elem.node[k-1]
					Node2 = elem.node[k-2]
					V.append([(Node1.x + Node2.x)/2., (Node1.y + Node2.y)/2.])
					Elements[edgeconv[0]].ifadap[k] = len(V)-1
					elemneighbor = elem.neighbor[k]
					if elemneighbor >= 0: # filter boundary edge
						for neighbork in range(3):
							if Elements[elemneighbor].neighbor[neighbork] == edgeconv[0]:
								if Elements[elemneighbor].ifadap[neighbork] >= 0:
									print("Error: Edge flagged twice")
									return 0
								Elements[elemneighbor].ifadap[neighbork] = len(V)-1

		else:
			elem = Elements[edgeconv[0]]
			for k in range(3):
				if elem.ifadap[k] < 0:
					Node1 = elem.node[k-1]
					Node2 = elem.node[k-2]
					V.append([(Node1.x + Node2.x)/2., (Node1.y + Node2.y)/2.])
					Elements[edgeconv[0]].ifadap[k] = len(V)-1
					elemneighbor = elem.neighbor[k]
					if elemneighbor >= 0: # filter boundary edge
						for neighbork in range(3):
							if Elements[elemneighbor].neighbor[neighbork] == edgeconv[0]:
								if Elements[elemneighbor].ifadap[neighbork] >= 0:
									print("Error: Edge flagged twice")
									return 0
								Elements[elemneighbor].ifadap[neighbork] = len(V)-1
			elem = Elements[edgeconv[2]]
			for k in range(3):
				if elem.ifadap[k] < 0:
					Node1 = elem.node[k-1]
					Node2 = elem.node[k-2]
					V.append([(Node1.x + Node2.x)/2., (Node1.y + Node2.y)/2.])
					Elements[edgeconv[2]].ifadap[k] = len(V)-1
					elemneighbor = elem.neighbor[k]
					if elemneighbor >= 0: # filter boundary edge
						for neighbork in range(3):
							if Elements[elemneighbor].neighbor[neighbork] == edgeconv[2]:
								if Elements[elemneighbor].ifadap[neighbork] >= 0:
									print("Error: Edge flagged twice")
									return 0
								Elements[elemneighbor].ifadap[neighbork] = len(V)-1

	for i in range(nE):
		for k in range(3):
			if Elements[i].ifbd[k] >= 0:
				Node1 = Elements[i].node[k-1]
				Node2 = Elements[i].node[k-2]
				if Elements[i].ifadap[k] >= 0:
					newB[Elements[i].ifbd[k]].append([Node1.idx, Elements[i].ifadap[k]])
					newB[Elements[i].ifbd[k]].append([Node2.idx, Elements[i].ifadap[k]])
				else:
					newB[Elements[i].ifbd[k]].append([Node2.idx, Node1.idx])


	# loop over elements to create new triangles
	for i in range(nE):

		elem = Elements[i]
		adapEgidx = []
		for k in range(3):
			if elem.ifadap[k] >= 0:
				adapEgidx.append(k)
		refpt[i] = len(adapEgidx)
		if len(adapEgidx) == 0:
			newE.append(E[elem.idx])
			

		elif len(adapEgidx) == 1:
			idx0 = adapEgidx[0]
			Node0 = elem.node[idx0]
			Node1 = elem.node[idx0-1]
			Node2 = elem.node[idx0-2]
			newidx = elem.ifadap[idx0]
			newE.append([Node0.idx, newidx, Node1.idx])
			newE.append([Node0.idx, Node2.idx, newidx])

		elif len(adapEgidx) == 2:
			idx0 = np.nan
			for k in range(3):
				if elem.ifadap[k] == -1:
					idx0 = k
			Node0 = elem.node[idx0]
			if elem.el[idx0-1] > elem.el[idx0-2]:
				Node1 = elem.node[idx0-1]
				Node2 = elem.node[idx0-2]
				newidx1 = elem.ifadap[idx0-1]
				newidx2 = elem.ifadap[idx0-2]
				newE.append([Node0.idx, newidx1, newidx2])
				newE.append([newidx1, Node1.idx, newidx2])
				newE.append([newidx1, Node2.idx, Node1.idx])

			else:
				Node2 = elem.node[idx0-1]
				Node1 = elem.node[idx0-2]
				newidx1 = elem.ifadap[idx0-2]
				newidx2 = elem.ifadap[idx0-1]
				newE.append([Node0.idx, newidx2, newidx1])
				newE.append([newidx2, Node1.idx, newidx1])
				newE.append([newidx1, Node1.idx, Node2.idx])

			
		elif len(adapEgidx) == 3:
			Node0 = elem.node[0]
			Node1 = elem.node[1]
			Node2 = elem.node[2]
			newidx0 = elem.ifadap[0]
			newidx1 = elem.ifadap[1]
			newidx2 = elem.ifadap[2]
			newE.append([Node0.idx, newidx2, newidx1])
			newE.append([Node1.idx, newidx0, newidx2])
			newE.append([Node2.idx, newidx1, newidx0])
			newE.append([newidx0, newidx1, newidx2])

		else:
			print("Error: triangle has more than 3 edges.")
			return -1

		for k in range(len(adapEgidx)+1):
			newu.append(uexist[i,:])

	'''
	newVplot = np.array(V)
	f = plt.figure(figsize = (16,16))
	plt.axis('equal')
	tcf = plt.tripcolor(V1, V2, E, facecolors=refpt[:,0], edgecolors='w', cmap = 'jet')
	plt.xlabel(r'$x$', fontsize=16)
	plt.ylabel(r'$y$', fontsize=16)
	#plt.xlim(-1.2, 1.5)
	#plt.ylim(-1.2, 1.2)
	plt.colorbar(tcf)
	plt.scatter(newVplot[:,0], newVplot[:,1])
	plt.show()
	'''

	return V, newE, newB, newu

def main():

	################################ TASK 1 ####################################
	
	#test_HLLE_flux()
	#test_proj1()

	################################ TASK 2&3 ##################################
	
	# user set parameters
	alpha = 5.0 / 180. * np.pi
	niter = 1000000
	CFL = 0.3
	tolerence = 0.1
	ifloop = 0

	# read mesh
	Mesh = readgri('../mesh/capsule.gri')
	V = Mesh['V']
	E = Mesh['E']
	B = Mesh['B']

	# Initialize triangles for FVM iteration
	Elements = init_Elements(1, V, E, alpha, '../cache/last.txt')
	Edges, Boundaries = init_Edges_Boundaries(V, E, B, Elements)
	
	'''
	refpt = np.zeros((len(Elements),1))
	nodesct = np.empty((len(Boundaries[4]),2)); nodesct[:,:] = np.nan
	for i in range(len(Boundaries[4])):
		elem =  Elements[Boundaries[4][i][0]]
		lk = Boundaries[4][i][1]
		refpt[Boundaries[4][i][0]] = 1
		node =  Elements[Boundaries[4][i][0]].node[Boundaries[4][i][1]]
		nodesct[i,:] = np.array([node.x, node.y])

	V1 = []; V2 = []
	for i in range(len(V)):
		V1.append(V[i][0])
		V2.append(V[i][1])

	f = plt.figure(figsize = (16,16))
	plt.axis('equal')
	tcf = plt.tripcolor(V1, V2, E, facecolors=refpt[:,0], edgecolors='w', cmap = 'jet')
	plt.scatter(nodesct[:,0], nodesct[:,1])
	plt.xlabel(r'$x$', fontsize=16)
	plt.ylabel(r'$y$', fontsize=16)
	plt.xlim(-1.2, 1.5)
	plt.ylim(-1.2, 1.2)
	plt.colorbar(tcf)
	plt.show()
	'''
	
	# runninng iteration until residual norm is less than tolerence
	Resd, cd, cl, cm, cp = runCFD(0, V, E, Elements, Edges, Boundaries, niter, alpha, CFL, tolerence, '')

	# Loop simulations with a specified iteration number until converged
	if ifloop:
		while (Resd > tolerence):
			Elements = init_Elements(1, V, E, alpha, '../cache/last.txt')
			Edges, Boundaries = init_Edges_Boundaries(V, E, B, Elements)
			Resd, cd, cl, cm, cp = runCFD(1, V, E, Elements, Edges, Boundaries, niter, alpha, CFL, tolerence, '')

	# plot Mach number contour
	#pltmesh(V, E, 'Mach number', 0)

	################################ TASK 4 ####################################
	
	# use set parameters
	alphaset = [10.0]
	nadapset = [1, 3, 5]

	for ia in range(len(alphaset)):

		alpha = alphaset[ia] / 180. * np.pi
		cdhist = []; clhist = []; cmhist = []; ncell = []; cphist = []

		print("\nalpha: %r degree\n" %alphaset[ia])

		# run CFD on baseline mesh
		V = Mesh['V']; E = Mesh['E']; B = Mesh['B']
		Elements = init_Elements(1, V, E, alpha, '../cache/last'+str(alphaset[ia])+'.txt')
		Edges, Boundaries = init_Edges_Boundaries(V, E, B, Elements)
		Resd, cd, cl, cm, cp = runCFD(1, V, E, Elements, Edges, Boundaries, niter, alpha, CFL, tolerence, str(alphaset[ia]))	
		
		cdhist.append(cd); clhist.append(cl); cmhist.append(cm); ncell.append(len(Elements))
		#pltcontour(V, E, Elements, 'Baseline Mach contour a='+str(alphaset[ia]), 1)
		#pltmesh(V, E, 'Baseline Mesh a='+str(alphaset[ia]), 1)

		# Initial state
		uexist = np.loadtxt('../cache/steady'+str(alphaset[ia])+'.txt')
		#Elements = init_Elements(1, V, E, alpha, '../cache/steady'+str(alphaset[0])+'.txt')
		#Edges, Boundaries = init_Edges_Boundaries(V, E, B, Elements)
		
		for iteradap in range(nadapset[-1]):

				print("\nadaptation %d\n" %(iteradap+1))

				V, E, B, u = update_Elements(V, E, B, Elements, Edges, Boundaries, uexist)
				np.savetxt('../cache/before_adap'+str(alphaset[ia])+'_'+str(iteradap+1)+'.txt', u)
				Elements = init_Elements(1, V, E, alpha, '../cache/before_adap'+str(alphaset[ia])+'_'+str(iteradap+1)+'.txt')
				Edges, Boundaries = init_Edges_Boundaries(V, E, B, Elements)

				# check refined mesh is CCW
				for i in range(len(Elements)):
					elem = Elements[i]
					d1 = np.cross(elem.edge[0], elem.edge[1])
					if d1 < 0:
						print("Error: not all elements are CCW")
						return 0

				Rsed, cd, cl, cm, cp = runCFD(1, V, E, Elements, Edges, Boundaries, niter, alpha, CFL, tolerence,'_adap'+str(alphaset[ia])+'_'+str(iteradap+1))
				
				uexist = np.loadtxt('../cache/steady_adap'+str(alphaset[ia])+'_'+str(iteradap+1)+'.txt')

				for ifinset in range(len(nadapset)):
					if iteradap+1 == nadapset[ifinset]:
						cphist.append(cp)

				pltcontour(V, E, Elements, str(iteradap+1)+' adaptive Mach contour a='+str(alphaset[ia]), 1)
				pltmesh(V, E, str(iteradap+1)+' adaptation Mesh a='+str(alphaset[ia]), 1)

				cdhist.append(cd); clhist.append(cl); cmhist.append(cm); ncell.append(len(Elements))

		fd = plt.figure(figsize = (10,6))
		plt.plot(ncell, cdhist);
		plt.xlabel(r'$number of elements$', fontsize=16)
		plt.ylabel(r'$c_d$', fontsize=16)
		plt.title('Drag coefficient vs. number of elements plot a='+str(alphaset[ia]))
		plt.savefig('../figs/cd_converg_adap' +str(alphaset[ia])+ '.jpg')
		plt.show(block = False)
		plt.close(fd)

		fl = plt.figure(figsize = (10,6))
		plt.plot(ncell, clhist);
		plt.xlabel(r'$number of elements$', fontsize=16)
		plt.ylabel(r'$c_l$', fontsize=16)
		plt.title('Lift coefficient vs. number of elements plot a='+str(alphaset[ia]))
		plt.savefig('../figs/cl_converg_adap' +str(alphaset[ia])+ '.jpg')
		plt.show(block = False)
		plt.close(fd)

		fm = plt.figure(figsize = (10,6))
		plt.plot(ncell, cmhist);
		plt.xlabel(r'$number of elements$', fontsize=16)
		plt.ylabel(r'$c_m$', fontsize=16)
		plt.title('Moment coefficient vs. number of elements plot a=' +str(alphaset[ia]))
		plt.savefig('../figs/cm_converg_adap' +str(alphaset[ia])+ '.jpg')
		plt.show(block = False)
		plt.close(fd)

		fp = plt.figure(figsize = (12,6))
		for fpi in range(len(cphist)):
			plt.plot(cphist[fpi][:,0], cphist[fpi][:,1], label=str(nadapset[fpi])+' adaptation')
			np.savetxt('../cache/cp_adap'+str(alphaset[ia])+'_'+str(fpi)+'.txt', cphist[fpi])
		plt.xlabel(r'$theta [degree]$', fontsize=16)
		plt.ylabel(r'$c_p$', fontsize=16)
		plt.title('pressure coefficient line plot a='+str(alphaset[ia]))
		plt.legend()
		plt.savefig('../figs/cp_adap'+str(alphaset[ia])+  '.jpg')
		plt.show(block = False)
		plt.close(fp)

	
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

def test_proj1():

	# Read mesh
	V = np.loadtxt('../mesh/proj1/V.txt')
	E = np.loadtxt('../mesh/proj1/E.txt', dtype=int)
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

if __name__ == "__main__":
	main()