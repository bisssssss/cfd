import matplotlib.pyplot as plt
import numpy as np

def print_elem(elem):

	print("\nElement - %d\t" %(elem.idx))

	print("\tNode: \t%d %r %r" %(elem.node[0].idx, elem.node[0].x, elem.node[0].y))
	print("\t\t%d %r %r" %(elem.node[1].idx, elem.node[1].x, elem.node[1].y))
	print("\t\t%d %r %r" %(elem.node[2].idx, elem.node[2].x, elem.node[2].y))

	print("\tEdge: \t%r" %(elem.edge[0]))
	print("\t\t%r" %(elem.edge[1]))
	print("\t\t%r" %(elem.edge[2]))

	print("\tnVec: \t%r" %(elem.nvec[0]))
	print("\t\t%r" %(elem.nvec[1]))
	print("\t\t%r" %(elem.nvec[2]))

	print("\tEgLen: \t%r" %elem.el[0])
	print("\t\t%r" %(elem.el[1]))
	print("\t\t%r" %(elem.el[2]))

	print("\tNeigb: \t%r" %elem.neighbor[0])
	print("\t\t%r" %(elem.neighbor[1]))
	print("\t\t%r" %(elem.neighbor[2]))

	print("\tifbd: \t%r" %elem.ifbd[0])
	print("\t\t%r" %(elem.ifbd[1]))
	print("\t\t%r" %(elem.ifbd[2]))

	print("\tPerim: \t%r" %(elem.perimeter))
	print("\tarea: \t%r" %elem.area)
	print("\tM: \t%r" %elem.M)

	print("\tResd: \t%r" %elem.residual[0])
	print("\t\t%r" %(elem.residual[1]))
	print("\t\t%r" %(elem.residual[2]))
	print("\t\t%r" %(elem.residual[3]))

	print("\tu: \t%r" %elem.u[0])
	print("\t\t%r" %(elem.u[1]))
	print("\t\t%r" %(elem.u[2]))
	print("\t\t%r" %(elem.u[3]))

	print("\tFlux: \t%r\t%r" %(elem.flux[0][0][0], elem.flux[0][0][1]))
	print("\t\t%r\t%r" %(elem.flux[1][0][0], elem.flux[1][0][1]))
	print("\t\t%r\t%r" %(elem.flux[2][0][0], elem.flux[2][0][1]))
	print("\t\t%r\t%r" %(elem.flux[3][0][0], elem.flux[3][0][1]))

	print("")



def pltcontour(V, E, Elements, title, ifsave=0):

	nE = len(Elements)
	Mset = np.zeros((nE,1))
	for i in range(nE):
		Mset[i] = Elements[i].M

	V1 = []; V2 = []
	for i in range(len(V)):
		V1.append(V[i][0])
		V2.append(V[i][1])
	f = plt.figure(figsize = (56, 16))

	f.add_subplot(1, 2, 1)
	plt.axis('equal')
	tcf = plt.tripcolor(V1, V2, E, facecolors=Mset[:,0], cmap = 'jet')
	plt.xlabel(r'$x$', fontsize=16)
	plt.ylabel(r'$y$', fontsize=16)
	plt.xlim(-10, 10)
	plt.ylim(-18, 18)
	plt.title(title)
	plt.colorbar(tcf)

	f.add_subplot(1, 2, 2)
	plt.axis('equal')
	tcf = plt.tripcolor(V1, V2, E, facecolors=Mset[:,0], cmap = 'jet')
	plt.xlabel(r'$x$', fontsize=16)
	plt.ylabel(r'$y$', fontsize=16)
	plt.xlim(-1.5, 2.5)
	plt.ylim(-1.0, 1.0)
	plt.title(title)
	plt.colorbar(tcf)

	if ifsave == 1:
		plt.savefig('../figs/'+title + '.jpg')
	elif ifsave == 2:
		plt.pause(0.05)
		plt.close()
	elif ifsave == 0:
		plt.show()

	plt.close(f)



def pltmesh(V, E, title, ifsave=0):

	V1 = []; V2 = []
	for i in range(len(V)):
		V1.append(V[i][0])
		V2.append(V[i][1])
	f = plt.figure(figsize = (56, 16))

	f.add_subplot(1, 2, 1)
	plt.axis('equal')
	tcf = plt.triplot(V1, V2, E)
	plt.xlabel(r'$x$', fontsize=16)
	plt.ylabel(r'$y$', fontsize=16)
	plt.xlim(-10, 10)
	plt.ylim(-18, 18)
	plt.title(title)

	f.add_subplot(1, 2, 2)
	plt.axis('equal')
	tcf = plt.triplot(V1, V2, E)
	plt.xlabel(r'$x$', fontsize=16)
	plt.ylabel(r'$y$', fontsize=16)
	plt.xlim(-1.5, 2.5)
	plt.ylim(-1.0, 1.0)
	plt.title(title)

	if ifsave == 1:
		plt.savefig('../figs/'+title + '.jpg')
	elif ifsave == 2:
		plt.pause(0.05)
		plt.close()
	elif ifsave == 0:
		plt.show()

	plt.close(f)