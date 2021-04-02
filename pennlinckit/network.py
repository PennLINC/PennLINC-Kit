import numpy as np
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
from scipy.sparse.csgraph import minimum_spanning_tree
import pennlinckit.brain
from multiprocessing import Pool
from functools import partial
from itertools import repeat

def part_coef(W, ci, degree='undirected'):
	'''
	Participation coefficient is a measure of diversity of intermodular
	connections of individual nodes.
	Parameters
	----------
	W : NxN np.ndarray
		binary/weighted directed/undirected connection matrix
	ci : Nx1 np.ndarray
		community affiliation vector
	degree : str
		Flag to describe nature of graph 'undirected': For undirected graphs
										 'in': Uses the in-degree
										 'out': Uses the out-degree
	Returns
	-------
	P : Nx1 np.ndarray
		participation coefficient
	'''
	if degree == 'in':
		W = W.T

	_, ci = np.unique(ci, return_inverse=True)
	ci += 1

	n = len(W)  # number of vertices
	Ko = np.sum(W, axis=1)  # (out) degree
	Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
	Kc2 = np.zeros((n,))  # community-specific neighbors

	for i in range(1, int(np.max(ci)) + 1):
		Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

	P = np.ones((n,)) - Kc2 / np.square(Ko)
	# P=0 if for nodes with no (out) neighbors
	P[np.where(np.logical_not(Ko))] = 0

	return P

def matrix_to_igraph(matrix,cost=0.01,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False,test_matrix=True):
	"""
	Convert a matrix to an igraph object
	Parameters
	----------
	matrix: a numpy square matrix
	cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent of all possible edges in the graph
	binary: False, convert weighted values to 1
	check_tri: True, ensure that the matrix contains upper and low triangles. if it does not, the cost calculation changes.
	interpolation: midpoint, the interpolation method to pass to np.percentile
	normalize: False, make all edges sum to 1. Convienient for comparisons across subjects, as this ensures the same sum of weights and number of edges are equal across subjects
	mst: False, calculate the maximum spanning tree, which is the strongest set of edges that keep the graph connected. This is convienient for ensuring no nodes become disconnected.

	Returns
	-------
	out : igraph graph object
	"""
	matrix = np.array(matrix)
	assert matrix.shape[0] == matrix.shape[1]
	matrix = threshold(matrix,cost,binary,check_tri,interpolation,normalize,mst)
	g = Graph.Weighted_Adjacency(matrix.tolist(),mode=ADJ_UNDIRECTED,attr="weight")
	# print ('Matrix converted to graph with density of: ' + str(g.density()))
	if abs(np.diff([cost,g.density()])[0]) > .005:
		print ('Density not %s! Did you want: ' %(cost)+ str(g.density()) + ' ?')
	return g

def threshold(matrix,cost=0.01,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False,test_matrix=True):
	"""
	Threshold a numpy matrix to obtain a certain "cost".
	Parameters
	----------
	matrix: a numpy matrix
	cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent of all possible edges in the graph
	binary: False, convert weighted values to 1
	check_tri: True, ensure that the matrix contains upper and low triangles. if it does not, the cost calculation changes.
	interpolation: midpoint, the interpolation method to pass to np.percentile
	normalize: False, make all edges sum to 1. Convienient for comparisons across subjects, as this ensures the same sum of weights and number of edges are equal across subjects
	mst: False, calculate the maximum spanning tree, which is the strongest set of edges that keep the graph connected. This is convienient for ensuring no nodes become disconnected.

	Returns
	-------
	out : thresholded matrix (NOT A COPY)

	"""
	matrix[np.isnan(matrix)] = 0.0
	matrix[matrix<0.0] = 0.0
	np.fill_diagonal(matrix,0.0)
	c_cost_int = 100-(cost*100)
	if check_tri == True:
		if np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0:
			c_cost_int = 100.-((cost/2.)*100.)
	if c_cost_int > 0:
		if mst == False:
			matrix[matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)] = 0.
		else:
			if test_matrix == True: t_m = matrix.copy()
			assert (np.tril(matrix,-1) == np.triu(matrix,1).transpose()).all()
			matrix = np.tril(matrix,-1)
			mst = minimum_spanning_tree(matrix*-1)*-1
			mst = mst.toarray()
			mst = mst.transpose() + mst
			matrix = matrix.transpose() + matrix
			if test_matrix == True: assert (matrix == t_m).all() == True
			matrix[(matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)) & (mst==0.0)] = 0.
	if binary == True:
		matrix[matrix>0] = 1
	if normalize == True:
		matrix = matrix/np.sum(matrix)
	return matrix

def metrics(self,m):
	graphs = []
	q = np.zeros((len(self.costs)))
	pc = np.zeros((len(self.costs),m.shape[0]))
	strength = np.zeros((len(self.costs),m.shape[0]))
	for idx,cost in enumerate(self.costs):
		graph = matrix_to_igraph(m.copy(), cost, binary=self.binary, normalize=self.normalize, mst=self.mst)
		if self.yeo_partition:
			vc = VertexClustering(graph,membership=self.membership ,modularity_params={'weights':'weight'})
			pc[idx] = part_coef(np.array(graph.get_adjacency(attribute='weight').data),self.membership)
		else:
			vc = graph.community_infomap(edge_weights='weight')
			pc[idx] = part_coef(np.array(graph.get_adjacency(attribute='weight').data),vc.membership)
		q[idx] = vc.modularity
		strength[idx] = vc.graph.strength(weights='weight')
		graphs.append(vc)
	return graphs,q,pc,strength

class make_networks:
	def __init__(self,dataset,costs=[0.15,0.1,0.05,0.025,0.01],yeo_partition=17,binary=False,sym=True,normalize=False,mst=True,cores=4):
		self.costs = costs
		self.yeo_partition = yeo_partition
		self.binary = binary
		self.sym = sym
		self.normalize = normalize
		self.mst = mst
		if self.sym == True:
			for m in range(dataset.matrix.shape[0]):
				dataset.matrix[m] = dataset.matrix[m]+ dataset.matrix[m].transpose()
				dataset.matrix[m] = np.tril(dataset.matrix[m],-1)
				dataset.matrix[m] = dataset.matrix[m] + dataset.matrix[m].transpose()
				dataset.matrix[m] = dataset.matrix[m] / 2.
		if self.yeo_partition != False:
			self.membership = pennlinckit.brain.yeo_partition(int(self.yeo_partition))[1]
		self.graphs = []
		self.pc = []
		self.strength = []
		self.modularity = []
		pool = Pool(cores)
		for r in pool.starmap(metrics, zip(repeat(self),dataset.matrix)):
			# return graphs,q,pc,strength
			self.graphs.append(r[0])
			self.modularity.append(r[1])
			self.pc.append(r[2])
			self.strength.append(r[3])

		self.pc = np.array(self.pc)
		self.strength = np.array(self.strength)
		self.modularity = np.array(self.modularity)
