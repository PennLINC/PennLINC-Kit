import numpy as np
from scipy.stats import pearsonr
def matrix_corr(x,y):
	"""
	calculate the pearson r between a 1D array and a 2D array

	Parameters
	----------
	x : 2D array
	y : 1D array

	Returns
	-------
	out : 1D array, array[i] = pearsonr(x[i],y)

	"""
	xm = np.reshape(np.mean(x,axis=1),(x.shape[0],1))
	ym = np.mean(y)
	r_num = np.sum((x-xm)*(y-ym),axis=1)
	r_den = np.sqrt(np.sum((x-xm)**2,axis=1)*np.sum((y-ym)**2))
	r = r_num/r_den
	return r

def make_dnn_structure(neurons = 80,layers = 10):
	neurons_array = np.zeros((layers))
	neurons_array[:] = neurons
	neurons_array = tuple(neurons_array.astype(int))
	return neurons_array


def nan_pearsonr(x,y):
	x,y = np.array(x),np.array(y)
	mask = ~np.logical_or(np.isnan(x), np.isnan(y))
	return pearsonr(x[mask],y[mask])

def log_p_value(p):
	if p == 0.0:
		p = "-log10(p)>25"
	elif p > 0.001:
		p = np.around(p,3)
		p = "p=%s"%(p)
	else:
		p = (-1) * np.log10(p)
		p = "-log10()=%s"%(np.around(p,0).astype(int))
	return p

def convert_r_p(r,p):
	return "r=%s\n%s"%(np.around(r,3),log_p_value(p))

def matrix_triu(n_nodes=400):
	return np.triu_indices(n_nodes,1)

def matrix_tril(n_nodes=400):
	return np.tril_indices(n_nodes,-1)
