import numpy as np

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
