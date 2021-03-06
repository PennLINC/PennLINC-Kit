import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

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

def predict(self,model='ridge',cv='KFold',folds=5,layers=5,neurons=50,remove_linear_vars=False,remove_cat_vars=False):
	if cv == 'KFold':
		model_cv = KFold(folds)
	self.prediction = np.zeros((self.measures.subject.values.shape[0]))
	self.corrected_targets = self.targets.copy()
	for train, test in model_cv.split(self.measures.subject.values):
		x_train,y_train,x_test,y_test = self.features[train].copy(),self.targets[train].copy(),self.features[test].copy(),self.targets[test].copy()
		if type(remove_linear_vars) != bool:
			nuisance_model = LinearRegression()
			nuisance_model.fit(self.measures[remove_linear_vars].values[train],y_train)
			y_train = y_train - nuisance_model.predict(self.measures[remove_linear_vars].values[train])
			y_test = y_test - nuisance_model.predict(self.measures[remove_linear_vars].values[test])
		if type(remove_cat_vars) != bool:
			nuisance_model = LogisticRegression()
			nuisance_model.fit(self.measures[remove_linear_vars].values[train],y_train)
			y_train = y_train - nuisance_model.predict(self.measures[remove_cat_vars].values[train])
			y_test = y_test - nuisance_model.predict(self.measures[remove_cat_vars].values[test])
		if model == 'ridge':
			m = RidgeCV()
		if model == 'deep':
			m = MLPRegressor(hidden_layer_sizes=make_dnn_structure(neurons,layers))
		m.fit(x_train,y_train)
		self.prediction[test] = m.predict(x_test)
		self.corrected_targets[test] = y_test
                   
def remove(remove_me,y,data_type='linear'):
	if data_type == 'linear': model = LinearRegression()
	if data_type == 'categorical': model = LogisticRegression()
	y_model = model.fit(remove_me,y)
	y_predict = y_model.predict(remove_me) # predicted values
	y_residual =  y - y_predict # residual values
	return y_residual

