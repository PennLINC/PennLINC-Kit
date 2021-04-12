import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import subprocess
import os
import pickle
import copy

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

def bootstrap_t_test(x,y,t_test,perms=10000):
	"""
	permutation test any given t_test

	x,y: numpy array, group data you are comparing
	t_test: scipy.stats object (e.g., scipy.stats.ttest_rel)
	perms: int, number of perms

	returns:
	t, original test value
	l,h, 95% confidence intervals
	p, bootstrap based p-value
	"""
	x_mean = x.mean()
	y_mean = y.mean()
	xy_mean = np.mean(np.array([x,y]))
	t, p = t_test(x,y)
	x_test = x - x_mean + xy_mean
	y_test = y - y_mean + xy_mean
	perm_shape = x.shape[0]
	stat = np.zeros(perms)
	for p in range(perms):
		this_perm = np.random.choice(range(perm_shape),perm_shape,replace=True)
		stat[p] =  t_test(x_test[this_perm],y_test[this_perm])[0]
	l,h =  np.percentile(stat,2.5),np.percentile(stat,97.5)
	if t > 0: return (t,l,h,len(stat[stat<0]) / (len(stat))*2)
	else:return(t,l,h,len(stat[stat>0]) / (len(stat))*2)

def bootstrap_corr(x,y,corr,perms=10000):
	"""
	permutation test any given correlation

	x,y: numpy array, group data you are comparing
	t_test: scipy.stats object (e.g., scipy.stats.pearsonr)
	perms: int, number of perms

	returns:
	t, original corr r value
	l,h, 95% confidence intervals
	p, bootstrap based p-value
	"""
	perm_shape = x.shape[0]
	r = corr(x,y)[0]
	stat = np.zeros(perms)
	for p in range(perms):
		this_perm = np.random.choice(range(perm_shape),perm_shape,replace=True)
		stat[p] =  corr(x[this_perm],y[this_perm])[0]
	l,h =  np.percentile(stat,2.5),np.percentile(stat,97.5)
	if r > 0:
		return (r,l,h,len(stat[stat<0]) / (len(stat))*2)
	else:
		return(r,l,h,len(stat[stat>0]) / (len(stat))*2)

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
			nuisance_model.fit(self.measures[remove_linear_vars].values[train],y_train) #fit the nuisance_model to training data
			y_train = y_train - nuisance_model.predict(self.measures[remove_linear_vars].values[train]) #remove nuisance from training data
			y_test = y_test - nuisance_model.predict(self.measures[remove_linear_vars].values[test]) #remove nuisance from test data
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

def submit_array_job(scipt_path,array_start,array_end,RAM=4,threads=1):
	"""
	submit and "array job", where you get a job with an ID from array_start to array_end
	this lets you submit a script and iterate over an array in it.
	in the script, use the function utils.get_sge_task_id to get the ID
	"""
	sgedir = os.path.expanduser('~/sge/')
	if os.path.isdir(sgedir) == False:
		os.system('mkdir {0}'.format(sgedir))
	command='qsub -t {0}-{1} -l h_vmem={2}G,s_vmem={2}G -pe threaded {3}\
	 -N data -V -j y -b y -o ~/sge/ -e ~/sge/ python {4}'.format(array_start,array_end,RAM,threads,scipt_path)
	os.system(command)

def submit_job(scipt_path,name,RAM=4,threads=1):
	"""
	submit an sge job
	"""
	sgedir = os.path.expanduser('~/sge/')
	if os.path.isdir(sgedir) == False:
		os.system('mkdir {0}'.format(sgedir))
	command='qsub -l h_vmem={0}G,s_vmem={0}G -pe threaded {1}\
	 -N {2} -V -j y -b y -o ~/sge/ -e ~/sge/ python {3}'.format(RAM,threads,name,scipt_path)
	os.system(command)

def get_sge_task_id():
	"""
	This function will return the sge_task_id that gets set by using the submit_array_job command
	SGE starts by default at 1, but this will start at 0, because we are in python
	So you should have an array in your script you submitted that gets indexed by this value
	subjects = [one,two,three]
	subject = subjects[get_sge_task_id()]
	"""
	sge_task_id = subprocess.check_output(['echo $SGE_TASK_ID'],shell=True).decode()
	return int(sge_task_id) -1

def load_dataset(name):
    return pickle.load(open("{0}".format(name), "rb"))

def save_dataset(dataset,name):
	pickle.dump(dataset, open("{0}".format(name), "wb"))


def cut_data(data,min_cut=1.5,max_cut=1.5):
	"""
	remove outlier so your colorscale is not driven by one or two large values

    Parameters
    ----------
    data: the data you want to cut
    min_cut: std cutoff for low values
    max_cut: std cutoff for high values

    Returns
    -------
    out : cut data
	"""
	d = data.copy()
	max_v = np.mean(d) + np.std(d)*max_cut
	min_v = np.mean(d) - np.std(d)*min_cut
	d[d>max_v] = max_v
	d[d<min_v] = min_v
	return d

def make_heatmap(data,cmap='stock'):
	"""
	Generate an RGB value for each value in "data"

    Parameters
    ----------
    data: the data you want to colormap
    cmap: nicegreen, nicepurp, stock, Reds, or send your own seaborn color_palette / cubehelix_palette object
    Returns
    -------
    out : RGB values
	"""
	if cmap == 'nicegreen': orig_colors = sns.cubehelix_palette(1001, rot=-.5, dark=.3)
	elif cmap == 'nicepurp': orig_colors = sns.cubehelix_palette(1001, rot=.5, dark=.3)
	elif cmap == 'stock': orig_colors = sns.color_palette("RdBu_r",n_colors=1001)
	elif cmap == 'Reds': orig_colors = sns.color_palette("Reds",n_colors=1001)
	else: orig_colors = cmap
	norm_data = copy.copy(data)
	if np.nanmin(data) < 0.0: norm_data = norm_data + (np.nanmin(norm_data)*-1)
	elif np.nanmin(data) > 0.0: norm_data = norm_data - (np.nanmin(norm_data))
	norm_data = norm_data / float(np.nanmax(norm_data))
	norm_data = norm_data * 1000
	norm_data = norm_data.astype(int)
	colors = []
	for d in norm_data:
		colors.append(orig_colors[d])
	return colors
