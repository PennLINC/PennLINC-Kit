import glob
import numpy as np
import os
from urllib.request import urlretrieve
import pkg_resources
import pandas as pd

def data_home():
	"""
	don't worry about it
	"""
	data_home = None
	if data_home is None:
	    data_home = os.environ.get('pennlinckit_data',
	                               os.path.join('~', 'pennlinckit_data'))
	data_home = os.path.expanduser(data_home)
	return data_home



class hcp:
	def __init__(self, cluster='PMACS'):
		self.cluster = cluster
		resource_package = 'pennlinckit'
		resource_path = 'unrestricted_mb3152_2_25_2021_8_59_45.csv'
		path = pkg_resources.resource_stream(resource_package, resource_path)
		self.measures = pd.read_csv(path.name)
		self.subjects = self.measures.Subject

	def get_hcp_matrix(self,subject):
		"""
		Child of hcp.load_matrices
		----------
	    Parameters
	    ----------
	    subject: either a single subject, or ** for all subjects
		"""
		if self.cluster == 'PMACS': root_dir = '/project/bbl_projects/hcp/CompCor_matrices_wsub_400'
		search_path = '/%s/yeo_400_**%s**_**%s**'%(root_dir,subject,self.task)
		matrix_paths = glob.glob(search_path)

		if len(matrix_paths) == 0:
			matrices = np.zeros((450,450))
			if self.subcortex == False:
				matrices = matrices[:400,:400]
			matrices[:,:] = np.nan
		else:
			matrices = []
			for m in matrix_paths:
				m = np.load(m)
				if self.subcortex == False:
					m = m[:400,:400]
				np.fill_diagonal(m,0.0) #so we don't get archtanh error
				matrices.append(np.arctanh(m))
			matrices = np.array(matrices)
		if len(matrices.shape) == 3:
			matrices = np.nanmean(matrices,axis=0)
		np.fill_diagonal(matrices,np.nan) #idiot proof
		return matrices


	def load_matrices(self, subjects, task='REST', subcortex=False):
		"""
		Get a matrix from the Human Connectome Project
		These were processed with CompCor (5 Components), GSR, Bandpass, and motion scrubbing.
		Made using: https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii
		Subcortex is from node 401-450: https://github.com/yetianmed/subcortex
		Please each out to max bertolero (mbertolero@me.com) if you publish using this data.
	    ----------
	    Parameters
	    ----------
	    subject: either a single subject, or ** for all subjects
	    task: either a single task, or ** for all task. REST, LANGUAGE, MATH, SOCIAL, GAMBLING, MOTOR, RELATIONAL, EMOTION
	    ----------
		Returns
	    ----------
	    out : mean numpy matrix, fisher-z transformed before meaning, np.nan down the diagonal
		drops subjects you did not call from the hcp.measures object
	    ----------
		Examples
	    ----------
		all_subjects_all_tasks = get_hcp_matrix('**','**')
		GAMBLING_126628 = get_hcp_matrix(126628,'GAMBLING')
		all_subjects_rest = get_hcp_matrix('**','REST')
		"""
		self.subjects = subjects
		self.task = task
		self.subcortex = subcortex
		if type(self.subjects) == str or type(self.subjects) == int or type(self.subjects) == float:
			self.matrix = self.get_hcp_matrix(self.subjects)
		else:
			self.matrix = []
			for subject in self.subjects:
				m = self.get_hcp_matrix(subject)
				self.matrix.append(m)
			self.matrix = np.array(self.matrix)
		self.measures = self.measures[self.measures.Subject.isin([self.subjects])]
	def filter(self,way,value=None,column=None):
		if way == 'identity':
			self.matrix = self.matrix[self.measures[column]==value]
			self.measures = self.measures[self.measures[column]==value]
		if way == '>':
			self.matrix = self.matrix[self.measures[column]>value]
			self.measures = self.measures[self.measures[column]>value]
		if way == '<':
			self.matrix = self.matrix[self.measures[column]<value]
			self.measures = self.measures[self.measures[column]<value]
		if way == 'matrix':
			mask = np.isnan(self.matrix).sum(axis=1).sum(axis=1) == self.matrix.shape[-1]
			self.measures = self.measures[mask]
			self.matrix = self.matrix[mask]



#
# hcp_data = hcp(cluster='PMACS')
# hcp_data.measures.head(5)
# hcp_data.load_matrices(subjects=hcp_data.measures.Subject[:25],task='WM',subcortex=False)
# hcp_data.matrix.shape
# hcp_data.measures.shape
# hcp_data.filter('matrix')
# hcp_data.matrix.shape
# hcp_data.measures.shape
# hcp_data.filter('identity',value='S900',column='Release')
# hcp_data.matrix.shape
# hcp_data.measures.shape

def all_brain_institute():
	"""
	get the allen brain institute gene expression for a given hemisphere
    ----------
	Returns
    ----------
    out : left hemisphere expression, right hemisphere expression, names of the genes
	"""
	path = os.path.join(data_home(), 'allen/')
	path
	if os.path.exists(path) == False:
		os.makedirs(path)
		urlretrieve('https://www.dropbox.com/s/1zahnd0k0jpk0xf/allen_expression_genes.npy?dl=1',path + 'allen_expression_genes.npy')
		urlretrieve('https://www.dropbox.com/s/879cuel80nntipq/allen_expression_lh.npy?dl=1',path + 'allen_expression_lh.npy')
		urlretrieve('https://www.dropbox.com/s/cnb6aacerdhhd4p/allen_expression_rh.npy?dl=1',path + 'allen_expression_rh.npy')


	names = np.load(path + 'allen_expression_genes.npy',allow_pickle=True)
	lh = np.load(path + 'allen_expression_lh.npy')
	rh = np.load(path + 'allen_expression_rh.npy')
	return lh,rh,names
