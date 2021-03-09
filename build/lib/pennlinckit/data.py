import glob
import numpy as np
import os
from urllib.request import urlretrieve
import pkg_resources
import pandas as pd



class dataset:
	def __init__(self, source='pnc', cluster='PMACS'):
		self.cluster = cluster
		self.source = source
		resource_package = 'pennlinckit'
		if self.source == 'pnc': resource_path = 'pnc_demo.csv'
		if self.source == 'hcp': resource_path = 'unrestricted_mb3152_2_25_2021_8_59_45.csv'
		path = pkg_resources.resource_stream(resource_package, resource_path)
		self.measures = pd.read_csv(path.name)
		if self.source == 'pnc': self.subject_column = 'scanid'
		if self.source == 'hcp': self.subject_column= 'Subject'
		self.subjects = self.measures[self.subject_column]
	def methods(self):
		resource_package = 'pennlinckit'
		resource_path = '%s_boiler.txt' %(self.source)
		path = pkg_resources.resource_stream(resource_package, resource_path)
		f = open(path.name, 'r').read()
		print (f)

	def get_matrix(self,subject):
		"""
		Child of dataset.load_matrices
		----------
	    Parameters
	    ----------
	    subject: either a single subject, or ** for all subjects
		"""
		if self.cluster == 'PMACS':
			if self.source == 'hcp':
				root_dir = '/home/mb3152/hcp/'
				search_path = '/%s/yeo_400_**%s**_**%s**'%(root_dir,subject,self.task)
			if self.source == 'pnc':
				root_dir = '/home/mb3152/pnc/'
				search_path = '%s/%s_Schaefer400_network.txt'%(root_dir,subject)
		matrix_paths = glob.glob(search_path)

		if len(matrix_paths) == 0:
			matrices = np.zeros((416,416))
			if self.subcortex == False:
				matrices = matrices[:400,:400]
			matrices[:,:] = np.nan
		else:
			matrices = []
			for m in matrix_paths:
				try: m = np.load(m)
				except: m = np.loadtxt(m) 
				if self.subcortex == False:
					m = m[:400,:400]
				np.fill_diagonal(m,0.0) #so we don't get archtanh error
				matrices.append(np.arctanh(m))
			matrices = np.array(matrices)
		if len(matrices.shape) == 3:
			matrices = np.nanmean(matrices,axis=0)
		np.fill_diagonal(matrices,np.nan) #idiot proof
		return matrices


	def load_matrices(self, task='REST', subcortex=False):
		"""
		Get a matrix from the dataset
	    ----------
	    Parameters
	    ----------
	    subject: either a single subject, ** for all subjects, or a list of subjects
	    task, str: either a single task, or ** for all task.
		HCP: REST, LANGUAGE, MATH, SOCIAL, GAMBLING, MOTOR, RELATIONAL, EMOTION
	    ----------
		Returns
	    ----------
	    out : mean numpy matrix, fisher-z transformed before meaning, np.nan down the diagonal
		drops subjects you did not call from the hcp.measures object
	    ----------
		HCP Examples
	    ----------
		all_subjects_all_tasks = get_hcp_matrix('**','**')
		GAMBLING_126628 = get_hcp_matrix(126628,'GAMBLING')
		all_subjects_rest = get_hcp_matrix('**','REST')
		"""
		self.task = task
		self.subcortex = subcortex
		if type(self.subjects) == str or type(self.subjects) == int or type(self.subjects) == float:
			self.matrix = self.get_matrix(self.subjects)
			self.measures[self.measures.Subject.isin([self.subjects])]
		elif type(self.subjects) == list or type(self.subjects) == np.ndarray  or type(self.subjects) == pd.core.series.Series:
			self.matrix = []
			for subject in self.subjects:
				m = self.get_matrix(subject)
				self.matrix.append(m)
			self.matrix = np.array(self.matrix)
			self.measures = self.measures[self.measures[self.subject_column].isin(self.subjects)]
		else: print ('what are you trying to do, you passed a weird subject list')
	def filter(self,way,value=None,column=None):
		if way == '==':
			self.matrix = self.matrix[self.measures[column]==value]
			self.measures = self.measures[self.measures[column]==value]
		if way == '!=':
			self.matrix = self.matrix[self.measures[column]!=value]
			self.measures = self.measures[self.measures[column]!=value]
		if way == 'np.nan':
			self.matrix = self.matrix[np.isnan(self.measures[column])==False]
			self.measures = self.measures[np.isnan(self.measures[column])==False]
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

class allen_brain_institute:
	def __init__(self):
		"""
		Allen Gene Expression data in the Scheafer 400 parcels.
	    ----------
		Returns
	    ----------
	    out : left hemisphere expression, right hemisphere expression, names of the genes
		"""
		self.data_home = os.path.expanduser('~/allen/')
		self.data_home = os.path.join(self.data_home)
		if os.path.exists(self.data_home) == False:
			print ("Gemme a sec, I am downloading allen gene expression to: %s" %(self.data_home))
			os.makedirs(self.data_home)
			urlretrieve('https://www.dropbox.com/s/1zahnd0k0jpk0xf/allen_expression_genes.npy?dl=1',self.data_home + 'allen_expression_genes.npy')
			urlretrieve('https://www.dropbox.com/s/879cuel80nntipq/allen_expression_lh.npy?dl=1',self.data_home + 'allen_expression_lh.npy')
			urlretrieve('https://www.dropbox.com/s/cnb6aacerdhhd4p/allen_expression_rh.npy?dl=1',self.data_home + 'allen_expression_rh.npy')
			names = np.load(self.data_home + 'allen_expression_genes.npy',allow_pickle=True)
			final_names = []
			for n in names:final_names.append(n[0][0])
			np.save(self.data_home + 'allen_expression_genes.npy',final_names)
			print ("Okay, done, I won't have to do this again!")

		self.names = np.load(self.data_home + 'allen_expression_genes.npy')
		self.expression= np.zeros((400,len(self.names)))
		self.expression[:200] = np.load(self.data_home + 'allen_expression_lh.npy')[:200]
		self.expression[200:400] = np.load(self.data_home + 'allen_expression_lh.npy')[200:]

class evo_expansion:
	def __init__(self):
		resource_package = 'pennlinckit'
		resource_path = 'schaefer400x17_rescaled_regional_evoExpansion_LHfixed.npy'
		path = pkg_resources.resource_stream(resource_package, resource_path)
		self.data = np.load(path.name)


class gradient:
	def __init__(self):
		resource_package = 'pennlinckit'
		resource_path = 'schaefer400x17_mean_regional_margulies_gradient.txt'
		path = pkg_resources.resource_stream(resource_package, resource_path)
		self.data = np.loadtxt(path.name)
