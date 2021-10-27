import glob
import numpy as np
import os
from urllib.request import urlretrieve
import pkg_resources
import pandas as pd
from multiprocessing import Pool
from functools import partial
from itertools import repeat
import pickle
import h5py
from os.path import expanduser



class self:
	def __init__(self):
		pass


class dataset:
	"""
	This is the main object to use to load an rbc dataset
	source: str, the name of the dataset
	cores: int, the number of cores you will use for analysis
	"""
	def __init__(self, source='hcpya',cores=1):

		self.source = source
		if self.source == 'hcpya': 
			self.source_path = '/cbica/projects/hcpya/'
			self.subject_measures = pd.read_csv('/cbica/projects/hcpya/unrestricted_mb3152_10_26_2021_13_40_49.csv')
		self.cores = cores



	def update_subjects(self,subjects):
		self.measures = self.measures[self.measures.subject.isin(subjects)]

	def get_methods(self,modality='functional'):
		resource_package = 'pennlinckit'
		resource_path = '{0}_boiler_{1}.txt'.format(self.source,modality)
		return np.loadtxt(resource_path)


	def load_matrices(self, matrix_type, wildcard='ses-1', parcels='schaefer',sub_cortex=False):
		"""
		load matrix from this dataset
	    ----------
	    parameters
	    ----------
	    matrix_type: str, what type of matrix do you want? bold, diffusion (name the type)
		wildcard: str,
		parcels: str, schaefer, gordon, yeo,
		sub_cortex: bool, do you want this (https://github.com/yetianmed/subcortex) added on?
	    ----------
		returns
	    ----------
	    out : numpy matrix, fisher-z transformed before meaning(if done), np.nan down the diagonal
	    ----------
		pnc examples
	    ----------
		dataset.get_matrix('nback')
		dataset.get_matrix('nback',parcels='gordon')
		dataset.get_matrix('diffusion_pr')
		"""

		self.matrix_type = matrix_type

		self.parcels = parcels
		if self.parcels == 'schaefer': n_parcels = 400
		if self.parcels == 'gordon': n_parcels = 333

		if self.sub_cortex == True:
			n_parcels = n_parcels +50



		# qc = self.imaging_qc() #this will be the audit outputs
		# self.measures = self.measures.merge(qc,how='inner',on='subject')
		self.matrix = []
		missing = []
		for subject in self.measures.subject.values:
			if self.source == 'pnc':
				if self.matrix_type == 'rest':
					if self.parcels == 'gordon':
						matrix_path = '/{0}/neuroimaging/rest/restNetwork_gordon/GordonPNCNetworks/{1}_GordonPNC_network.txt'.format(self.data_path,subject)
					if self.parcels == 'schaefer':
						matrix_path = '/{0}//neuroimaging/rest/restNetwork_schaefer400/restNetwork_schaefer400/Schaefer400Networks/{1}_Schaefer400_network.npy'.format(self.data_path,subject)
			if self.source == 'hcp':
				matrix_path = '/{0}/matrices/{1}_{2}.npy'.format(self.data_path,subject,self.matrix_type)

			try:
				m = np.load(matrix_path)
				self.matrix.append(m)
			except:
				missing.append(subject)

		for missing_sub in missing:
			missing_sub = self.measures.loc[self.measures.subject == missing_sub]
			self.measures = self.measures.drop(missing_sub.index,axis=0)
		self.matrix = np.array(self.matrix)
		assert self.matrix.shape[0] == self.measures.shape[0]


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
		if way == 'cognitive':
			factors = ['F1_Exec_Comp_Res_Accuracy_RESIDUALIZED','F2_Social_Cog_Accuracy_RESIDUALIZED','F3_Memory_Accuracy_RESIDUALIZED']
			mask = np.isnan(self.measures[factors]).sum(axis=1) == 0
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
		self.expression[200:] = np.load(self.data_home + 'allen_expression_rh.npy')[200:]

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


# self = dataset('pnc')
# matrix_type = 'rest'
# parcels = 'schaefer'
# self.load_matrices('rest')
# self.filter('cognitive')
# self.filter('==',value=0,column='restRelMeanRMSMotionExclude')
