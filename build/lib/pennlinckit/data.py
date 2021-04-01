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

def get_matrix(self,subject):
	"""
	Child of dataset.load_matrices
	----------
    Parameters
    ----------
    subject: either a single subject, or ** for all subjects
	"""
	if self.parcels == 'schaefer': n_parcels = 400
	if self.parcels == 'gordon': n_parcels = 333

	if self.source == 'pnc':
		if self.matrix_type == 'rest':
			if self.parcels == 'gordon':
				matrix_path = '/{0}/neuroimaging/rest/restNetwork_gordon/GordonPNCNetworks/{1}_GordonPNC_network.txt'.format(self.data_path,subject)
			if self.parcels == 'schaefer':
				matrix_path = '/{0}//neuroimaging/rest/restNetwork_schaefer400/restNetwork_schaefer400/Schaefer400Networks/{1}_Schaefer400_network.txt'.format(self.data_path,subject)

	if self.source == 'hcp':
		matrix_path = '/{0}/matrices/{1}_{2}.npy'.format(self.data_path,subject,self.matrix_type)

	if self.source == 'pnc':
		try:
			m = np.loadtxt(matrix_path)
		except:
			m = np.zeros((n_parcels,n_parcels))
			m[:,:] = np.nan

	if self.source == 'hcp':
		try:
			m = np.load(matrix_path)
		except:
			m = np.zeros((n_parcels,n_parcels))
			m[:,:] = np.nan

	np.fill_diagonal(m,np.nan) #idiot proof
	return m

def load_dataset(name):
    return pickle.load(open("{0}".format(name), "rb"))

class dataset:
	"""
	This is the main object to use to load a dataset
	"""
	def __init__(self, source='pnc',cores=1):
		self.source = source
		self.cores = cores
		if self.source == 'pnc':
			self.data_path = '/gpfs/fs001/cbica/home/bertolem/pnc/'
			self.subject_column = 'scanid'
			self.measures = pd.read_csv('{0}/demographics/n1601_demographics_go1_20161212.csv'.format(self.data_path))
			self.subject_column = {'scanid':'subject'}
			self.measures = self.measures.rename(columns=self.subject_column)
			clinical = pd.read_csv('{0}/clinical/n1601_goassess_itemwise_bifactor_scores_20161219.csv'.format(self.data_path)).rename(columns=self.subject_column)
			self.measures = self.measures.merge(clinical,how='outer',on='subject')
			clinical_dict = pd.read_csv('{0}/clinical/goassess_clinical_factor_scores_dictionary.txt'.format(self.data_path),sep='\t')[24:29].drop(columns=['variablePossibleValues','source', 'notes'])
			self.data_dict = {}
			for k,i in zip(clinical_dict.variableName.values,clinical_dict.variableDefinition.values):
				self.data_dict[k.strip(' ')] = i.strip(' ')
			cognitive = pd.read_csv('{0}/cnb/n1601_cnb_factor_scores_tymoore_20151006.csv'.format(self.data_path)).rename(columns=self.subject_column)
			self.measures = self.measures.merge(cognitive,how='outer',on='subject')
			cognitive_dict = pd.read_csv('{0}/cnb/cnb_factor_scores_dictionary.txt'.format(self.data_path),sep='\t').drop(columns=['source'])
			for k,i in zip(cognitive_dict.variableName.values,cognitive_dict.variableDefinition.values):
				self.data_dict[k.strip(' ')] = i.strip(' ')
			cog_factors = pd.read_csv('{0}/cnb/cog_factors.csv'.format(self.data_path)).rename(columns=self.subject_column)
			self.measures = self.measures.merge(cog_factors,how='outer',on='subject')
		if self.source == 'hcp':
			self.data_path = '/home/mb3152/hcp/'
			self.subject_column = 'Subject'
			self.measures = pd.read_csv('{0}/unrestricted_mb3152_2_25_2021_8_59_45.csv'.format(self.data_path))
			self.subject_column = {'Subject':'subject'}
			self.measures = self.measures.rename(columns=self.subject_column)

	def update_subjects(self,subjects):
		self.measures = self.measures[self.measures.subject.isin(subjects)]

	def methods(self):
		resource_package = 'pennlinckit'
		resource_path = '{0}_boiler.txt'.format(self.source)
		path = pkg_resources.resource_stream(resource_package, resource_path)
		f = open(path.name, 'r').read()
		print (f)

	def asl(self):
		self.asl = 0

	def imaging_qc(self):
		if self.source == 'pnc':
			if self.matrix_type == 'rest':
				qc = pd.read_csv('{0}/neuroimaging/rest/n1601_RestQAData_20170714.csv'.format(self.data_path)).rename(columns=self.subject_column)
				qa_dict = pd.read_csv('{0}/neuroimaging/rest/restQADataDictionary_20161010.csv'.format(self.data_path))
				for k,i in zip(qa_dict.columnName.values,qa_dict.columnDescription.values):
					self.data_dict[k] = i
			if self.matrix_type == 'diffusion_pr':
				1/0
		return qc

	def load_matrices(self, matrix_type, parcels='schaefer'):
		"""
		get a matrix from this dataset
	    ----------
	    parameters
	    ----------
	    matrix_type: what type of matrix do you want? can be a task, resting-state, diffusion
		parcels: schaefer or gordon
	    ----------
		returns
	    ----------
	    out : mean numpy matrix, fisher-z transformed before meaning, np.nan down the diagonal
	    ----------
		pnc examples
	    ----------
		dataset.get_matrix('nback')
		dataset.get_matrix('nback',parcels='gordon')
		dataset.get_matrix('diffusion_pr')
		"""
		self.matrix_type = matrix_type
		self.parcels = parcels
		qc = self.imaging_qc()
		self.measures = self.measures.merge(qc,how='inner',on='subject')
		self.matrix = []
		for subject in self.measures.subject:
			self.matrix.append(get_matrix(self,subject))
		self.matrix = np.array(self.matrix)

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
	def save(self,name):
		pickle.dump(self, open("{0}".format(name), "wb"))  # save it into a file named save.p

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
