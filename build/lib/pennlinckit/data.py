import glob
import numpy as np
import os
from urllib.request import urlretrieve
import pkg_resources
import nibabel as nib
import pandas as pd
from multiprocessing import Pool
from functools import partial
from itertools import repeat
import pickle
import h5py
import scipy.io
from os.path import expanduser



class self:
	def __init__(self):
		pass


"""
#for testing
source = 'hcpd-dcan'
matrix_type = 'fc'
task = '**'
parcels = 'Schaefer417'
sub_cortex = False
session = '**'
cores = 1
fisher_z = True
"""

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
			self.subject_measures = pd.read_csv('/cbica/projects/hcpya/unrestricted_mb3152_10_26_2021_13_40_49.csv').rename(columns={'Subject':'subject'})
		elif self.source == 'hcpd-dcan':
			self.source_path = '/cbica/projects/hcpd/'
			self.subject_measures = pd.read_csv('{0}/data/hcpd_demographics.csv'.format(self.source_path),low_memory=False)
			self.subject_measures = self.subject_measures.rename(columns={'src_subject_id':'subject'})
			self.subject_measures.subject = self.subject_measures.subject.str.replace('HCD','') 
			self.subject_measures['meanFD'] = np.nan    
		else:
			self.source_path = '/cbica/projects/RBC/RBC_DERIVATIVES/{0}'.format(self.source.upper())
			self.subject_measures = pd.read_csv('/cbica/projects/RBC/RBC_DERIVATIVES/{0}/{1}_demographics.csv'.format(self.source.upper(),self.source))
			if self.source=='pnc':
				self.subject_measures = self.subject_measures.rename(columns={'reid':'subject'})
		self.cores = cores

	def update_subjects(self,subjects):
		selfsubject_measures = self.subject_measures[self.subject_measures.subject.isin(subjects)]

	def get_methods(self,modality='functional'):
		resource_package = 'pennlinckit'
		resource_path = '{0}_boiler_{1}.txt'.format(self.source,modality)
		return np.loadtxt(resource_path)


	def load_matrices(self, matrix_type, task='**', session = '**',parcels='Schaefer417',sub_cortex=False,fisher_z=True):
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
	    out : numpy matrix, fisher-z transformed, np.nan down the diagonal
	    ----------
		pnc examples
	    ----------
		dataset.get_matrix('nback')
		dataset.get_matrix('nback',parcels='gordon')
		dataset.get_matrix('diffusion_pr')
		"""

		self.matrix_type = matrix_type
		self.sub_cortex = sub_cortex
		self.parcels = parcels
		self.task = task
		self.session = session
		self.fisher_z = fisher_z
		if self.parcels == 'Schaefer417': n_parcels = 400
		if self.parcels == 'Schaefer217': n_parcels = 200
		if self.parcels == 'Gordon': n_parcels = 333
		if self.sub_cortex == True: n_parcels = n_parcels +50
		self.subject_measures['n_frames'] = np.nan

		self.matrix = []
		missing = []
		for subject in self.subject_measures.subject.values:
			if self.source == 'hcpya': 
				glob_matrices = glob.glob('/{0}/DERIVATIVES/XCP/sub-{1}/func/sub-{1}_task-{2}**atlas-{3}_den-91k_den-91k_bold.pconn.nii'.format(self.source_path,subject,self.task,self.parcels))
				glob_qc = glob.glob('/{0}/DERIVATIVES/XCP/sub-{1}/func/sub-{1}_task-{2}_acq-**_space-fsLR_desc-qc_den-91k_bold.csv'.format(self.source_path,subject,self.task))
			elif self.source == 'hcpd-dcan':
				glob_matrices = glob.glob('/{0}/data/sub-{1}/ses-V1/files/MNINonLinear/Results/task-{2}_DCANBOLDProc_v4.0.0_{3}.ptseries.nii'.format(self.source_path,subject,self.task,self.parcels))
				glob_qc = glob.glob('/{0}/data/motion/fd/sub-{1}/ses-V1/files/DCANBOLDProc_v4.0.0/analyses_v2/motion/task-{2}_power_2014_FD_only.mat'.format(self.source_path,subject,self.task))
				glob_scrub = glob.glob('/{0}/data/motion/fd/sub-{1}/ses-V1/files/DCANBOLDProc_v4.0.0/analyses_v2/motion/task-{2}_power_2014_FD_only.mat'.format(self.source_path,subject,self.task))
				glob_scrub.sort()
				glob_qc.sort()
				glob_matrices.sort()
			else: #RBC datasets
				glob_matrices = glob.glob('/{0}/XCP/sub-{1}/{2}/func/sub-{1}_{2}_task-{3}_space-fsLR_atlas-{4}_den-91k_den-91k_bold.pconn.nii'.format(self.source_path,subject,self.session,self.task,self.parcels))
				glob_qc = glob.glob('/{0}/XCP/sub-{1}/{2}/func/sub-{1}_{2}_task-{3}_space-fsLR_desc-qc_den-91k_bold.csv'.format(self.source_path,subject,self.session,self.task))	

			if len(glob_matrices)==0:
				print (subject)
				missing.append(subject)
				continue
			
			if self.source == 'hcpd-dcan':
				FD = []
				for qc in glob_qc:
					FD.append(scipy.io.loadmat(qc,squeeze_me=True,struct_as_record=False)['motion_data'][20].remaining_frame_mean_FD)
				self.subject_measures.loc[self.subject_measures.subject==subject,self.subject_measures.columns=='meanFD'] = np.mean(FD)		

				time_series = []
				idx = 0
				for ts,scrub_mask in zip(glob_matrices,glob_scrub):
					scrub_mask = scipy.io.loadmat(scrub_mask,squeeze_me=True,struct_as_record=False)['motion_data'][20].frame_removal
					ts = nib.load(ts).get_fdata()
					ts = ts[scrub_mask==0]
					if idx == 0: time_series = ts.copy()
					else: time_series = np.append(time_series,ts.copy(),axis=0)
					idx =+ 1
				self.subject_measures.loc[self.subject_measures.subject==subject,self.subject_measures.columns=='n_frames'] = time_series.shape[0]
				subject_matrix = np.corrcoef(time_series.swapaxes(0,1))
				self.matrix.append(subject_matrix)
				
			else:
				columns = pd.read_csv(glob_qc[0]).columns
				sub_df = pd.DataFrame(columns=columns)
				for qc in glob_qc:
					sub_df = sub_df.append(pd.read_csv(qc),ignore_index=True)
				sub_df = sub_df.groupby('sub').mean()
				sub_df['subject'] = sub_df.index
				if 'qc_df' in locals(): qc_df = qc_df.append(sub_df,ignore_index=True)
				else: qc_df = sub_df.copy()

				if len(glob_matrices)==1:
					subject_matrices = nib.load(glob_matrices[0]).get_fdata()
					np.fill_diagonal(subject_matrices,0.0)
					if self.fisher_z: subject_matrices = np.arctanh(subject_matrices)
				else:
					subject_matrices = []
					for matrix_path in glob_matrices:
						m = nib.load(matrix_path).get_fdata()
						np.fill_diagonal(m,0)
						if self.fisher_z: m = np.arctanh(m)
						subject_matrices.append(m)
					subject_matrices = np.nanmean(subject_matrices,axis=0)

				np.fill_diagonal(subject_matrices,np.nan)
				self.matrix.append(subject_matrices)

		if self.source != 'hcpd-dcan':self.subject_measures=self.subject_measures.merge(qc_df,on='subject')
		for missing_sub in missing:
			missing_sub = self.subject_measures.loc[self.subject_measures.subject == missing_sub]
			self.subject_measures = self.subject_measures.drop(missing_sub.index,axis=0)
		self.matrix = np.array(self.matrix)
		assert self.matrix.shape[0] == self.subject_measures.shape[0]


	def filter(self,way,value=None,column=None):
		"""
		way: operators like ==; "matrix" for subjects with a matrix;
			"has_subject_measure" for subjects with that value
		value: what the operator way is applied
		colums: the column the operator is applied to
		"""
		if way == '==':
			self.matrix = self.matrix[self.subject_measures[column]==value]
			self.subject_measures = self.subject_measures[self.subject_measures[column]==value]
		if way == '!=':
			self.matrix = self.matrix[self.subject_measures[column]!=value]
			self.subject_measures = self.subject_measures[self.subject_measures[column]!=value]
		if way == 'np.nan':
			self.matrix = self.matrix[np.isnan(self.subject_measures[column])==False]
			self.subject_measures = self.subject_measures[np.isnan(self.subject_measures[column])==False]
		if way == '>':
			self.matrix = self.matrix[self.subject_measures[column]>value]
			self.subject_measures = self.subject_measures[self.subject_measures[column]>value]
		if way == '<':
			self.matrix = self.matrix[self.subject_measures[column]<value]
			self.subject_measures = self.subject_measures[self.subject_measures[column]<value]
		if way == 'matrix':
			mask = np.isnan(self.matrix).sum(axis=1).sum(axis=1) == self.matrix.shape[-1]
			self.subject_measures = self.subject_measures[mask]
			self.matrix = self.matrix[mask]
		if way == 'has_subject_measure':
			mask = np.isnan(self.subject_measures[value]) == False
			self.subject_measures = self.subject_measures[mask]
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
