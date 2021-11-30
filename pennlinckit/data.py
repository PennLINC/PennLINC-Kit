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
			if self.source=='pnc': self.subject_measures = self.subject_measures.rename(columns={'reid':'subject'})
		self.cores = cores

	def get_methods(self,modality='functional'):
		"""
		this won't work yet, but will load the methods text for papers
		"""
		resource_package = 'pennlinckit'
		resource_path = '{0}_boiler_{1}.txt'.format(self.source,modality)
		return np.loadtxt(resource_path)

	def load_matrices(self, task='**', session = '**',parcels='Schaefer417',sub_cortex=False,fd_scrub=False):
		"""
		load matrices from this dataset
		----------
		parameters
		----------
		task: str, the name of the task or ** to grab all tasks
		session: str, the name of the session, or ** to grab all sessions
		parcels: str, Schaefer417, Gordon, Glasser,
		sub_cortex: bool, if you want this (https://github.com/yetianmed/subcortex) sub_cortical data added on
		fd_scrub: bool or float, False for no scrubbing, or float to remove frames with FD above fd_scrub
		----------
		returns
		----------
		out : numpy matrix, np.nan down the diagonal
		note! this will also add the "n_frames" column to subject_measures, listing the number of frames remaining post scrubbing
		----------
		example (from PNC)
		----------
		dataset.get_matrix('nback', session = '**',parcels='Schaefer417',sub_cortex=True,fd_scrub=0.2)
		"""

		self.sub_cortex = sub_cortex
		self.parcels = parcels
		self.task = task
		self.session = session
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
				glob_scrub = glob.glob('/{0}/DERIVATIVES/XCP/sub-{1}/{2}/func/sub-{1}_{2}_task-{3}_**-framewisedisplacement_res-2_bold.tsv'.format(self.source_path,subject,self.session,self.task))
			elif self.source == 'hcpd-dcan':
				glob_matrices = glob.glob('/{0}/data/sub-{1}/ses-V1/files/MNINonLinear/Results/task-{2}_DCANBOLDProc_v4.0.0_{3}.ptseries.nii'.format(self.source_path,subject,self.task,self.parcels))
				glob_qc = glob.glob('/{0}/data/motion/fd/sub-{1}/ses-V1/files/DCANBOLDProc_v4.0.0/analyses_v2/motion/task-{2}_power_2014_FD_only.mat'.format(self.source_path,subject,self.task))
				glob_scrub = glob.glob('/{0}/data/motion/fd/sub-{1}/ses-V1/files/DCANBOLDProc_v4.0.0/analyses_v2/motion/task-{2}_power_2014_FD_only.mat'.format(self.source_path,subject,self.task))
			else: #RBC datasets
				glob_matrices = glob.glob('/{0}/XCP/sub-{1}/{2}/func/sub-{1}_{2}_task-{3}_space-fsLR_atlas-{4}_den-91k_den-91k_bold.ptseries.nii'.format(self.source_path,subject,self.session,self.task,self.parcels))
				glob_qc = glob.glob('/{0}/XCP/sub-{1}/{2}/func/sub-{1}_{2}_task-{3}_space-fsLR_desc-qc_den-91k_bold.csv'.format(self.source_path,subject,self.session,self.task))	
				glob_scrub = glob.glob('/{0}/XCP/sub-{1}/{2}/func/sub-{1}_{2}_task-{3}_**-framewisedisplacement_res-2_bold.tsv'.format(self.source_path,subject,self.session,self.task))
			if len(glob_matrices)==0:
				print (subject)
				missing.append(subject)
				continue
			glob_scrub.sort()
			glob_qc.sort()
			glob_matrices.sort()
			
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

				for ts_file,scrub_mask in zip(glob_matrices,glob_scrub):
					scrub_mask = pd.read_csv(scrub_mask,header=None).values.reshape(1,-1).flatten()
					ts = nib.load(ts_file).get_fdata()
					if type(fd_scrub) == float: fd_scrubts = ts[scrub_mask<=fd_scrub]
					if 'time_series' in locals(): time_series = np.append(time_series,ts.copy(),axis=0)
					else: time_series = ts.copy()
					if sub_cortex:
						sub_ts = ts_file.replace(parcels,'subcortical')
						sub_ts = nib.load(sub_ts).get_fdata()
						if type(fd_scrub) == float: sub_ts = sub_ts[scrub_mask<=fd_scrub]
						time_series = np.append(time_series,sub_ts.copy(),axis=1)
				self.subject_measures.loc[self.subject_measures.subject==subject,self.subject_measures.columns=='n_frames'] = time_series.shape[0]
				subject_matrix = np.corrcoef(time_series.swapaxes(0,1))
				self.matrix.append(subject_matrix)

		if self.source != 'hcpd-dcan':self.subject_measures=self.subject_measures.merge(qc_df,on='subject')
		for missing_sub in missing:
			missing_sub = self.subject_measures.loc[self.subject_measures.subject == missing_sub]
			self.subject_measures = self.subject_measures.drop(missing_sub.index,axis=0)
		self.matrix = np.array(self.matrix)
		assert self.matrix.shape[0] == self.subject_measures.shape[0]


	def filter(self,way,value=None,column=None):
		"""
		way: 
			operators like ==;
			OR
			"matrix" for subjects with a matrix;
			OR
			"has_subject_measure" for subjects with that column in subject_measures
		value: what the operator way is applied
		colums: the column the operator is applied to

		returns
		---------
		this edits the data.matrix and data.subject_measures according to the filter inputs, inplace 
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
		if way == '>=':
			self.matrix = self.matrix[self.subject_measures[column]>=value]
			self.subject_measures = self.subject_measures[self.subject_measures[column]>value]
		if way == '<=':
			self.matrix = self.matrix[self.subject_measures[column]<=value]
			self.subject_measures = self.subject_measures[self.subject_measures[column]<value]
		if way == 'matrix':
			mask = np.isnan(self.matrix).sum(axis=1).sum(axis=1) == self.matrix.shape[-1]
			self.subject_measures = self.subject_measures[mask]
			self.matrix = self.matrix[mask]
		if way == 'has_subject_measure':
			mask = np.isnan(self.subject_measures[value]) == False
			self.subject_measures = self.subject_measures[mask]
			self.matrix = self.matrix[mask]


"""
#for testing

class self:
	def __init__(self):
		pass

source = 'pnc'
matrix_type = 'fc'
task = '**'
parcels = 'Schaefer417'
sub_cortex = False
session = '**'
cores = 1
fisher_z = True
"""
