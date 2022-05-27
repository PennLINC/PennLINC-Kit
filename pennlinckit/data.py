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


def load(self,loadtype,subjects):
	"""
	helper function called by load_ptseries, load_matrix, load_cifti
	"""
	if loadtype=='ptseries' or loadtype == 'matrix': 
		name = 'ptseries'
	else:
		name = 'dtseries'

	missing = []
	for subject in subjects:
		"""
		there are a few options here, mostly as we used datasets that were not part of RBC.
		sooner or later, you can probably remove the hcpd-dcan (after we do hcp-d in RBC).
		you could always chage the path structure for hcpya to match RBC, but probably not worth the trouble.
		"""
		if self.source == 'hcpya':
			if loadtype == 'cifti': glob_matrices = glob.glob('{0}/DERIVATIVES/XCP/sub-{1}/func/sub-{1}_task-{3}_acq-{4}_space-fsLR_den-91k_desc-residual_den-91k_bold.dtseries.nii'.format(self.source_path,subject,self.task,self.session,self.acq))
			else: glob_matrices = glob.glob('{0}/DERIVATIVES/XCP/sub-{1}/func/sub-{1}_task-{3}_acq-{4}_space-fsLR_atlas-{5}_den-91k_den-91k_bold.{6}.nii'.format(self.source_path,subject,self.task,self.session,self.acq,self.parcels,name))
			glob_qc = glob.glob('{0}/DERIVATIVES/XCP/sub-{1}/func/sub-{1}_task-{3}_acq-{4}_space-fsLR_desc-qc_den-91k_bold.csv'.format(self.source_path,subject,self.task,self.session,self.acq))
			glob_scrub = glob.glob('{0}/DERIVATIVES/XCP/sub-{1}/func/sub-{1}_task-{3}_acq-{4}_space-fsLR_desc-framewisedisplacement_den-91k_bold.tsv'.format(self.source_path,subject,self.task,self.session,self.acq))
		elif self.source == 'hcpd-dcan':
			if loadtype == 'cifti': glob_matrices = glob.glob('/{0}/data/sub-{1}/ses-V1/files/MNINonLinear/Results/task-{2}_DCANBOLDProc_v4.0.0_Atlas.{3}.nii'.format(self.source_path,subject,self.task,name)) 
			else:glob_matrices = glob.glob('/{0}/data/sub-{1}/ses-V1/files/MNINonLinear/Results/task-{2}_DCANBOLDProc_v4.0.0_{3}.{4}.nii'.format(self.source_path,subject,self.task,self.parcels,name))
			glob_qc = glob.glob('/{0}/data/motion/fd/sub-{1}/ses-V1/files/DCANBOLDProc_v4.0.0/analyses_v2/motion/task-{2}_power_2014_FD_only.mat'.format(self.source_path,subject,self.task))
			glob_scrub = glob.glob('/{0}/data/motion/fd/sub-{1}/ses-V1/files/DCANBOLDProc_v4.0.0/analyses_v2/motion/task-{2}_power_2014_FD_only.mat'.format(self.source_path,subject,self.task))
		else: #RBC datasets
			#if we load cifti, that's the entire dense time series. This is very big, but i wanted people to have the option.
			if loadtype == 'cifti': glob_matrices = glob.glob('{0}/XCP/sub-{1}/{3}/func/sub-{1}_ses-{2}_task-{3}_acq-{4}_space-fsLR_den-91k_desc-residual_den-91k_bold.dtseries.nii'.format(self.source_path,subject,self.task,self.session,self.acq))
			#in most cases, we will just load the time series, either to pass that directly to the user, or to make the scrubbed matrix.
			else: glob_matrices = glob.glob('{0}/XCP/sub-{1}/{3}/func/sub-{1}_ses-{2}_task-{3}_acq-{4}_space-fsLR_atlas-{5}_den-91k_den-91k_bold.{6}.nii'.format(self.source_path,subject,self.task,self.session,self.acq,self.parcels,name))
			glob_qc = glob.glob('{0}/XCP/sub-{1}/{3}/func/sub-{1}_ses-{2}_task-{3}_acq-{4}_space-fsLR_desc-qc_den-91k_bold.csv'.format(self.source_path,subject,self.task,self.session,self.acq))
			# i call this scrub because it's the FD file, which we use to decide what frames to remove, ie scrub
			glob_scrub = glob.glob('{0}/XCP/sub-{1}/{3}/func/sub-{1}_ses-{2}_task-{3}_acq-{4}_space-fsLR_desc-framewisedisplacement_den-91k_bold.tsv'.format(self.source_path,subject,self.task,self.session,self.acq))
		if len(glob_matrices)==0: # this is mostly to warn people if there are subjects in the demographics that don't have data.
			print ('missing subject: {0}'.format(subject))
			missing.append(subject)
			continue
			
		# these are pretty important, might want to build something more robust or a test here. 
		# I checked that these work, but a change in the name of files could totally wreck that
		glob_scrub.sort()
		glob_qc.sort()
		glob_matrices.sort()

		# this is doing scrubbing with the DCAN data. I doubt many people will use this, but this is 
		# all we had for the time for HCPD, so I wrote it
		if self.source == 'hcpd-dcan':
			FD = []
			for qc in glob_qc:
				FD.append(scipy.io.loadmat(qc,squeeze_me=True,struct_as_record=False)['motion_data'][20].remaining_frame_mean_FD)
			self.subject_measures.loc[self.subject_measures.subject==subject,self.subject_measures.columns=='meanFD'] = np.mean(FD)		

			for ts,scrub_mask in zip(glob_matrices,glob_scrub):
				scrub_mask = scipy.io.loadmat(scrub_mask,squeeze_me=True,struct_as_record=False)['motion_data'][20].frame_removal
				ts = nib.load(ts).get_fdata()
				ts = ts[scrub_mask==0]
				if 'time_series' in locals(): time_series = np.append(time_series,ts.copy(),axis=0)
				else: time_series = ts.copy()
			if loadtype =='cifti': return np.mean(FD),time_series	
			
		else:
			post_scrub_fd = []
			# going to loop over each time series and scrub file
			for ts_file,scrub_mask in zip(glob_matrices,glob_scrub):
				#load em
				scrub_mask = pd.read_csv(scrub_mask,header=None).values.reshape(1,-1).flatten() #load the scrub / FD file
				ts = nib.load(ts_file).get_fdata() #load the time series
				if type(self.fd_scrub) == float: # if not a float, means no scrubbing.
					ts = ts[scrub_mask<=self.fd_scrub] #actual scrubbing
					post_scrub_fd.append(scrub_mask[scrub_mask<=self.fd_scrub]) #save post scrub FC

				# sub cortex is always in a different file, but sometimes we want it all in one big matrix/timeseries
				# so we do the same thing here as above for the sub-cortical parcels.
				# this is untested, so I am going to put a break line!
				if self.sub_cortex:
					1/0
					# this needs some editing on second glance,
					sub_ts = ts_file.replace(parcels,'subcortical')
					sub_ts = nib.load(sub_ts).get_fdata()
					if type(self.fd_scrub) == float: 
						sub_ts = sub_ts[scrub_mask<=self.fd_scrub]
					ts = np.append(ts,sub_ts.copy(),axis=1)
				# start to build the actual time series
				# this little line might be smart or dangerous, not really sure, but we check to see if the time_series exists		
				if 'time_series' in locals(): time_series = np.append(time_series,ts.copy(),axis=0)
				# if not, we intantiate it with this first time series
				else: time_series = ts.copy()

			"""
			okay, this is pulling the first qc file, getting the colums
			then it gets the QC files for just the requested tasks/runs/sessions/acq whatever
			then it gets the mean across them
			reading this, i was smart to do this. You now have a single QC value for each metric,
			for the specific BOLD scans you wanted.
			we also add post-scrubbing FD, which can be useful; when you scrub, you should probably use 
			this metric to control later analysis instead of the pre-scrub FD. 
			"""
			columns = pd.read_csv(glob_qc[0]).columns
			sub_df = pd.DataFrame(columns=columns)
			for qc in glob_qc:
				sub_df = sub_df.append(pd.read_csv(qc),ignore_index=True)
			sub_df = sub_df.groupby('sub').mean()
			sub_df['subject'] = sub_df.index
			sub_df['post_scrub_fd'] = np.mean([item for sublist in post_scrub_fd for item in sublist])  
			if 'qc_df' in locals(): qc_df = qc_df.append(sub_df,ignore_index=True)
			else: qc_df = sub_df.copy()
			if loadtype =='cifti': return sub_df,time_series	

		self.subject_measures.loc[self.subject_measures.subject==subject,self.subject_measures.columns=='n_frames'] = time_series.shape[0]
		
		if loadtype == 'matrix': # if the person wants the matrix, turn it in to matrix
			subject_matrix = np.corrcoef(time_series.swapaxes(0,1))
			self.matrix.append(subject_matrix)
		if loadtype =='ptseries':
			self.ptseries.append(time_series)
		del time_series

	if self.source != 'hcpd-dcan':self.subject_measures=self.subject_measures.merge(qc_df,on='subject')
	
	for missing_sub in missing: #remove missing subjcets from the dataset object
		missing_sub = self.subject_measures.loc[self.subject_measures.subject == missing_sub]
		self.subject_measures = self.subject_measures.drop(missing_sub.index,axis=0)

	#couple sanity checks
	if loadtype == 'matrix':
		self.matrix = np.array(self.matrix)
		assert len(self.matrix) == self.subject_measures.shape[0]
		if len(self.ptseries) != 0: assert len(self.ptseries) == len(self.matrix)

	elif loadtype == 'ptseries':
		assert len(self.ptseries) == self.subject_measures.shape[0]
		if len(self.matrix)!= 0: assert len(self.ptseries) == len(self.matrix)

class dataset:
	"""
	This is the main object to use to load an rbc dataset

	--------------
	parameters
	--------------
	task: str, the name of the task or ** to grab all tasks
	session: str, the name of the session, or ** to grab all sessions
	acq: str, the name of the scanner acquisition, or ** to grab all acquisitions
	parcels: str, Schaefer417, Gordon, Glasser,
	sub_cortex: bool, if you want this (https://github.com/yetianmed/subcortex) sub_cortical data added on
	fd_scrub: bool or float, False for no scrubbing, or float to remove frames with FD above fd_scrub
	source: str, the name of the dataset
	cores: int, the number of cores you will use for analysis
	"""
	def __init__(self, source,task='**', session = '**',acq='**',parcels='Schaefer417',sub_cortex=False,fd_scrub=False,cores=1):
		self.source = source
		"""
		really all that is happening here is getting the demographics csv
		i imagine you could also load the cubids and datalad information here,
		this is meant to sort of be a lightweight instantiation, so it runs super quick
		the load() functions take a while, but this gets you subject (and hopefully datalad info quickly)
		"""
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
			if self.source.upper()=='PNC': self.subject_measures = self.subject_measures.rename(columns={'reid':'subject'})
		self.cores = cores
		self.sub_cortex = sub_cortex
		self.parcels = parcels
		self.task = task
		self.session = session
		self.acq = acq
		self.fd_scrub = fd_scrub
		self.subject_measures['n_frames'] = np.nan #we make this later, so np.nan for now
		self.matrix = []
		self.ptseries = []

	def get_methods(self,modality='functional'):
		"""
		this won't work yet, but will load the methods text for papers
		"""
		resource_package = 'pennlinckit'
		resource_path = '{0}_boiler_{1}.txt'.format(self.source,modality)
		return np.loadtxt(resource_path)



	def load_matrices(self):
		"""
		load functional connectivity matrices from this dataset
		----------
		returns
		----------
		out : numpy matrix, np.nan down the diagonal
		note! this will also add the "n_frames" column to subject_measures, listing the number of frames remaining post scrubbing
		----------
		"""		
		load(self,'matrix',self.subject_measures.subject.values)

	def load_ptseries(self):
		"""
		load parcellated time series from this dataset
		----------
		returns
		----------
		out : a list of numpy matrices of the parcellated time series
		(why a list of numpy matrices? if you scrub, the matrices are different shapes, numpy does not like this)
		note! this will also add the "n_frames" column to subject_measures, listing the number of frames remaining post scrubbing
		----------
		"""
		load(self,'ptseries',self.subject_measures.subject.values)

	def load_cifti(self,subject):
		"""
		load cifti from a single subject
		----------
		returns
		----------
		out : nibabel object of the loaded cifti file
		note! this will also add the "n_frames" column to subject_measures, listing the number of frames remaining post scrubbing
		----------
		example (from PNC)
		----------
		# first subject in dataset
		qc,cifti = load_cifti(dataset.subject_measures.subject[0])
		"""
		return load(self,'cifti',[subject])

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
		
		This function is probably one of the most important ones. This is what people should use for subject inclusion 
		and subject exclusion.
		
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


def test():
	hcpd = dataset(source='hcpd-dcan',fd_scrub=0.1)  
	hcpya = dataset(source='hcpya',fd_scrub=0.1)  
	pnc = dataset(source='pnc',fd_scrub=0.1)  
	for data in [hcpd,hcpya,pnc]:
		data.subject_measures = data.subject_measures[:10]
		qc,c = data.load_cifti(data.subject_measures.subject[2])  
		data.load_matrices()
		data.load_ptseries()   

"""
#for testing

class self:
	def __init__(self):
		pass

source = 'pnc'
task = '**'
acq = '**'
parcels = 'Schaefer417'
sub_cortex = False
session = '**'
cores = 1
scrub_fd = .2
"""
