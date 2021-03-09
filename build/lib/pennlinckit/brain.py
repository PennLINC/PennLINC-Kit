import seaborn as sns
import numpy as np
import os
import copy
import scipy.io
import pandas as pd
import pkg_resources
import nibabel as nib
import numpy.linalg as npl
import math
import nilearn.plotting


def vol2fslr(volume,out):
	resource_package = 'pennlinckit'
	resource_path = 'Q1-Q6_R440.HEMI.SURFACE.32k_fs_LR.surf.gii'
	file = pkg_resources.resource_filename(resource_package, resource_path)
	lh_inflated = file.replace('HEMI','L').replace('SURFACE','inflated')
	rh_inflated = file.replace('HEMI','R').replace('SURFACE','inflated')
	lh_pial = file.replace('HEMI','L').replace('SURFACE','pial')
	rh_pial = file.replace('HEMI','R').replace('SURFACE','pial')
	lh_white = file.replace('HEMI','L').replace('SURFACE','white')
	rh_white = file.replace('HEMI','R').replace('SURFACE','white')

	left_command = "wb_command -volume-to-surface-mapping %s \
	%s.L.func.gii \
	-ribbon-constrained %s %s \
	-interpolate ENCLOSING_VOXEL -thin-columns" %(lh_inflated,out,lh_white,lh_pial)


	right_command = "wb_command -volume-to-surface-mapping %s\
	%s.R.func.gii\
	-ribbon-constrained %s %s\
	-interpolate ENCLOSING_VOXEL -thin-columns" %(rh_inflated,out,rh_white,rh_pial)

	os.system(left_command)
	os.system(right_command)

def view_surf(surf,hemi='left'):
	resource_package = 'pennlinckit'
	resource_path = 'Q1-Q6_R440.HEMI.SURFACE.32k_fs_LR.surf.gii'
	file = pkg_resources.resource_filename(resource_package, resource_path)
	if hemi == 'left': inflated = file.replace('HEMI','L').replace('SURFACE','inflated')
	if hemi == 'right': inflated = file.replace('HEMI','R').replace('SURFACE','inflated')
	nilearn.plotting.view_surf(inflated,surf)

def view_nifti(path):
	nifti = nib.load(path)
	nifti_data = nifti.get_fdata()
	nib.viewers.OrthoSlicer3D(nifti_data,nifti.affine)

def yeo_partition(n_networks=17,parcels='Schaefer400'):
	if parcels=='Schaefer400': resource_path = 'Schaefer2018_400Parcels_17Networks_order_info.txt'
	resource_package = 'pennlinckit'
	yeo_file = pkg_resources.resource_stream(resource_package, resource_path).name
	full_dict_17 = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6, 'SalVentAttnB':7,'LimbicA':8,
		'LimbicB':9,'ContA':10,'ContB':11,'ContC':12,'DefaultA':13,'DefaultB':14,'DefaultC':15,'TempPar':16}
	full_dict_7  = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3, 'SalVentAttnB':3,'LimbicA':4,
		'LimbicB':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
	name_dict = {0:'Visual',1:'Sensory\nMotor',2:'Dorsal\nAttention',3:'Ventral\nAttention',4:'Limbic',5:'Control',6:'Default'}
	membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv(yeo_file,sep='\t',header=None,names=['name','R','G','B','0'])['name']
	for i,n in enumerate(yeo_df[::2]):
		if n_networks == 17:
			membership[i] = n.split('_')[2]
			membership_ints[i] = int(full_dict_17[membership[i]])
		if n_networks == 7:
			membership_ints[i] = int(full_dict_7[n.split('_')[2]])
			membership[i] = name_dict[membership_ints[i]]

	if n_networks == 17: names = ['VisCent','VisPeri','SomMotA','SomMotB','DorsAttnA','DorsAttnB','SalVentAttnA','SalVentAttnB','LimbicA','LimbicB,''ContA','ContB','ContC','DefaultA','DefaultB','DefaultC','TempPar']
	if n_networks == 7: names = ['Visual','Sensory Motor','Dorsal Attention','Ventral Attention', 'Limbic','Control','Default']

	return membership,membership_ints,names

def spin_test(data,hemi='lh',parcels='Schaefer400'):
	resource_package = 'pennlinckit'
	resource_path = '%s_ROIwise_geodesic_distance_midthickness.mat'%(parcels)
	mat_file = scipy.io.loadmat(pkg_resources.resource_stream(resource_package, resource_path))
	hemi_dist = mat_file['%s_dist'%(hemi)]

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

def write_cifti(colors,out_path,parcels='Schaefer400'):
	"""
	You have data, you want it on the brain

    Parameters
    ----------
    colors: RGB valus for each parcel
    atlas_path: the cifti file you are basing this on
    out_path: what you want to call it!
    Returns
    -------
    out : out_path.dlabel file to open on connectome_wb
	"""
	if parcels=='Schaefer400':
		resource_path = 'Schaefer2018_400Parcels_17Networks_order.dlabel.nii'
		resource_package = 'pennlinckit'
		atlas_path = pkg_resources.resource_stream(resource_package, resource_path).name
	os.system('wb_command -cifti-label-export-table %s 1 temp.txt'%(atlas_path))
	df = pd.read_csv('temp.txt',header=None)
	for i in range(df.shape[0]):
		try:
			d = np.array(df[0][i].split(' ')).astype(int)
		except:
			continue
		real_idx = d[0] -1
		try: df[0][i] = str(d[0]) + ' ' + str(int(colors[real_idx][0]*255)) + ' ' + str(int(colors[real_idx][1]*255)) + ' ' + str(int(colors[real_idx][2]*255)) + ' ' + str(int(colors[real_idx][3]*255))
		except: df[0][i] = str(d[0]) + ' ' + str(int(colors[real_idx][0]*255)) + ' ' + str(int(colors[real_idx][1]*255)) + ' ' + str(int(colors[real_idx][2]*255)) + ' ' + str(255)
	df.to_csv('temp.txt',index=False,header=False)
	os.system('wb_command -cifti-label-import %s temp.txt %s.dlabel.nii'%(atlas_path,out_path))
	os.system('rm temp.txt')

def three_d_dist(p1,p2):
	return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

def distance(nifti,roi,mni):
	# roi = 1
	# mni = 50,25,50
	# nifti = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii'

	nifti = nib.load(nifti)
	nifti_data = nifti.get_fdata()
	nifti.dataobj

	np.mean(np.argwhere(nifti_data==roi),axis=0)

	r = three_d_dist(np.mean(np.argwhere(nifti_data==roi),axis=0),real_2_mm(nifti,mni))

def real_2_mm(target_image, real_pt):
	aff = target_image.affine
	return nib.affines.apply_affine(npl.inv(aff), real_pt)
