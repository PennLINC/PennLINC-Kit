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
import brainsmash.mapgen.base
import brainsmash.mapgen.stats
from scipy.stats import pearsonr


def vol2fslr(volume,out,roi=False,wb_path='/cbica/home/bertolem/workbench//bin_rh_linux64/wb_command'):
	"""
	wraps the connectome workbench method
	Parameters:
		volume: str, path to nifti, 2mm MNI152 image
		out: str, what you want to call your surface
		roi: False for continuous values, True for ROIs /
		"parcels" if you don't want values "mixing".
		wb_path: this is for CUBIC, edit if you are running locally
	
	Returns:
		writes out a surface version of your volume in fslr32k space. You can view
		this with wb_view 
	"""

	resource_package = 'pennlinckit'
	resource_path = 'Q1-Q6_R440.HEMI.SURFACE.32k_fs_LR.surf.gii'
	file = pkg_resources.resource_filename(resource_package, resource_path)
	lh_inflated = file.replace('HEMI','L').replace('SURFACE','inflated')
	rh_inflated = file.replace('HEMI','R').replace('SURFACE','inflated')
	lh_pial = file.replace('HEMI','L').replace('SURFACE','pial')
	rh_pial = file.replace('HEMI','R').replace('SURFACE','pial')
	lh_white = file.replace('HEMI','L').replace('SURFACE','white')
	rh_white = file.replace('HEMI','R').replace('SURFACE','white')

	if roi == True:
		right_command = "%s -volume-to-surface-mapping %s %s \
		%s.R.func.gii \
		-ribbon-constrained %s %s \
		-volume-roi %s -interpolate ENCLOSING_VOXEL" %(wb_path,volume,rh_inflated,out,rh_white,rh_pial,volume)
		left_command = "%s -volume-to-surface-mapping %s %s \
		%s.L.func.gii \
		-ribbon-constrained %s %s \
		-volume-roi %s -interpolate ENCLOSING_VOXEL"%(wb_path,volume,lh_inflated,out,lh_white,lh_pial,volume)

	if roi == False:
		right_command = "%s -volume-to-surface-mapping %s %s \
		%s.R.func.gii \
		-ribbon-constrained %s %s -interpolate ENCLOSING_VOXEL" %(wb_path,volume,rh_inflated,out,rh_white,rh_pial)
		left_command = "%s -volume-to-surface-mapping %s %s \
		%s.L.func.gii \
		-ribbon-constrained %s %s -interpolate ENCLOSING_VOXEL" %(wb_path,volume,lh_inflated,out,lh_white,lh_pial)

	os.system(left_command)
	os.system(right_command)
	os.system('%s -cifti-create-dense-scalar %s.dscalar.nii -left-metric %s.L.func.gii -right-metric %s.R.func.gii' %(wb_path,out,out,out))


def fs5_to_hcp(lh,rh,out,img_type='label',wb_path='/cbica/home/bertolem/workbench//bin_rh_linux64/wb_command'):	

	"""
	maps from fsaverage5 to fslr32k aka Human Connectome Specat

	Parameters:

		lr: left hemisphere fsaverage file
		rh: right hemisphere fsaverage file
		out: name for new file
		img_type: label (for parcels) or metric (for analysis / results)

	Returns: writes out a left and right gii file in fslr32k
	"""
	resource_package = 'pennlinckit'
	resource_path = 'standard_mesh_atlases/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii'
	path_2_maps = pkg_resources.resource_filename(resource_package, resource_path).split('/') 
	path_2_maps = '/'.join(path_2_maps.split('/')[:-1]) + '/resample_fsaverage/'
	if img_type == 'label':out_img_type = 'dlabel'
	if img_type == 'metric':out_img_type = 'dscalar'

	lh_name = '{0}.32k_fs_LR.lh.{1}.gii'.format(out,out_img_type)
	rh_name = '{0}.32k_fs_LR.rh.{1}.gii'.format(out,out_img_type)

	lh_cmd = '{0} -{1}-resample {2} {3}/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii \
		{3}/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii ADAP_BARY_AREA {4} \
		-area-metrics {3}/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii \
		{3}/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii'.format(wb_path,img_type,lh,path_2_maps,lh_name)

	os.system(lh_cmd)
	rh_cmd = '{0} -{1}-resample {2} {3}/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii \
		{3}/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii ADAP_BARY_AREA {4} \
		-area-metrics {3}/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii \
		{3}/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii'.format(wb_path,img_type,rh,path_2_maps,rh_name)
	os.system(rh_cmd)	

def hcp_to_fs5(input,out,img_type='label',wb_path='/cbica/home/bertolem/workbench//bin_rh_linux64/wb_command'):

	"""
	maps from fslr32k aka Human Connectome Space to fsaverage5

	Parameters:

		input: fslr32k cifti file
		out: name for new file
		img_type: label (for parcels) or metric (for analysis / results)

	Returns: 
		writes out a left and right gii file in fsaverage5
	"""

	resource_package = 'pennlinckit'
	resource_path = 'standard_mesh_atlases/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii'
	path_2_maps = pkg_resources.resource_filename(resource_package, resource_path).split('/')
	path_2_maps = '/'.join(path_2_maps.split('/')[:-1]) + '/resample_fsaverage/'
	cmd = '{0} -cifti-separate {1} COLUMN -{2} CORTEX_LEFT {3}-LR.{2}.gii'.format(wb_path,input,img_type,out)
	os.system(cmd)
	cmd = '{0} -cifti-separate {1} COLUMN -{2} CORTEX_RIGHT {3}-RH.{2}.gii'.format(wb_path,input,img_type,out)
	os.system(cmd)

	rh_cmd = '{0} -{1}-resample {2}-LH.{1}.gii {3}/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii \
		{3}/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii ADAP_BARY_AREA {2}-LH.{1}.gii -area-metrics \
		{3}/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii \
		{3}/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii'.format(wb_path,img_type,out,path_2_maps)
	os.system(rh_cmd)

	rh_cmd = '{0} -{1}-resample {2}-LH.{1}.gii {3}/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii \
		{3}/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii ADAP_BARY_AREA {2}-RH.{1}.gii -area-metrics \
		{3}/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii \
		{3}/fsaverage5.R.midthickness_va_avg.10k_fsavg_L.shape.gii'.format(wb_path,img_type,out,path_2_maps)
	os.system(rh_cmd)

def cerebellum_vol2surf(input,output):
	"""
	Uses a pretty great MATLAB library to map volume to surface in the cerebellum
	
	Parameters:
		input: MNI space volume
		output: string to save your surface to
	
	Returns:
		gii file of your surface cerebellum, in fsaverage5 space,
		you can view this in wb_view
	"""
	command = """matlab -nosplash -nodesktop -r "addpath /cbica/home/bertolem/spm12/toolbox/suit/;addpath\
	 /cbica/home/bertolem/spm12/;C.cdata=suit_map2surf('{0}','space','FSL');C=gifti(C);save(C,'{1}.func.gii');exit" """.format(input,output)
	os.system(command)

def view_surf(surf,hemi='left'):
	"""
	plots an inline version of the cortical surface data in fslr32k space

	Parameters:
		surf: your data as an array
		hemi: left or right

	"""
	resource_package = 'pennlinckit'
	resource_path = 'Q1-Q6_R440.HEMI.SURFACE.32k_fs_LR.surf.gii'
	file = pkg_resources.resource_filename(resource_package, resource_path)
	if hemi == 'left': inflated = file.replace('HEMI','L').replace('SURFACE','inflated')
	if hemi == 'right': inflated = file.replace('HEMI','R').replace('SURFACE','inflated')
	nilearn.plotting.view_surf(inflated,surf)

def view_nifti(path):
	"""
	plots an inline version of the volume data 

	Parameters:
		path: path to the nifti volume
	"""
	nifti = nib.load(path)
	nifti_data = nifti.get_fdata()
	nib.viewers.OrthoSlicer3D(nifti_data,nifti.affine)

def yeo_partition(n_networks=17,parcels='Schaefer400'):
	"""
	Loads the Schaefer network labels and names for 17 or 7 networks
	We require you used the 17 network parcels, but we can reduce this to 7 networks if you request it

	Parameters:
		n_networks: 7 or 17

	Returns:
		membership: string based names for the 7 or 17 networks
		membership_ints: int based names for the 7 or 17 networks
		names: corresponding names for each network
	"""

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

def spin_test(map1,map2,parcels='Schaefer400',n=1000):
	"""
	Uses a precomputed cortical distance file to run the brainsmash spin test for the 
	Schaefer400 parcels

	Parameters:
		map1: one cortical map you care about
		map2: another coritcal map you care about

	Returns:
		p_value of the spin test
	"""
    if parcels == 'Schaefer400': split = 200 #where the right hemi starts
        resource_package = 'pennlinckit'
        resource_path = '%s_ROIwise_geodesic_distance_midthickness.mat'%(parcels)
        mat_file = scipy.io.loadmat(pkg_resources.resource_stream(resource_package, resource_path))
        lh_gen = brainsmash.mapgen.base.Base(map2[:split], D= mat_file['lh_dist'])
        lh_maps = lh_gen(n=n)
        rh_gen = brainsmash.mapgen.base.Base(map2[split:], D= mat_file['rh_dist'])
        rh_maps = rh_gen(n=n)
        maps = np.append(lh_maps,rh_maps,axis=1)
        assert (lh_maps[0] == maps[0,:200]).all()
        return brainsmash.mapgen.stats.pearsonr(map1,maps)[0]

def spin_stat(map1,map2,spincorrs):
	"""
	helper function for brain.spin_test
	"""
    real_r = pearsonr(map1,map2)[0]
    if real_r >= 0.0:
        smash_p = len(spincorrs[spincorrs>real_r])/float(len(spincorrs))
    else:
        smash_p = len(spincorrs[spincorrs<real_r])/float(len(spincorrs))
    return smash_p

def write_cifti(colors,out_path,parcels='Schaefer400',wb_path='/cbica/home/bertolem/workbench/bin_rh_linux64/wb_command'):
	"""
	You have data in the Schaefer 400 parcels, you want it on the cortical surface

    Parameters:
		colors: RGB valus for each parcel
		atlas_path: the cifti file you are basing this on
		out_path: what you want to call it!
    Returns:
	    out_path.dlabel file to open on connectome_wb
	"""
	if parcels=='Schaefer400':
		resource_path = 'Schaefer2018_400Parcels_17Networks_order.dlabel.nii'
		resource_package = 'pennlinckit'
		atlas_path = pkg_resources.resource_stream(resource_package, resource_path).name
	os.system('{0} -cifti-label-export-table {1} 1 temp.txt'.format(wb_path,atlas_path))
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
	os.system('{0} -cifti-label-import {1} temp.txt {2}.dlabel.nii'.format(wb_path,atlas_path,out_path))
	os.system('rm temp.txt')

def three_d_dist(p1,p2):
	return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

def distance_mni(nifti,roi,mni):
	# roi = 1
	# mni = 50,-25,50
	# nifti = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii'

	nifti = nib.load(nifti)
	nifti_data = nifti.get_fdata()
	nifti.dataobj

	r = three_d_dist(np.mean(np.argwhere(nifti_data==roi),axis=0),real_2_mm(nifti,mni))
	return r

def distance_voxel_coord(nifti,roi,coord):
	# roi = 1
	# mni = 50,25,50
	# nifti = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii'

	nifti = nib.load(nifti)
	nifti_data = nifti.get_fdata()
	nifti.dataobj

	r = three_d_dist(np.mean(np.argwhere(nifti_data==roi),axis=0),coord)
	return r

def real_2_mm(target_image, real_pt):
	aff = target_image.affine
	return nib.affines.apply_affine(npl.inv(aff), real_pt)
