import seaborn as sns
import numpy as np
import os
import copy
import pandas as pd

def cut_data(data,min_cut=1.5,max_cut=1.5):
	d = data.copy()
	max_v = np.mean(d) + np.std(d)*max_cut
	min_v = np.mean(d) - np.std(d)*min_cut
	d[d>max_v] = max_v
	d[d<min_v] = min_v
	return d

def make_heatmap(data,cmap='stock'):
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

def write_cifti(atlas_path,out_path,colors):
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