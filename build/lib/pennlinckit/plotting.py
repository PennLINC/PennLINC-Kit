import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
import matplotlib as mpl
import pkg_resources
import glob
import os
import subprocess

try: matplotlib.use('TkAgg')
except: matplotlib.use('nbagg')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = "Palatino"
plt.rcParams['font.serif'] = "Palatino"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Palatino:italic'
plt.rcParams['mathtext.bf'] = 'Palatino:bold'
plt.rcParams['mathtext.cal'] = 'Palatino'
plt.rcParams["figure.figsize"] = (7.44094,(7.44094/3)*2) #nature width for figures, stock aspect
resource_package = 'pennlinckit'
resource_path = 'Palatino*'
path = pkg_resources.resource_filename(resource_package, resource_path)

python = subprocess.check_output(['which python'],shell=True).decode().split('/bin/python')[0]
if len(glob.glob('/%s/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/Palatino**'%(python))) == 0:
	os.system('cp %s /%s/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/'%(path,python))
	os.system('rm -f -r ~/.cache/matplotlib')
resource_path = 'Palatino.ttf'
path = pkg_resources.resource_stream(resource_package, resource_path)
mpl.font_manager.FontProperties(fname=path.name)
mpl.rcParams['font.family'] = 'serif'
sns.set(style='white',font='Palatino')
