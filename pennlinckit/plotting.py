import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl
import pkg_resources


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
resource_path = 'Palatino.ttf'
path = pkg_resources.resource_stream(resource_package, resource_path)
mpl.font_manager.FontProperties(fname=path.name)
mpl.rcParams['font.family'] = 'serif'
sns.set(style='white',font='Palatino')
