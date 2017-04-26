from collections import Counter, OrderedDict
import re, sys, os
import pandas as pd
import numpy as np
from utils import unroll
# import seaborn as sns
# from cleaner import Cleaner

from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'FreeSerif'
plt.rcParams['font.style'] = 'italic'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 1
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 1


FILENAME = sys.argv[1]

data = open(FILENAME, 'r', encoding='utf8')
txt = data.read().split('\n')
chr_freq = Counter()
sent_sizes = Counter()

for i, line in enumerate(txt):
	# for c in line.strip():
	# 	if c == ' ':
	# 		continue
	# 	chr_freq[c] += 1
	sent_sizes[len(line.strip().split())] += 1
	sys.stdout.write('%d/%d\r' % (i, len(txt)))
	sys.stdout.flush()

print(sent_sizes)
ord_freq = Counter(dict(zip(list(map(ord, chr_freq.keys())), chr_freq.values())))

def showplot(d, min_f=5, title='Char Frequence'):
	a = OrderedDict(sorted([(k, v) for k, v in d.items() if v >= min_f]))
	x = np.arange(len(a))
	plt.bar(x, list(a.values()))
	plt.xticks(x, list(a.keys()))
	plt.title(title, fontsize=14)
	plt.show()


# showplot(chr_freq, min_f=100, title='Char Frequence')
# showplot(ord_freq, min_f=100, title='Ord Frequence')
showplot(sent_sizes, min_f=50, title='Sent Size Frequence')


# print(data)


# ord_data = list(map(lambda x: map(ord, x), data))
# chr_data = list(map(lambda x: list(x), data))
# np_ord = np.array(ord_data)

# print(np_ord)

# my_xticks = ['John','Arnold','Mavis','Matt']
# plt.xticks(x, my_xticks)

# plt.hist(np_ord, bins=50)



# plt.show()
