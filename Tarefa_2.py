import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import scipy.stats as st
from matplotlib.patches import Rectangle
import math





#DATFRAME:

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])


df = df.transpose()

import math

mean = list(df.mean())

std = list(df.std())

ye1 = []

for i in range (4) :
    ye1.append(1.96*(std[i]/math.sqrt(len(df))))
    
nearest = 100
Y = 39500

df_p = pd.DataFrame()

df_p['diff'] = nearest*((Y - df.mean())//nearest)

df_p['sign'] = df_p['diff'].abs()/df_p['diff']

old_range = abs(df_p['diff']).min(), df_p['diff'].abs().max()

new_range = .5,1

df_p['shade'] = df_p['sign']*np.interp(df_p['diff'].abs(), old_range, new_range)
# changing color of the bars with respect to Y
shade = list(df_p['shade'])


color = ['blue','red','grey','red']
fig,ax = plt.subplots(num = None,figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')

ax.bar(range(len(df.columns)), height = df.values.mean(axis = 0), 
        yerr=ye1, error_kw={'capsize': 10, 'elinewidth': 2, 'alpha':0.5}, color = color)

ax.axhline(y=Y, color = 'black', label = 'Y')


ax.set_xticks(range(len(df.columns)), df.columns)
plt.text(3.7, 40000, "39500")

ax.set_title('DATA GERADA DE 1992 - 95')
ax.tick_params(top='off', bottom='off',  right='off', labelbottom='on')
ax.set_ylim(0,55000)
ax.set_yticks([0,10000,25000,35000,45000,55000])



plt.show()