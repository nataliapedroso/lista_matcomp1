# -*- coding: utf-8 -*-
"""

@author: nataliapedroso
"""

# Questão 6.3

##############################################################################
################################################# Importando módulos e pacotes
##############################################################################

import mfdfa
import funcs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

##############################################################################
################################### MAIN #####################################
##############################################################################

namefile="daily-cases-covid-19.csv"
l=pd.read_csv(namefile)
codes=list(set(l["Entity"]))
codes=codes[1:]
l=l.set_index("Entity")
values=[]
countries=["Brazil", "India", "Iran", "South Africa", "Egypt" ]
for i in codes:
    y=list(l.filter(like=i, axis=0)["Daily confirmed cases (cases)"])
    if len(y) > 50:
        alfa,xdfa,ydfa, reta = funcs.dfa1d(y,1)
        freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = funcs.psd(y)
        values.append([funcs.variance(y), funcs.skewness(y), funcs.kurtosis(y), alfa, index, mfdfa.makemfdfa(y), i])

skew2=[]
alfa=[]
kurt=[]
index=[]
psi=[]

for i in range(len(values)):
    skew2.append(values[i][1]**2)
    kurt.append(values[i][2])
    alfa.append(values[i][3])
    index.append(values[i][6])
    
skew2=np.array(skew2)
alfa=np.array(alfa)
kurt=np.array(kurt)

kk=pd.DataFrame({'Skew²': skew2, 'Alpha': alfa})
K=20
model1=KMeans()
visualizer = KElbowVisualizer(model1, k=(1,K))
kIdx=visualizer.fit(kk)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
kIdx=kIdx.elbow_value_
model1=KMeans(n_clusters=kIdx).fit(kk)
# scatter plot
ax = plt.figure()
cmap = plt.get_cmap('gnuplot')
clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
for i in range(0,kIdx):
    ind = (model1.labels_==i)
    plt.scatter(skew2[ind],alfa[ind], s=30, c=clr[i], label='Cluster %d'%i)

plt.xlabel("Skew²")
plt.ylabel("Alfa")
plt.title('KMeans clustering with K=%d' % kIdx)
plt.legend()
plt.show()

kk=pd.DataFrame({'Skew²': skew2,'Alpha': alfa,'Cluster skew²': model1.labels_}, index=index)
kk=kk.sort_values(by=["Cluster skew²"])
kk.to_csv("sort_by_skew.csv")

kk=pd.DataFrame({'Kurtosis': kurt, 'Alpha': alfa})
K=20
model2=KMeans()
visualizer = KElbowVisualizer(model2, k=(1,K))
kIdx=visualizer.fit(kk)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure
kIdx=kIdx.elbow_value_
model2=KMeans(n_clusters=kIdx).fit(kk)
# scatter plot
ax = plt.figure()
cmap = plt.get_cmap('gnuplot')
clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
for i in range(0,kIdx):
    ind = (model2.labels_==i)
    plt.scatter(kurt[ind],alfa[ind], s=30, c=clr[i], label='Cluster %d'%i)

plt.xlabel("Kurtosis")
plt.ylabel("Alfa")
plt.title('KMeans clustering with K=%d' % kIdx)
plt.legend()
plt.show()

kk=pd.DataFrame({'Kurt': kurt,'Alpha': alfa,'Cluster kurt': model2.labels_}, index=index)
kk=kk.sort_values(by=["Cluster kurt"])
kk.to_csv("sort_by_kurt.csv")