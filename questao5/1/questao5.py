# -*- coding: utf-8 -*-
"""

@author: nataliapedroso
"""

# Questão 5

##############################################################################
################################################# Importando módulos e pacotes
##############################################################################

import funcs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import KElbowVisualizer

##############################################################################
#################################################### Funções que geram modelos
##############################################################################

############################################ Algoritmo do mapeamento logístico
# Usado na questão 5

def LogisticMap(dummy):
    N=8192
    rho=3.85 + 0.15*np.random.uniform()
    tau = 1.1
    x = [0.001]
    y = [0.001]
    for i in range(1,N):
      y.append( tau*x[-1] )
      x.append( rho*x[-1]*(1.0-x[-1]))
    return y,x

############################################# Algoritmo do mapeamento de Henon
# Usado na questão 5

def HenonMap(dummy):
    N=8192
    a=1.350 + 0.05*np.random.uniform()
    b=0.21 + 0.08*np.random.uniform()
    x = [0.1]
    y = [0.3]
    for i in range(1,N):
        y.append(b * x[-1])
        x.append(y[-2] + 1.0 - a *x[-1]*x[-1])
    return x,y

##############################################################################
################################################### Função que constrói séries
##############################################################################

def makeseries(func, iterationlist, amount):
    values=[]
    ilist=[]
    rawdata=[]
    for i in iterationlist:
        for j in range(amount):
            x,y=func(i)
            alfa,xdfa,ydfa, reta = funcs.dfa1d(y,1)
            freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = funcs.psd(y)
            values.append([funcs.variance(y), funcs.skewness(y), funcs.kurtosis(y)+3, alfa, index])
            ilist.append(i)
        rawdata.append([i,x,y, alfa, xdfa, ydfa, reta, freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM])
    return values, ilist, rawdata

##############################################################################
############################################## Função para construir o k-means
##############################################################################

def makeK(d,ilist, title):
    d=np.array(d)
    kk=pd.DataFrame({'Variance': d[:,0], 'Skewness': d[:,1], 'Kurtosis': d[:,2]})
    K=20
    model=KMeans()
    visualizer = KElbowVisualizer(model, k=(1,K))
    kIdx=visualizer.fit(kk)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    kIdx=kIdx.elbow_value_
    model=KMeans(n_clusters=kIdx).fit(kk)
    # scatter plot
    fig = plt.figure()
    ax = Axes3D(fig) #.add_subplot(111))
    cmap = plt.get_cmap('gnuplot')
    clr = [cmap(i) for i in np.linspace(0, 1, kIdx)]
    for i in range(0,kIdx):
        ind = (model.labels_==i)
        ax.scatter(d[ind,2],d[ind,1], d[ind,0], s=30, c=clr[i], label='Cluster %d'%i)
    
    ax.set_xlabel("Kurtosis")
    ax.set_ylabel("Skew")
    ax.set_zlabel("Variance")
    plt.title(title+': KMeans clustering with K=%d' % kIdx)
    plt.legend()
    plt.savefig(title+"clustersnoises.png")
    plt.show()
    d=pd.DataFrame({'Variance': d[:,0], 'Skewness': d[:,1], 'Kurtosis': d[:,2], 'Alpha': d[:,3], 'Beta': d[:,4], "Cluster": model.labels_}, index=ilist)
    return d

##############################################################################
################################### MAIN #####################################
##############################################################################
    
title="Série: Chaos Noise. {0}"
d,ilist,rawdata=makeseries(LogisticMap, ["Logistic"], 30)
aux1,aux2,aux3=makeseries(HenonMap, ["Henon"], 30)
rawdata+=aux3
d+=aux1
ilist+=aux2
for i in range(len(rawdata)):
    plt.figure(figsize=(20, 12))
    #Plot da série temporal
    ax1 = plt.subplot(211)
    ax1.set_title(title.format(rawdata[i][0]), fontsize=18)
    ax1.plot(rawdata[i][1],rawdata[i][2],color="firebrick", marker='o', linestyle='')
    #Plot e cálculo do DFA
    ax2 = plt.subplot(223)
    ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(rawdata[i][3], fontsize=15))
    ax2.plot(rawdata[i][4],rawdata[i][5], marker='o', linestyle='', color="#12355B")
    ax2.plot(rawdata[i][4], rawdata[i][6], color="#9DACB2")
    #Plot e cálculo do PSD
    ax3 = plt.subplot(224)
    ax3.set_title(r"Power Spectrum Density $\beta$={0:.3}".format(rawdata[i][12]), fontsize=15)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.plot(rawdata[i][7], rawdata[i][8], '-', color = 'deepskyblue', alpha = 0.7)
    ax3.plot(rawdata[i][9], rawdata[i][10], color = "darkblue", alpha = 0.8)
    ax3.axvline(rawdata[i][7][rawdata[i][14]], color = "darkblue", linestyle = '--')
    ax3.axvline(rawdata[i][7][rawdata[i][15]], color = "darkblue", linestyle = '--')    
    ax3.plot(rawdata[i][9], rawdata[i][13](rawdata[i][9], rawdata[i][11], rawdata[i][12]),color="#D65108", linestyle='-', linewidth = 3, label = '$%.4f$' %(rawdata[i][12]))
    ax2.set_xlabel("log(s)")
    ax2.set_ylabel("log F(s)")
    ax3.set_xlabel("Frequência (Hz)")
    ax3.set_ylabel("Potência")
    plt.savefig("Chaosserietemporalpsddfa{}.png".format(i))
    plt.show()
title="chaosnoise"
d=makeK(d,ilist, title)