# -*- coding: utf-8 -*-
"""

@author: nataliapedroso
"""

# Questão 3

##############################################################################
################################################# Importando módulos e pacotes
##############################################################################

import funcs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import KElbowVisualizer

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
###################################################### Função que gera modelos
##############################################################################
    
############################################################# Algoritmo pmodel
# Enviado pelo professor - usado na questão 3

def pmodel(seriestype):
    if(seriestype=="Endogenous"):
        p=0.32 + 0.1*rnd.uniform()
    else:
        p=0.18 + 0.1*rnd.uniform()
    noValues=8192
    slope=0.4
    noOrders = int(np.ceil(np.log2(noValues)))
    
    y = np.array([1])
    for n in range(noOrders):
        y = next_step_1d(y, p)
    
    if (slope):
        fourierCoeff = fractal_spectrum_1d(noValues, slope/2)
        meanVal = np.mean(y)
        stdy = np.std(y)
        x = np.fft.ifft(y - meanVal)
        phase = np.angle(x)
        x = fourierCoeff*np.exp(1j*phase)
        x = np.fft.fft(x).real
        x *= stdy/np.std(x)
        x += meanVal
    else:
        x = y
    return x[0:noValues], y[0:noValues]


def next_step_1d(y, p):
    y2 = np.zeros(y.size*2)
    sign = np.random.rand(1, y.size) - 0.5
    sign /= np.abs(sign)
    y2[0:2*y.size:2] = y + sign*(1-2*p)*y
    y2[1:2*y.size+1:2] = y - sign*(1-2*p)*y
    
    return y2


def fractal_spectrum_1d(noValues, slope):
    ori_vector_size = noValues
    ori_half_size = ori_vector_size//2
    a = np.zeros(ori_vector_size)
    
    for t2 in range(ori_half_size):
        index = t2
        t4 = 1 + ori_vector_size - t2
        if (t4 >= ori_vector_size):
            t4 = t2
        coeff = (index + 1)**slope
        a[t2] = coeff
        a[t4] = coeff
        
    a[1] = 0
    
    return a

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
    
qtd=1
p=[]
for i in range(qtd):
    p.append("Endogenous")
for i in range(qtd):
    p.append("Exogenous")
title="Série: pmodel. {0}"
d,ilist,rawdata=makeseries(pmodel, p,30)
for i in range(len(rawdata)):
    plt.figure(figsize=(20, 12))
    #Plot da série temporal
    ax1 = plt.subplot(211)
    ax1.set_title(title.format(rawdata[i][0]), fontsize=18)
    ax1.plot(rawdata[i][2],color="firebrick", linestyle='-')
    #Plot e cálculo do DFA
    ax2 = plt.subplot(223)
    ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(rawdata[i][3]), fontsize=15)
    plt.plot(rawdata[i][4],rawdata[i][5], marker='o', linestyle='', color="#12355B")
    plt.plot(rawdata[i][4], rawdata[i][6], color="#9DACB2")
    #Plot e cálculo do PSD
    ax3 = plt.subplot(224)
    ax3.set_title(r"Power Spectrum Density $\beta=${0:.3}".format(rawdata[i][12]), fontsize=15)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.plot(rawdata[i][7], rawdata[i][8], '-', color = 'deepskyblue', alpha = 0.7)
    ax3.plot(rawdata[i][9], rawdata[i][10], color = "darkblue", alpha = 0.8)
    ax3.axvline(rawdata[i][7][rawdata[i][14]], color = "darkblue", linestyle = '--')
    ax3.axvline(rawdata[i][7][rawdata[i][15]], color = "darkblue", linestyle = '--')    
    ax3.plot(rawdata[i][9], rawdata[i][13](rawdata[i][9], rawdata[i][11], rawdata[i][12]), color="#D65108", linestyle='-', linewidth = 3, label = '$%.4f$' %(rawdata[i][12]))
    ax2.set_xlabel("log(s)")
    ax2.set_ylabel("log F(s)")
    ax3.set_xlabel("Frequência (Hz)")
    ax3.set_ylabel("Potência")
    plt.savefig("Pmodelserietemporalpsddfa{}.png".format(i))
    plt.show()
title="pmodel"
d=makeK(d,ilist, title)
