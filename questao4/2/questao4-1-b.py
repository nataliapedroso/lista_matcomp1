# -*- coding: utf-8 -*-
"""

@author: nataliapedroso
"""

# Questão 4.1 usando a questão 2
# Classificando as séries no espaço de Cullen-Frey

##############################################################################
################################################# Importando módulos e pacotes
##############################################################################

import funcs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as rnd
from scipy.stats import norm
from numpy.fft import rfftfreq, irfft
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
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

####################################################### Algoritmo colorednoise
# Enviado pelo professor - usado na questão 2
    
def powerlaw_psd_gaussian(exponent, size=8192, fmin=0):
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)
    
    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./samples) # Low frequency cutoff
    ix   = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)
    
    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(np.newaxis,) * dims_to_add + (Ellipsis,)]
    
    # Generate scaled random power + phase
    sr = rnd.normal(scale=s_scale, size=size)
    si = rnd.normal(scale=s_scale, size=size)
    
    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2): si[...,-1] = 0
    
    # Regardless of signal length, the DC component must be real
    si[...,0] = 0
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    x=range(0,len(y))
    return x,y

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
################################################### Funções que geram gráficos
##############################################################################

######################################################### Algoritmo cullenfrey
# Versão do Giovanni

def cullenfrey(xd,yd,legend, title):
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots()
    maior=max(xd)
    polyX1=maior if maior > 4.4 else 4.4
    polyY1=polyX1+1
    polyY2=3/2.*polyX1+3
    y_lim = polyY2 if polyY2 > 10 else 10
    
    x = [0,polyX1,polyX1,0]
    y = [1,polyY1,polyY2,3]
    scale = 1
    poly = Polygon( np.c_[x,y]*scale, facecolor='#1B9AAA', edgecolor='#1B9AAA', alpha=0.5)
    ax.add_patch(poly)
    ax.plot(xd,yd, marker="o", c="#e86a92", label=legend, linestyle='')
    ax.plot(0, 4.187999875999753, label="logistic", marker='+', c='black')
    ax.plot(0, 1.7962675925351856, label ="uniform", marker='^',c='black')
    ax.plot(4, 9, label="exponential", marker='s', c='black')
    ax.plot(0, 3, label="normal", marker='*',c='black')
    ax.plot(np.arange(0,polyX1,0.1), 3/2.*np.arange(0,polyX1,0.1)+3, label="gamma", linestyle='-',c='black')
    ax.plot(np.arange(0,polyX1,0.1), 2*np.arange(0,polyX1,0.1)+3, label="lognormal", linestyle='-.',c='black')
    ax.legend()
    ax.set_ylim(y_lim,0)
    ax.set_xlim(-0.2,polyX1)
    plt.xlabel("Skewness²")
    plt.title(title+": Cullen and Frey map")
    plt.ylabel("Kurtosis")
    plt.savefig(title+legend+"cullenfrey.png")
    plt.show()

##############################################################################
################################### MAIN #####################################
##############################################################################
    
title="Série: Colored Noise. Expoente = {0}"
d,ilist,rawdata=makeseries(powerlaw_psd_gaussian, range(0,3), 20)
while(0 in ilist or 1 in ilist or 2 in ilist):
    ilist[ilist.index(0)] = 'white noise'
    ilist[ilist.index(1)] = 'pink noise'
    ilist[ilist.index(2)] = 'red noise'
    for i in range(len(rawdata)):
        #Plot e ajuste do histograma da série temporal
        (mu,sigma)=norm.fit(rawdata[i][2])
        plt.title((title+"\nMu= {1:.3}, Sigma={2:.3}.").format(rawdata[i][0], mu, sigma))
        n, bins, patches = plt.hist(rawdata[i][2], 60, density=1, facecolor='powderblue', alpha=0.75)
        plt.plot(bins,norm.pdf(bins,mu,sigma), c="black", linestyle='--')
        plt.savefig("colorednoise{}PDF.png".format(i))
        plt.show()
        plt.figure(figsize=(20, 12))
        #Plot da série temporal
        ax1 = plt.subplot(211)
        ax1.set_title(title.format(rawdata[i][0]), fontsize=18)
        ax1.plot(rawdata[i][1],rawdata[i][2],color="firebrick", linestyle='-')
        #Plot e cálculo do DFA
        ax2 = plt.subplot(223)
        ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(rawdata[i][3]), fontsize=15)
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
        ax2.set_xlabel("log(s)")
        ax2.set_ylabel("log F(s)")
        ax3.set_xlabel("Frequência (Hz)")
        ax3.set_ylabel("Potência")
        plt.savefig("CNserietemporalpsddfa{}.png".format(i))
        plt.show()
    title="colorednoise"
    d=makeK(d,ilist, title)

s2=[i**2 for i in d.filter(like=str(8192), axis=0)['Skewness']] #skew²
k=d.filter(like=str(8192),axis=0)["Kurtosis"]
legend=str(8192)
cullenfrey(s2,k, legend, title)