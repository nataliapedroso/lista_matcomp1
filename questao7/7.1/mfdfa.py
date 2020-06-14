# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:18:33 2020

@author: nataliapedroso
"""

# Questão 7.1

##############################################################################
################################################# Importando módulos e pacotes
##############################################################################

import numpy as np
import matplotlib.pyplot as plt

##############################################################################
######################### Função usadas no cáluclo do Modified first-order DFA
##############################################################################
    
################################################### Função getHurstByUpscaling
    
def getHurstByUpscaling(dx, normType_p = np.inf, isDFA = 1, normType_q = 1.0):
    ## Some initialiation
    dx_len = len(dx)
    
    # We have to reserve the most major scale for shifts, so we divide the data
    # length by two. (As a result, the time measure starts from 2.0, not from
    # 1.0, see below.)
    dx_len = np.int(dx_len / 2)
    
    dx_shift = np.int(dx_len / 2)
    
    nScales = np.int(np.round(np.log2(dx_len)))    # Number of scales involved. P.S. We use 'round()' to prevent possible malcomputing of the logarithms
    j = 2 ** (np.arange(1, nScales + 1) - 1) - 1
    
    meanDataMeasure = np.zeros(nScales)
    
    ## Computing the data measure
    for ji in range(1, nScales + 1):
        # At the scale 'j(ji)' we deal with '2 * (j(ji) + 1)' elements of the data 'dx'
        dx_k_len = 2 * (j[ji - 1] + 1)
        n = np.int(dx_len / dx_k_len)
        
        dx_leftShift = np.int(dx_k_len / 2)
        dx_rightShift = np.int(dx_k_len / 2)
        
        for k in range(1, n + 1):
            # We get a portion of the data of the length '2 * (j(ji) + 1)' plus the data from the left and right boundaries
            dx_k_withShifts = dx[(k - 1) * dx_k_len + 1 + dx_shift - dx_leftShift - 1 : k * dx_k_len + dx_shift + dx_rightShift]
            
            # Then we perform free upscaling and, using the above-selected data (provided at the scale j = 0),
            # compute the velocities at the scale 'j(ji)'
            j_dx = np.convolve(dx_k_withShifts, np.ones(dx_rightShift), 'valid')
            
            # Then we compute the accelerations at the scale 'j(ji) + 1'
            r = (j_dx[1 + dx_rightShift - 1 : ] - j_dx[1 - 1 : -dx_rightShift]) / 2.0
            
            # Finally, we compute the range ...
            if (normType_p == 0):
                R = np.max(r[2 - 1 : ]) - np.min(r[2 - 1 : ])
            elif (np.isinf(normType_p)):
                R = np.max(np.abs(r[2 - 1 : ]))
            else:
                R = (np.sum(r[2 - 1 : ] ** normType_p) / len(r[2 - 1 : ])) ** (1.0 / normType_p)
            # ... and the normalisation factor ("standard deviation")
            S = np.sqrt(np.sum(np.abs(np.diff(r)) ** 2.0) / (len(r) - 1))
            if (isDFA == 1):
                S = 1.0
            
            meanDataMeasure[ji - 1] += (R / S) ** normType_q
        meanDataMeasure[ji - 1] = (meanDataMeasure[ji - 1] / n) ** (1.0 / normType_q)
    
    # We pass from the scales ('j') to the time measure; the time measure at the scale j(nScales) (the most major one)
    # is assumed to be 2.0, while it is growing when the scale is tending to j(1) (the most minor one).
    # (The scale j(nScales)'s time measure is NOT equal to 1.0, because we reserved the highest scale for shifts
    # in the very beginning of the function.)
    timeMeasure = 2.0 * dx_len / (2 * (j + 1))
    
    scales = j + 1
    
    return [timeMeasure, meanDataMeasure, scales]

##################################################### Função getMSSByUpscaling

def getMSSByUpscaling(dx, normType = np.inf, isDFA = 1, isNormalised = 1):
    ## Some initialiation
    aux_eps = np.finfo(float).eps
    
    # We prepare an array of values of the variable q-norm
    aux = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.0001, 0.0, 0.0001, 0.5, 0.9999, 1.0, 1.0001, 2.0, 4.0, 8.0, 16.0, 32.0]
    nq = len(aux)
    
    q = np.zeros((nq, 1))
    q[:, 1 - 1] = aux
    
    dx_len = len(dx)
    
    # We have to reserve the most major scale for shifts, so we divide the data
    # length by two. (As a result, the time measure starts from 2.0, not from
    # 1.0, see below.)
    dx_len = np.int(dx_len / 2)
    
    dx_shift = np.int(dx_len / 2)
    
    nScales = np.int(np.round(np.log2(dx_len)))    # Number of scales involved. P.S. We use 'round()' to prevent possible malcomputing of the logarithms
    j = 2 ** (np.arange(1, nScales + 1) - 1) - 1
    
    dataMeasure = np.zeros((nq, nScales))
    
    ## Computing the data measures in different q-norms
    for ji in range(1, nScales + 1):
        # At the scale 'j(ji)' we deal with '2 * (j(ji) + 1)' elements of the data 'dx'
        dx_k_len = 2 * (j[ji - 1] + 1)
        n = np.int(dx_len / dx_k_len)
        
        dx_leftShift = np.int(dx_k_len / 2)
        dx_rightShift = np.int(dx_k_len / 2)
        
        R = np.zeros(n)
        S = np.ones(n)
        for k in range(1, n + 1):
            # We get a portion of the data of the length '2 * (j(ji) + 1)' plus the data from the left and right boundaries
            dx_k_withShifts = dx[(k - 1) * dx_k_len + 1 + dx_shift - dx_leftShift - 1 : k * dx_k_len + dx_shift + dx_rightShift]
            
            # Then we perform free upscaling and, using the above-selected data (provided at the scale j = 0),
            # compute the velocities at the scale 'j(ji)'
            j_dx = np.convolve(dx_k_withShifts, np.ones(dx_rightShift), 'valid')
            
            # Then we compute the accelerations at the scale 'j(ji) + 1'
            r = (j_dx[1 + dx_rightShift - 1 : ] - j_dx[1 - 1 : -dx_rightShift]) / 2.0
            
            # Finally we compute the range ...
            if (normType == 0):
                R[k - 1] = np.max(r[2 - 1 : ]) - np.min(r[2 - 1 : ])
            elif (np.isinf(normType)):
                R[k - 1] = np.max(np.abs(r[2 - 1 : ]))
            else:
                R[k - 1] = (np.sum(r[2 - 1 : ] ** normType) / len(r[2 - 1 : ])) ** (1.0 / normType)
            # ... and the normalisation factor ("standard deviation")
            if (isDFA == 0):
                S[k - 1] = np.sqrt(np.sum(np.abs(np.diff(r)) ** 2.0) / (len(r) - 1))
    
        if (isNormalised == 1):      # Then we either normalise the R / S values, treating them as probabilities ...
            p = np.divide(R, S) / np.sum(np.divide(R, S))
        else:                        # ... or leave them unnormalised ...
            p = np.divide(R, S)
            # ... and compute the measures in the q-norms
            for k in range(1, n + 1):
                # This 'if' is needed to prevent measure blow-ups with negative values of 'q' when the probability is close to zero
                if (p[k - 1] < 1000.0 * aux_eps):
                    continue
                
                dataMeasure[:, ji - 1] = dataMeasure[:, ji - 1] + np.power(p[k - 1], q[:, 1 - 1])
    # We pass from the scales ('j') to the time measure; the time measure at the scale j(nScales) (the most major one)
    # is assumed to be 2.0, while it is growing when the scale is tending to j(1) (the most minor one).
    # (The scale j(nScales)'s time measure is NOT equal to 1.0, because we reserved the highest scale for shifts
    # in the very beginning of the function.)
    timeMeasure = 2.0 * dx_len / (2 * (j + 1))
    
    scales = j + 1
    
    ## Determining the exponents 'tau' from 'dataMeasure(q, timeMeasure) ~ timeMeasure ^ tau(q)'
    tau = np.zeros((nq, 1))
    log10tm = np.log10(timeMeasure)
    log10dm = np.log10(dataMeasure)
    log10tm_mean = np.mean(log10tm)
    
    # For each value of the q-norm we compute the mean 'tau' over all the scales
    for qi in range(1, nq + 1):
        tau[qi - 1, 1 - 1] = np.sum(np.multiply(log10tm, (log10dm[qi - 1, :] - np.mean(log10dm[qi - 1, :])))) / np.sum(np.multiply(log10tm, (log10tm - log10tm_mean)))

    ## Finally, we only have to pass from 'tau(q)' to its conjugate function 'f(alpha)'
    # In doing so, first we find the Lipschitz-Holder exponents 'alpha' (represented by the variable 'LH') ...
    aux_top = (tau[2 - 1] - tau[1 - 1]) / (q[2 - 1] - q[1 - 1])
    aux_middle = np.divide(tau[3 - 1 : , 1 - 1] - tau[1 - 1 : -1 - 1, 1 - 1], q[3 - 1 : , 1 - 1] - q[1 - 1 : -1 - 1, 1 - 1])
    aux_bottom = (tau[-1] - tau[-1 - 1]) / (q[-1] - q[-1 - 1])
    LH = np.zeros((nq, 1))
    LH[:, 1 - 1] = -np.concatenate((aux_top, aux_middle, aux_bottom))
    # ... and then compute the conjugate function 'f(alpha)' itself
    f = np.multiply(LH, q) + tau
    ## The last preparations
    # We determine the minimum and maximum values of 'alpha' ...
    LH_min = LH[-1, 1 - 1]
    LH_max = LH[1 - 1, 1 - 1]
    # ... and find the minimum and maximum values of another multifractal characteristic, the so-called
    # generalised Hurst (or DFA) exponent 'h'. (These parameters are computed according to [2, p. 27].)
    h_min = -(1.0 + tau[-1, 1 - 1]) / q[-1, 1 - 1]
    h_max = -(1.0 + tau[1 - 1, 1 - 1]) / q[1 - 1, 1 - 1]
    stats = {'tau':       tau,
        'LH':        LH,
            'f':         f,
                'LH_min':    LH_min,
                    'LH_max':    LH_max,
                        'h_min':     h_min,
                            'h_max':     h_max}
    
    return [timeMeasure, dataMeasure, scales, stats, q]

################################################### Função getScalingExponents
    
def getScalingExponents(timeMeasure, dataMeasure):
    ## Initialisation
    nScales = len(timeMeasure)
    
    log10tm = np.log10(timeMeasure)
    log10dm = np.log10(dataMeasure)
    
    res = 1.0e+07
    bsIndex = nScales
    
    ## Computing
    # We find linear approximations for major and minor subsets of the data measure and determine the index of the
    # boundary scale at which the approximations are optimal in the sense of best fitting to the data measure
    for i in range(3, nScales - 2 + 1):
        # Major 'i' scales are approximated by the function 'k * x + b' ...
        curr_log10tm = log10tm[nScales - i + 1 - 1 : nScales]
        curr_log10dm = log10dm[nScales - i + 1 - 1 : nScales]
        detA = i * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
        detK = i * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
        detB = np.sum(curr_log10dm) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) * np.sum(np.multiply(curr_log10tm, curr_log10dm))
        k = detK / detA
        b = detB / detA
        # ... and the maximum residual is computed
        resMajor = max(np.abs(k * curr_log10tm + b - curr_log10dm))
        
        # Minor 'nScales - i + 1' scales are approximated by the function 'k * x + b' ...
        curr_log10tm = log10tm[1 - 1 : nScales - i + 1]
        curr_log10dm = log10dm[1 - 1 : nScales - i + 1]
        detA = (nScales - i + 1) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
        detK = (nScales - i + 1) * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
        detB = np.sum(curr_log10dm) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) * np.sum(np.multiply(curr_log10tm, curr_log10dm))
        k = detK / detA
        b = detB / detA
        # ... and the maximum residual is computed
        resMinor = max(np.abs(k * curr_log10tm + b - curr_log10dm))
        
        if (resMajor ** 2.0 + resMinor ** 2.0 < res):
            res = resMajor ** 2.0 + resMinor ** 2.0
            bsIndex = i

# Now we determine the boundary scale and the boundary scale's data measure, ...
    bScale = 2.0 * timeMeasure[1 - 1] / timeMeasure[nScales - bsIndex + 1 - 1] / 2.0
    bDM = dataMeasure[nScales - bsIndex + 1 - 1]
    # ... as well as compute the unifractal dimensions using the boundary scale's index:
    # at the major 'bsIndex' scales ...
    curr_log10tm = log10tm[nScales - bsIndex + 1 - 1 : nScales]
    curr_log10dm = log10dm[nScales - bsIndex + 1 - 1 : nScales]
    detA = bsIndex * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
    detK = bsIndex * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
    DMajor = detK / detA
    HMajor = -DMajor
    # ... and at the minor 'nScales - bsIndex + 1' scales
    curr_log10tm = log10tm[1 - 1 : nScales - bsIndex + 1]
    curr_log10dm = log10dm[1 - 1 : nScales - bsIndex + 1]
    detA = (nScales - bsIndex + 1) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
    detK = (nScales - bsIndex + 1) * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
    DMinor = detK / detA
    HMinor = -DMinor
    
    return [bScale, bDM, bsIndex, HMajor, HMinor]

################################ Função que calcula o Modified first-order DFA

def makemfdfa(dx, makegraphs=False):
    [timeMeasure, meanDataMeasure, scales] = getHurstByUpscaling(dx)                    # Set of parameters No. 1
    #[timeMeasure, meanDataMeasure, scales] = getHurstByUpscaling(dx, 3.0, 0, 2.0)       # Set of parameters No. 2
    
    [bScale, bDM, bsIndex, HMajor, HMinor] = getScalingExponents(timeMeasure, meanDataMeasure)
    
    # Modified first-order MF-DFA
    [_, dataMeasure, _, stats, q] = getMSSByUpscaling(dx, isNormalised = 0)
    
    if makegraphs==True:
        ## Output
        # Modified first-order DFA
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.loglog(timeMeasure, meanDataMeasure, 'ko-')
        plt.xlabel(r'$\mu(t)$')
        plt.ylabel(r'$\mu(\Delta x)$')
        plt.grid('on', which = 'minor')
        plt.title('Modified First-Order DFA of a Multifractal Noise')
        
        plt.subplot(2, 1, 2)
        plt.loglog(scales, meanDataMeasure, 'ko-')
        plt.loglog(bScale, bDM, 'ro')
        plt.xlabel(r'$j$')
        plt.ylabel(r'$\mu(\Delta x)$')
        plt.grid('on', which = 'minor')
        
        # Modified first-order MF-DFA
        print('alpha_min = %g, alpha_max = %g, dalpha = %g' % (stats['LH_min'], stats['LH_max'], stats['LH_max'] - stats['LH_min']))
        print('h_min = %g, h_max = %g, dh = %g\n' % (stats['h_min'], stats['h_max'], stats['h_max'] - stats['h_min']))
        
        
        plt.figure()
        nq = np.int(len(q))
        leg_txt = []
        for qi in range(1, nq + 1):
            llh = plt.loglog(scales, dataMeasure[qi - 1, :], 'o-')
            leg_txt.append('tau = %g (q = %g)' % (stats['tau'][qi - 1], q[qi - 1]))
        plt.xlabel(r'$j$')
        plt.ylabel(r'$\mu(\Delta x, q)$')
        plt.grid('on', which = 'minor')
        plt.title('Modified First-Order MF-DFA of a Multifractal Noise')
        plt.legend(leg_txt)
        
        plt.figure()
        
        #plt.subplot(2, 1, 1)
        plt.plot(q, stats['tau'], 'ko-')
        plt.xlabel(r'$q$')
        plt.ylabel(r'$\tau(q)$')
        plt.grid('on', which = 'major')
        plt.title('Statistics of Modified First-Order MF-DFA of a Multifractal Noise')
        
        plt.figure()
        
        #plt.subplot(2, 1, 2)
        plt.plot(stats['LH'], stats['f'], 'ko-')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$f(\alpha)$')
        plt.grid('on', which = 'major')
        plt.show()
    psi= (stats['LH_max']-stats['LH_min'])/stats['LH_max']
    # return stats['LH_min'], stats['LH_max'], stats['LH_max']-stats['LH_min'], psi
    return psi