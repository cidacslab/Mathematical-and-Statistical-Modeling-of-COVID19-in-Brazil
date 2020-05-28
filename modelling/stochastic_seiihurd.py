#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:53:05 2020

@author: Felipe A. C. Pereira

A stochastic version of the age group separated SEIIHURD model.
It is a good test for the fitting functions, as a experimental-like series can
be generated with known parameters.
The fitted parameters of the ODE version is not expected to converge to the 
stochastic ones when the populations or initial infected are too small.

TODO: create a version with input of all parameters to generate a change
of betas, like the ODE fit feature.
"""

import numpy as np


def SEIIHURD_agestoc(N, I0, pars, dt=1, tf=np.inf, ti=None):
    '''
    Simulate one realization of the SEIIHURD model with age group separation
    Parameters
    ----------
    N: 1d-array
        Population of each age group.

    I0: 1d-array
        the full initial condition if len(I0) == 1 + 9 * len(N) or only number of exposed, asymptomatic and symptomatic infected, for each
        population if len(I0) == len(N)

    pars: dict
        Dictionary with all the model parameters, with their names as
        keys, with exception of beta, each parameter may be a scalar
        ou a 1d-array with the same size of N.
        
    dt: float
        Time step of the return. Default 1 day
    
    tf: float
        total time of simulation. If equals to inf, the simulation stops
        when there is no more possible transition.
        
    ti: float
        first output instant (besides t0)

    Returns
    -------

    saida: 2d-array
        The system state with roughly the time step asked. Order:
        t, S, E, Ia, Is, H, U, R, D, NI.
    
    
    Notes
    -----
    
    Example of pars, for two age-groups:
    param = {'delta': 0.62,
    'kappa': 0.25,
    'gammaA': 1./3.5,
    'gammaS': 0.25,
    'h': 0.28,
    'xi': 0.53,
    'gammaH': 0.14,
    'gammaU': 0.14,
    'muH': 0.15,
    'muU': 0.35,
    'wH': 0.14,
    'wU': 0.29,
    'p': .2,
    'beta': 1.06 * np.ones((2,2))    }
    '''
    N = N.astype(int)
    if len(I0) == len(N):
        I0 = I0.astype(int)
        zer = np.zeros_like(I0)
        t = 0
        S, E, Ia, Is, H, U, R, D = ((N - 3*I0), I0.copy(), I0.copy(), I0.copy(), \
                                    zer.copy(), zer.copy(), zer.copy(), zer.copy()) 
        NI = Is + H + U
    elif len(I0) == 1 + 9*len(N):
        t = I0[0]
        S, E, Ia, Is, H, U, R, D, NI = np.split(I0[1:].astype(int), 9)
    else:
        raise ValueError("the I0 must be of the same shape of N or 9 times plus 1 of it")
    nag = len(N)
    if ti == None or ti < t:
        ta = t + dt
    else:
        ta = ti
#    n0 = np.sum(N)
    A = np.sum(3*I0)
    saida = list()
    saida.append(np.r_[t, S, E, Ia, Is, H, U, R, D, NI])
    PM = np.empty((nag*6))
    while A > 0 and t < tf:
        rands = np.random.rand(2)
        Ic = Ia * pars['delta'] + Is
        #Ps 
        PM[:nag] =  S * (pars['beta']  @ (Ic/N).reshape((-1,1))).flatten()
        PM[nag:2*nag] = pars['kappa'] * E
        PM[2*nag:3*nag] = pars['gammaA'] * Ia
        PM[3*nag:4*nag] = pars['gammaS'] * Is
        PM[4*nag:5*nag] = pars['gammaH'] * H
        PM[5*nag:] = pars['gammaU'] * U
        Wc = PM.cumsum()
        W = Wc[-1]
        Wc = Wc / W
#        print(PM)
#        print(A - np.sum(E+Ia+Is+H+U))
        t -= np.log(rands[0]) / W
        i = 0
        if rands[1] < Wc[nag-1]:
            #S -> E
#            print("S")
            while rands[1] > Wc[i]:
                i = i + 1
#            print("S", i, S, E, Ia, Is, H, U)
            S[i] = S[i] - 1
            E[i] = E[i] + 1
            A = A + 1
#            print("S", i, S, E, Ia, Is, H, U)
        elif rands[1] < Wc[2*nag-1]:
#            print("E")
            # E- > ?
            while rands[1] > Wc[i+nag]:
                i = i + 1
#            print("E", i)
            E[i] = E[i] - 1
            temp = np.random.rand()
            if temp < pars['p']:
#                print("E-Is", i)
                Is[i] = Is[i] + 1
                NI[i] = NI[i] + 1
            else:
#                print("E-Ia", i)
                Ia[i] = Ia[i] + 1
        elif rands[1] < Wc[3*nag-1]:
#            print("Ia")
            #Ia -> R
            while rands[1] > Wc[i+2*nag]:
                i = i + 1
            Ia[i] = Ia[i] - 1
            R[i] = R[i] + 1
            A = A - 1
        elif rands[1] < Wc[4*nag-1]:
#            print("Is")
            #Is -> ?
            while rands[1] > Wc[i+3*nag]:
                i = i + 1
            Is[i] = Is[i] - 1
            temp = np.random.rand()
            if temp >= pars['h']:
                R[i] = R[i] + 1
                A = A - 1
            elif temp < pars['h'] * pars['xi']:
                U[i] = U[i] + 1
            else:
                H[i] = H[i] +  1        
        elif rands[1] < Wc[5*nag-1]:
#            print("H")
            while rands[1] > Wc[i+4*nag]:
                i = i + 1
            H[i] = H[i] - 1
            temp = np.random.rand()
            if temp < pars['wH']:
                U[i] = U[i] + 1
            elif temp < pars['wH'] + (1. - pars['wH']) * pars['muH']:
                D[i] = D[i] + 1
                A = A - 1
            else:
                R[i] = R[i] + 1
                A = A - 1
        else:
#            print("U")
            while rands[1] > Wc[i+5*nag]:
                i = i + 1
            U[i] = U[i] - 1
            temp = np.random.rand()
            if temp < (1.- pars['wU']) *pars['muU']:
                D[i] = D[i] + 1
                A = A - 1
            else:
                H[i] = H[i] + 1
        if t > ta:
            saida.append(np.r_[t, S, E, Ia, Is, H, U, R, D, NI])
            ta += dt
    saida.append(np.r_[t, S, E, Ia, Is, H, U, R, D, NI])
    return np.array(saida)


def call_stoch_SEIIHURD(N, tmax, dt, params):
    pars = params.copy()
    betas = params['beta']
    if 'tcut' not in params.keys():
        tcorte = None
    else:
        tcorte = params['tcut']
    ts = np.arange(0,tmax,dt)
    if tcorte == None:
        tcorte = [ts[-1]]
        if type(betas) != list:
            betas = [betas]
    if type(tcorte) in [int, float]:
        tcorte = [tcorte]
    if tcorte[-1] < ts[-1]:
        tcorte.append(ts[-1])
    tcorte = [ts[0]] + tcorte
    x0 = pars['x0']
    if len(x0) > len(N):
        saida = np.r_[ts[0], x0].reshape((1,-1))
    else:
        zer = np.zeros_like(N)
        saida = np.r_[ts[0], N-3*x0, x0, x0, x0, zer, zer, zer, zer, x0].reshape((1,-1))
    Y = saida.copy()
    for i in range(1, len(tcorte)):
        cut_last = False
        pars['beta'] = betas[i-1]
        t = ts[(ts >= tcorte[i-1]) * (ts<= tcorte[i])]
        if t[0] > tcorte[i-1]:
            t = np.r_[tcorte[i-1], t]
        if t[-1] < tcorte[i]:
            t = np.r_[t, tcorte[i]]
            cut_last = True
        Y = SEIIHURD_agestoc(N, Y[-1], pars, dt=dt, tf=t[-1], ti=t[0])
        if cut_last:
            saida = np.r_[saida, Y[1:-1]]
        else:
            saida = np.r_[saida, Y[1:]]
    return saida
    