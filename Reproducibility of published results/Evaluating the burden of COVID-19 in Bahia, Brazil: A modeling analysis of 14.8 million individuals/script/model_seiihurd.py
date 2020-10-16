#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:08:01 2020

SEIIHURD fitting class and bootstrap auxiliary functions.
"""

import numpy as np
from functools import reduce
import scipy.integrate as spi
from platypus import NSGAII, Problem, Real
from pyswarms.single.global_best import GlobalBestPSO
import pyswarms as ps
from pyswarms.backend.topology import Star
from itertools import repeat
import multiprocessing as mp
import copy
from scipy.stats import poisson


def gen_bootstrap_serie(serie, iscumul=False):
    if iscumul:
        serie = np.r_[serie[0], np.diff(serie)]
    boots = poisson.rvs(mu=serie)
    if iscumul:
        boots = np.cumsum(boots)
    return boots

def gen_bootstrap_full(data, newcases):
    saida = copy.deepcopy(data)
    for key in saida.keys():
        if key in ['D', 'Nw'] and not newcases:
            saida[key] = gen_bootstrap_serie(saida[key], True)
        elif key != 't':
            saida[key] = gen_bootstrap_serie(saida[key])
    return saida

class SEIIHURD:
    ''' SEIIHURD Model'''
    def __init__(self,tamanhoPop,numeroProcessadores=None, usenew=False):
        self.N = tamanhoPop
        self.numeroProcessadores = numeroProcessadores
        self.pos = None
        self.usenew = usenew

#pars dict betas, delta, kappa, p, gammaA, gammaS, h, epsilon, gammaH, gammaU, muU, muH, wU, wH
# seguindo a notação beta_12 é 2 infectando 1, onde 1 é a linha e 2 a coluna.
    def _SEIIHURD_age_eq(self, X, t, pars):
        S, E, Ia, Is, H, U, R, D, Nw = X
        StE = S * pars['beta'] *  (Ia * pars['delta'] + Is)
        dS = - StE
        dE = StE - pars['kappa'] * E
        dIa = (1. - pars['p']) * pars['kappa'] * E - pars['gammaA'] * Ia
        dIs = pars['p'] * pars['kappa'] * E - pars['gammaS'] * Is
        dH = pars['h'] * pars['xi'] * pars['gammaS'] * Is + (1 - pars['muU'] +\
            pars['wU'] * pars['muU']) * pars['gammaU'] * U - pars['gammaH'] * H    
        dU = pars['h'] * (1 - pars['xi']) * pars['gammaS'] * Is + pars['wH'] *\
            pars['gammaH'] * H - pars['gammaU'] * U
        dR = pars['gammaA'] * Ia + (1. - pars['h']) * pars['gammaS'] * Is + \
            (1 - pars['muH']) * (1 - pars['wH']) * pars['gammaH'] * H
        dD = (1 - pars['wH']) * pars['muH'] * pars['gammaH'] * H + \
            (1 - pars['wU']) * pars['muU'] * pars['gammaU'] * U
        dNw = pars['p'] * pars['kappa'] * E 
        return np.r_[dS, dE, dIa, dIs, dH, dU, dR, dD, dNw]
    
    
    def _call_ODE(self, ts, ppars):
        betas = ppars['beta'].copy()
        pars = copy.deepcopy(ppars)
        if 'tcut' not in ppars.keys():
            tcorte = None
        else:
            tcorte = pars['tcut']
        if type(ts) in [int, float]:
            ts = np.arange(ts)
        if tcorte == None:
            tcorte = [ts[-1]]
            if type(betas) != list:
                betas = [betas]
        if tcorte[-1] < ts[-1]:
            tcorte.append(ts[-1])
        tcorte = [ts[0]] + tcorte
        tcorte.sort()
        Is0 = pars['x0'].reshape((3,-1)).sum(axis=0)
        x0 = np.r_[1. - Is0, pars['x0'], np.zeros(4*len(Is0)), pars['x0'][2*len(Is0):]]
        saida = x0.reshape((1,-1))
        Y = saida.copy()
        for i in range(1, len(tcorte)):
            cut_last = False
            pars['beta'] = betas[i-1]
            t = ts[(ts >= tcorte[i-1]) * (ts<= tcorte[i])]
            if len(t) > 0:
                if t[0] > tcorte[i-1]:
                    t = np.r_[tcorte[i-1], t]
                if t[-1] < tcorte[i]:
                    t = np.r_[t, tcorte[i]]
                    cut_last = True
                Y = spi.odeint(self._SEIIHURD_age_eq, Y[-1], t, args=(pars,))
                if cut_last:
                    saida = np.r_[saida, Y[1:-1]]
                else:
                    saida = np.r_[saida, Y[1:]]
            else:
                Y = spi.odeint(self._SEIIHURD_age_eq, Y[-1], tcorte[i-1:i+1], args=(pars,))
            
        return ts, saida


    def _fill_paramPSO(self, paramPSO):
        if 'options' not in paramPSO.keys():
            paramPSO['options'] = {'c1': 0.3, 'c2': 0.3, 'w': 0.9,'k':5,'p':2}
        if 'n_particles' not in paramPSO.keys():
            paramPSO['n_particles'] = 300
        if 'iter' not in paramPSO.keys():
            paramPSO['iter'] = 1000
        return paramPSO
    
    
    def _prepare_input(self, data):
        list_states = ['S', 'E', 'Ia', 'Is', 'H', 'U', 'R', 'D', 'Nw']
        i_integ = list()
        Y = list()
        newsim = list()
        for ke in data.keys():
            if ke == 't':
                t = data[ke]
            else:
                Y.append(data[ke])
                n0 = list_states.index(ke)
                i_integ.append(n0)
                if (ke in ['D', 'Nw']) and self.usenew:
                    newsim.append(True)
                else:
                    newsim.append(False)
        return i_integ, Y, t, newsim
                
                
    def _prepare_conversor(self, p2f, pothers, bound):
        padjus = list()
        if  bound != None:
            bound_new = [[], []]
        for i, par in enumerate(p2f):
            if '_ALL' in par:
                name = par.split('_')[0]
                for j in range(len(pothers[name])):
                    padjus.append('{}_{}'.format(name, j))
                    if  bound != None:
                        bound_new[0].append(bound[0][i])
                        bound_new[1].append(bound[1][i])
            else:
                padjus.append(par)
                if  bound != None:
                    bound_new[0].append(bound[0][i])
                    bound_new[1].append(bound[1][i])
        if  bound != None:
            bound_new[0] = np.array(bound_new[0])
            bound_new[1] = np.array(bound_new[1])
        return bound_new, padjus
    
    def _conversor(self, coefs, pars0, padjus):
        pars = copy.deepcopy(pars0)
        for i, coef in enumerate(coefs):
            if '_' in padjus[i]:
                name, indx = padjus[i].split('_')
                pars[name][int(indx)] = coef                  
            else:
                pars[padjus[i]] = coef
        return pars
        
    
    def objectiveFunction(self, coefs_list, stand_error=False, weights=None):
        errsq = np.zeros(coefs_list.shape[0])
        for i, coefs in enumerate(coefs_list):
            errs = self._residuals(coefs, stand_error, weights)
            errsq[i] = (errs*errs).mean()
        return errsq

    def _residuals(self, coefs, stand_error=False, weights=None):
        if type(weights) == type(None):
            weights = np.ones(len(self.Y))
        error_func = (lambda x: np.sqrt(x+1)) if stand_error else (lambda x:np.ones_like(x))
        errs = np.empty((0,))
        ts, mY = self._call_ODE(self.t, self._conversor(coefs, self.pars_init, self.padjus))
        for indY, indODE in enumerate(self.i_integ):                   
            temp =  self.N *  mY[:,indODE]
            if self.newsim[indY]:
                temp = np.r_[temp[0], np.diff(temp)]
            errs = np.r_[errs, weights[indY] * ((self.Y[indY] - temp) / error_func(temp)) ]
        errs = errs[~np.isnan(errs)]
        return errs
        
    def prepare_to_fit(self, data, pars, pars_to_fit, bound=None, stand_error=False):
        self.pars_init = copy.deepcopy(pars)
        self.i_integ, self.Y, self.t, self.newsim = self._prepare_input(data)
        self.bound, self.padjus = self._prepare_conversor(pars_to_fit, pars, bound)
        self.n_to_fit = len(self.padjus)
        
    
    def fit(self, data, pars, pars_to_fit, bound=None,  paramPSO=dict(),  stand_error=False):
        '''
        data: dictionary:
            t -> times
            S, E, Ia, Is, H, U, R, D, Nw -> variable
        
        pars: dictionary, with the parameter names as keys. 
        
        pars_to_fit: the name of the parameters to fits, if the parameter is a array,
        add _N with the index you want to fit or _ALL to fit all array
        the 'beta' parameter has index: beta_I, with I indicating the
        which tcut it belongs 
        
        bound => (min_bound_list, max_bound_list)
        '''
        paramPSO = self._fill_paramPSO(paramPSO)
        self.prepare_to_fit(data, pars, pars_to_fit, bound=bound, stand_error=stand_error)
        optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=self.n_to_fit, options=paramPSO['options'],bounds=self.bound)
        cost = pos = None
        cost, pos = optimizer.optimize(self.objectiveFunction,paramPSO['iter'],  stand_error=stand_error, n_processes=self.numeroProcessadores, verbose = True) #add verbose option to no print pb every run
        self.pos = pos
        self.pars_opt = self._conversor(pos, self.pars_init, self.padjus )
        self.rmse = cost
        self.optimize = optimizer

        
    def predict(self, t=None, coefs=None, model_output=False):
        if type(t) == type(None):
            t = self.t
        if type(coefs) == type(None):
            coefs = self.pos
        ts, mY = self._call_ODE(t, self._conversor(coefs, self.pars_init, self.padjus))
        saida = np.zeros((len(ts), 0))
        for u, i in enumerate(self.i_integ):
            temp = mY[:,i] * self.N
            if self.newsim[u]:
                temp = np.r_[temp[0], np.diff(temp)]
            saida = np.c_[saida, temp.reshape((-1,1))]
        if model_output:
            return ts, saida, mY
        else:
            return ts, saida
    