#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:08:01 2020

@author: Felipe A. C. Pereira

Implementação do ajuste do modelo SEIIHURD com separação de grupos. Necessita
de mais verificações e funções para simplificar o input. Baseado nas classes
disponíveis no modelos.py
"""

import numpy as np
from functools import reduce
import scipy.integrate as spi
from scipy.optimize import least_squares
from platypus import NSGAII, Problem, Real
from pyswarms.single.global_best import GlobalBestPSO
import pyswarms as ps
from pyswarms.backend.topology import Star
from pyswarms.utils.plotters import plot_cost_history
from itertools import repeat
import multiprocessing as mp
import copy


class SEIIHURD_age:
    ''' SEIIHURD Model'''
    def __init__(self,tamanhoPop,numeroProcessadores=None):
        self.N = tamanhoPop
        self.numeroProcessadores = numeroProcessadores

#pars dict betas, delta, kappa, p, gammaA, gammaS, h, epsilon, gammaH, gammaU, muU, muH, wU, wH
# seguindo a notação beta_12 é 2 infectando 1, onde 1 é a linha e 2 a coluna.
    def _SEIIHURD_age_eq(self, X, t, pars):
        S, E, Ia, Is, H, U, R, D, Nw = np.split(X, 9)
        StE = S * (pars['beta'] @  ((Ia * pars['delta'] + Is).reshape((-1,1)))).flatten()
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
        Is0 = pars['x0'].reshape((3,-1)).sum(axis=0)
        x0 = np.r_[1. - Is0, pars['x0'], np.zeros(4*len(Is0)), pars['x0'][2*len(Is0):]]
        saida = x0.reshape((1,-1))
        Y = saida.copy()
        for i in range(1, len(tcorte)):
            cut_last = False
            try:
                pars['beta'] = betas[i-1]
                t = ts[(ts >= tcorte[i-1]) * (ts<= tcorte[i])]
                if t[0] > tcorte[i-1]:
                    t = np.r_[tcorte[i-1], t]
                if t[-1] < tcorte[i]:
                    t = np.r_[t, tcorte[i]]
                    cut_last = True
                Y = spi.odeint(self._SEIIHURD_age_eq, Y[-1], t, args=(pars,))
            except:
                print(pars, i, betas, tcorte)
                raise
            if cut_last:
                saida = np.r_[saida, Y[1:-1]]
            else:
                saida = np.r_[saida, Y[1:]]
        return ts, saida


    def _fill_paramPSO(self, paramPSO):
        if 'options' not in paramPSO.keys():
            paramPSO['options'] = {'c1': 0.1, 'c2': 0.3, 'w': 0.9,'k':5,'p':2}
        if 'n_particles' not in paramPSO.keys():
            paramPSO['n_particles'] = 300
        if 'iter' not in paramPSO.keys():
            paramPSO['iter'] = 1000
        return paramPSO
    
    
    def _prepare_input(self, data):
        list_states = ['S', 'E', 'Ia', 'Is', 'H', 'U', 'R', 'D', 'Nw']
        i_integ = list()
        Y = list()
        for ke in data.keys():
            if ke == 't':
                t = data[ke]
            else:
                Y.append(data[ke])
                simb, num = ke.split("_")
                n0 = self.nages * list_states.index(simb)
                if '_ALL' in ke:
                    i_integ.append(list(range(n0,n0 + self.nages)))
                else:
                    i_integ.append(int(num) + n0)
        return i_integ, Y, t
                
                
    def _prepare_conversor(self, p2f, pothers, bound):
        padjus = list()
        if  bound != None:
            bound_new = [[], []]
        for i, par in enumerate(p2f):
            if 'beta' in par:
                if '_ALL' in par:
                    for i in range(len(pothers['beta'])):
                        for j in range(pothers['beta'][i].shape[0]):
                            for k in range(pothers['beta'][i].shape[1]):
                                padjus.append('beta_{}_{}_{}'.format(i,j,k))
                                if  bound != None:
                                    bound_new[0].append(bound[0][i])
                                    bound_new[1].append(bound[1][i])
                else:
                    padjus.append(par)
                    if  bound != None:
                        bound_new[0].append(bound[0][i])
                        bound_new[1].append(bound[1][i])
                    
            elif '_ALL' in par:
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
            if 'beta' in padjus[i]:
                if '_M_' in padjus[i]:
                    indx = int(padjus[i].split('_')[-1])
                    pars['beta'][indx] = coef * pars['beta'][indx]
                else:
                    indx = padjus[i].split('_')
                    pars['beta'][int(indx[1])][int(indx[2]), int(indx[3])] = coef
            elif '_' in padjus[i]:
                name, indx = padjus[i].split('_')
                pars[name][int(indx)] = coef                    
            else:
                pars[padjus[i]] = coef
        return pars
        
    
    def objectiveFunction(self, coefs_list, stand_error, weights=None):
        if type(weights) == type(None):
            weights = np.ones(len(self.Y))
        error_func = (lambda x: np.sqrt(x+1)) if stand_error else (lambda x:np.ones_like(x))
        errsq = np.zeros(coefs_list.shape[0])
        for i, coefs in enumerate(coefs_list):
            ts, mY = self._call_ODE(self.t, self._conversor(coefs, self.pars_init, self.padjus))
            for indY, indODE in enumerate(self.i_integ):
                if type(indODE) == list:
                    temp = (self.N.reshape((1,-1)) *  mY[:,indODE]).sum(axis=1)
                    errsq[i] += weights[indY] * (((self.Y[indY] - temp) / error_func(temp))**2 ).mean()
                else:
                    errsq[i] += weights[indY] * (((self.Y[indY] - self.N[indODE%self.nages] * mY[:,indODE]) / error_func(mY[:,indODE]))**2 ).mean()
        return errsq

    def _residuals(self, coefs, stand_error=False, weights=None):
        if type(weights) == type(None):
            weights = np.ones(len(self.Y))
        error_func = (lambda x: np.sqrt(x+1)) if stand_error else (lambda x:np.ones_like(x))
        errs = np.empty((0,))
        ts, mY = self._call_ODE(self.t, self._conversor(coefs, self.pars_init, self.padjus))
        for indY, indODE in enumerate(self.i_integ):
            if type(indODE) == list:
                temp = (self.N.reshape((1,-1)) *  mY[:,indODE]).sum(axis=1)
                errs = np.r_[errs, weights[indY] * ((self.Y[indY] - temp) / error_func(temp)) ]
            else:
                errs = np.r_[errs, weights[indY] * ((self.Y[indY] - self.N[indODE%self.nages] *  mY[:,indODE]) / error_func(mY[:,indODE])) ]
        return errs
        
    
    def fit(self, data, pars, pars_to_fit, bound=None, nages=2, paramPSO=dict(),  stand_error=False):
        '''
        data: dictionary:
            t -> times
            X_N -> variable:
                X is the simbol of the parameter: S, E, Ia, Is, H, U, R, D, Nw
                N is the index of the age-group, starting on 0
        
        pars: dictionary, with the variable names as keys. 
        
        pars_to_fit: the name of the parameters to fits, if the parameter is a list,
        add _N with the index you want to if or _ALL to fit all
        the 'beta' parameter has 3 indexes: beta_I_J_K, with I indicating the
        which tcut it belongs and J_K indicating the position in the matrix.
        the beta also has a option 'beta_M_I' that fits a multiplicative
        constant of the infection matrix, without changing the relative weights
        (the _M_ and _ALL_ options are incompatible by now, and _M_ requires
        testing)
        
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound)
        '''
        paramPSO = self._fill_paramPSO(paramPSO)
        self.pars_init = copy.deepcopy(pars)
        self.nages = nages
        self.i_integ, self.Y, self.t = self._prepare_input(data)
        self.bound, self.padjus = self._prepare_conversor(pars_to_fit, pars, bound)
        self.n_to_fit = len(self.padjus)
        optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=self.n_to_fit, options=paramPSO['options'],bounds=self.bound)
        cost = pos = None
        cost, pos = optimizer.optimize(self.objectiveFunction,paramPSO['iter'],  stand_error=stand_error, n_processes=self.numeroProcessadores)
        self.pos = pos
        self.pars_opt = self._conversor(pos, self.pars_init, self.padjus )
        self.rmse = cost
        self.optimize = optimizer

    def fit_lsquares(self, data, pars, pars_to_fit, bound=None, nages=2,  stand_error=False, init=None, nrand=10):
        self.pars_init = copy.deepcopy(pars)
        self.nages = nages
        self.i_integ, self.Y, self.t = self._prepare_input(data)
        self.bound, self.padjus = self._prepare_conversor(pars_to_fit, pars, bound)
        self.n_to_fit = len(self.padjus)
        if init == None:
            cost_best = np.inf
            res_best = None
            for i in range(nrand):
                print("{} / {}".format(i, nrand))
                par0 = np.random.rand(self.n_to_fit)
                par0 = self.bound[0] + par0 * (self.bound[1] - self.bound[0])
                res = least_squares(self._residuals, par0, bounds=self.bound)
                if res.cost < cost_best:
                    cost_best = res.cost
                    res_best = res
        else:
            res_best = least_squares(self._residuals, init, bounds=bound )
        self.pos_ls = res_best.x
        self.pars_opt_ls = self._conversor(res_best.x, self.pars_init, self.padjus )
        self.rmse_ls = (res_best.fun**2).mean()
        self.result_ls = res_best
        
    def predict(self, t=None, coefs=None):
        if type(t) == type(None):
            t = self.t
        if type(coefs) == type(None):
            coefs = self.pos
        elif type(coefs) == str and coefs  == 'LS':
            coefs = self.pos_ls
        ts, mY = self._call_ODE(self.t, self._conversor(coefs, self.pars_init, self.padjus))
        saida = np.zeros((len(ts), 0))
        for i in self.i_integ:
            if type(i) == list:
                ytemp = (mY[:,i] *self.N.reshape((1,-1))).sum(axis=1)
            else:
                ytemp = mY[:,i] * self.N[i%self.nages]
            saida = np.c_[saida, ytemp.reshape((-1,1))]
        return ts, saida
    
  

#ts, X = call_ODE(X0, tmax, betas, param, tcorte=tcorte)
#plt.plot(ts, X[:,:2], '.-')
