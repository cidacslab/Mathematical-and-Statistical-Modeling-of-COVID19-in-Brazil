#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:18:44 2020

@author: Rafael Veiga
@author: matheustorquato matheusft@gmail.com
"""
import functools, os
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
import logging
from functools import reduce
import scipy.integrate as spi
from platypus import NSGAII, Problem, Real
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.general_optimizer import GeneralOptimizerPSO
from pyswarms.backend.topology import Star
from pyswarms.utils.plotters import plot_cost_history
from itertools import repeat
import multiprocessing as mp




class SIR_PSO:
    ''' SIR Model'''
    def __init__(self,tamanhoPop,numeroProcessadores=None):
        self.N = tamanhoPop
        self.beta = None
        self.gamma = None
        self.numeroProcessadores = numeroProcessadores
    
    def __cal_EDO(self,x,beta,gamma):
            ND = len(x)-1
            t_start = 0.0
            t_end = ND
            t_inc = 1
            t_range = np.arange(t_start, t_end + t_inc, t_inc)
            beta = np.array(beta)
            gamma = np.array(gamma)
            def SIR_diff_eqs(INP, t, beta, gamma):
                Y = np.zeros((3))
                V = INP
                Y[0] = - beta * V[0] * V[1]                 #S
                Y[1] = beta * V[0] * V[1] - gamma * V[1]    #I
                Y[2] = gamma * V[1]                         #R
                
                return Y
            result_fit = spi.odeint(SIR_diff_eqs, (self.S0, self.I0,self.R0), t_range,
                                    args=(beta, gamma))
            
            S=result_fit[:, 0]*self.N
            R=result_fit[:, 2]*self.N
            I=result_fit[:, 1]*self.N
            
            return S,I,R
    
    def objectiveFunction(self,coef,x ,y):
        tam2 = len(coef[:,0])
        soma = np.zeros(tam2)
        y = y*self.N
        for i in range(tam2):
            S,I,R = self.__cal_EDO(x,coef[i,0],coef[i,1])
            soma[i]= ((y-(I+R))**2).mean()
        return soma
    

    def fit(self, x,y , bound = ([0,1/21-0.0001],[1,1/5+0.0001]), name=None):
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound)
        '''
        self.name=name
        self.y = y
        df = np.array(y)/self.N
        self.I0 = df[0]
        self.S0 = 1-self.I0
        self.R0 = 0
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        if bound==None:
            optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options)
            cost, pos = optimizer.optimize(self.objectiveFunction, 500, x = x,y=df,n_processes=self.numeroProcessadores)
            self.beta = pos[0]
            self.gamma = pos[1]
            self.x = x
            self.rmse = cost
            self.optimize = optimizer
            
        else:
            optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=bound)
            cost, pos = optimizer.optimize(self.objectiveFunction, 500, x = x,y=df,n_processes=self.numeroProcessadores)
            self.beta = pos[0]
            self.gamma = pos[1]
            self.x = x
            self.rmse = cost
            self.optimize = optimizer
            
            
    def predict(self,x):
        ''' x = dias passados do dia inicial 1'''
        S,I,R = self.__cal_EDO(x,self.beta,self.gamma)
        self.ypred = I+R
        self.S = S
        self.I = I
        self.R = R         
        return I+R
    def getResiduosQuadatico(self):
        y = np.array(self.y)
        ypred = np.array(self.ypred)
        y = y[0:len(self.x)]
        ypred = ypred[0:len(self.x)]
        return (y - ypred)**2
    def getReQuadPadronizado(self):
        y = np.array(self.y)
        ypred = np.array(self.ypred)
        y = y[0:len(self.x)]
        ypred = ypred[0:len(self.x)]
        res = ((y - ypred)**2)/y
        return res 
    
    def plotCost(self):
        plot_cost_history(cost_history=self.optimize.cost_history)
        plt.show()
    def plot(self,local):
        ypred = self.predict(self.x)
        plt.plot(ypred,c='b',label='Predição Infectados')
        plt.plot(self.y,c='r',marker='o', markersize=3,label='Infectados')
        plt.legend(fontsize=15)
        plt.title('Dinâmica do CoviD19 - {}'.format(local),fontsize=20)
        plt.ylabel('Casos COnfirmados',fontsize=15)
        plt.xlabel('Dias',fontsize=15)
        plt.show()
    def getCoef(self):
        return ['beta','gamma',['suscetivel','infectados','recuperados','casos']], [self.beta,self.gamma,self.y]