#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:18:44 2020

@author: Rafael Veiga rafaelvalenteveiga@gmail.com
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
#from pyswarms.single.global_best import GlobalBestPSO
import pyswarms as ps
from pyswarms.backend.topology import Star
from pyswarms.utils.plotters import plot_cost_history
from itertools import repeat
import multiprocessing as mp

class SEIRHUD:
    ''' SEIRHU Model'''
    def __init__(self,tamanhoPop,numeroProcessadores=None):
        self.N = tamanhoPop
        self.numeroProcessadores = numeroProcessadores
    
    def __cal_EDO(self,x,beta,gammaH,gammaU,delta,h,ia0,is0,e0):
            ND = len(x)-1
            t_start = 0.0
            t_end = ND
            t_inc = 1
            t_range = np.arange(t_start, t_end + t_inc, t_inc)
            beta = np.array(beta)
            delta = np.array(delta)
            def SIR_diff_eqs(INP, t, beta,gammaH,gammaU, delta,h):
                Y = np.zeros((9))
                V = INP
                Y[0] = - beta*V[0]*(V[3] + delta*V[2])                    #S
                Y[1] = beta*V[0]*(V[3] + delta*V[2]) -self.kappa * V[1]
                Y[2] = (1-self.p)*self.kappa*V[1] - self.gammaA*V[2]
                Y[3] = self.p*self.kappa*V[1] - self.gammaS*V[3]
                Y[4] = h*self.xi*self.gammaS*V[3] + (1-self.muU + self.omegaU*self.muU)*gammaU*V[5] -gammaH*V[4]
                Y[5] = h*(1-self.xi)*self.gammaS*V[3] +self.omegaH*gammaH*V[4] -gammaU*V[5]
                Y[6] = self.gammaA*V[2] + (1-(self.muH))*(1-self.omegaH)*gammaH*V[4] + (1-h)*self.gammaS*V[3]
                Y[7] = (1-self.omegaH)*self.muH*gammaH*V[4] + (1-self.omegaU)*self.muU*gammaU*V[5]#R
                Y[8] = self.p*self.kappa*V[1] 
                return Y
            result_fit = spi.odeint(SIR_diff_eqs, (1-ia0-is0-e0,e0 ,ia0,is0,0,0,0,0,0), t_range,
                                    args=(beta,gammaH,gammaU, delta,h))
            
            S=result_fit[:, 0]*self.N
            E = result_fit[:, 1]*self.N
            IA=result_fit[:, 2]*self.N
            IS=result_fit[:, 3]*self.N
            H=result_fit[:, 4]*self.N
            U=result_fit[:, 5]*self.N
            R=result_fit[:, 6]*self.N
            D=result_fit[:, 7]*self.N
            Nw=result_fit[:, 8]*self.N
            
            return S,E,IA,IS,H,U,R,D,Nw
        
    def __cal_EDO_2(self,x,beta1,beta2,tempo,gammaH,gammaU,delta,h,ia0,is0,e0):
            ND = len(x)-1
            
            t_start = 0.0
            t_end = ND
            t_inc = 1
            t_range = np.arange(t_start, t_end + t_inc, t_inc)
            def Hf(t):
                h = 1.0/(1.0+ np.exp(-2.0*50*t))
                return h
            def beta(t,t1,b,b1):
                beta = b*Hf(t1-t) + b1*Hf(t-t1) 
                return beta

            delta = np.array(delta)
            def SIR_diff_eqs(INP, t, beta1, beta2,t1,gammaH,gammaU, delta,h):
                #Y[0] = - beta(t,t1,beta1,beta2) * V[0] * V[1]                 #S
                Y = np.zeros((9))
                V = INP
                Y[0] = - beta(t,t1,beta1,beta2)*V[0]*(V[3] + delta*V[2])                    #S
                Y[1] = beta(t,t1,beta1,beta2)*V[0]*(V[3] + delta*V[2]) -self.kappa * V[1]
                Y[2] = (1-self.p)*self.kappa*V[1] - self.gammaA*V[2]
                Y[3] = self.p*self.kappa*V[1] - self.gammaS*V[3]
                Y[4] = h*self.xi*self.gammaS*V[3] + (1-self.muU + self.omegaU*self.muU)*gammaU*V[5] -gammaH*V[4]
                Y[5] = h*(1-self.xi)*self.gammaS*V[3] +self.omegaH*gammaH*V[4] -gammaU*V[5]
                Y[6] = self.gammaA*V[2] + (1-(self.muH))*(1-self.omegaH)*gammaH*V[4] + (1-h)*self.gammaS*V[3]
                Y[7] = (1-self.omegaH)*self.muH*gammaH*V[4] + (1-self.omegaU)*self.muU*gammaU*V[5]#R
                Y[8] = self.p*self.kappa*V[1]                      #R
                
                return Y
            result_fit = spi.odeint(SIR_diff_eqs, (1-ia0-is0-e0,e0 ,ia0,is0,0,0,0,0,0), t_range,
                                    args=(beta1,beta2,tempo,gammaH,gammaU, delta,h))
            
            S=result_fit[:, 0]*self.N
            E = result_fit[:, 1]*self.N
            IA=result_fit[:, 2]*self.N
            IS=result_fit[:, 3]*self.N
            H=result_fit[:, 4]*self.N
            U=result_fit[:, 5]*self.N
            R=result_fit[:, 6]*self.N
            D=result_fit[:, 7]*self.N
            Nw=result_fit[:, 8]*self.N
            
            return S,E,IA,IS,H,U,R,D,Nw
    
    def objectiveFunction(self,coef,x ,y,d,stand_error):
        tam2 = len(coef[:,0])
        soma = np.zeros(tam2)
        if stand_error:
            if (self.beta_variavel) & (self.day_mudar==None):
                for i in range(tam2):
                    S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3],coef[i,4],coef[i,5],coef[i,6],coef[i,7],coef[i,8],coef[i,9])
                    soma[i]= (((y-(Nw))/np.sqrt(Nw+1))**2).mean()*(1-self.pesoMorte)+(((d-(D))/np.sqrt(D+1))**2).mean()*self.pesoMorte
            elif self.beta_variavel:
                for i in range(tam2):
                    S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],self.day_mudar,coef[i,3],coef[i,4],coef[i,5],coef[i,6],coef[i,7],coef[i,8])
                    soma[i]= (((y-(Nw))/np.sqrt(Nw+1))**2).mean()*(1-self.pesoMorte)+(((d-(D))/np.sqrt(D+1))**2).mean()*self.pesoMorte
            else:
                for i in range(tam2):
                    S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3],coef[i,4],coef[i,5],coef[i,6],coef[i,7])
                    soma[i]= (((y-(Nw))/np.sqrt(Nw+1))**2).mean()*(1-self.pesoMorte)+(((d-(D))/np.sqrt(D+1))**2).mean()*self.pesoMorte
        else:
            if (self.beta_variavel) & (self.day_mudar==None):
                for i in range(tam2):
                    S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3],coef[i,4],coef[i,5],coef[i,6],coef[i,7],coef[i,8],coef[i,9])
                    soma[i]= ((y-(Nw))**2).mean()*(1-self.pesoMorte)+((d-(D))**2).mean()*self.pesoMorte
            elif self.beta_variavel:
                for i in range(tam2):
                    S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],self.day_mudar,coef[i,3],coef[i,4],coef[i,5],coef[i,6],coef[i,7],coef[i,8])
                    soma[i]= ((y-(Nw))**2).mean()*(1-self.pesoMorte)+((d-(D))**2).mean()*self.pesoMorte
            else:
                for i in range(tam2):
                    S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3],coef[i,4],coef[i,5],coef[i,6],coef[i,7])
                    soma[i]= ((y-(Nw))**2).mean()*(1-self.pesoMorte)+((d-(D))**2).mean()*self.pesoMorte
        return soma
    def fit(self, x,y,d,pesoMorte = 0.5, kappa = 1/4,p=0.2,gammaA = 1/3.5, gammaS=1/4.001, muH = 0.15,muU=0.4,xi = 0.53,omegaU = 0.29,omegaH=0.14 , bound = [[0,1/8,1/12,0,0.05],[2,1/4,1/3,0.7,0.25]],stand_error=True, beta2=True,day_mudar = None,particles=300,itera=1000,c1=0.1,c2=0.3,w=0.9,k=5,norm=2):
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound)
        '''
        if len(bound)==2:
            if len(bound[0])==5:
                bound[0]=bound[0].copy()
                bound[1]=bound[1].copy()
                bound[0].append(0)
                bound[0].append(0)
                bound[0].append(0)
                bound[1].append(10/self.N)
                bound[1].append(10/self.N)
                bound[1].append(10/self.N)
        self.pesoMorte = pesoMorte
        self.kappa = kappa
        self.p = p
        self.gammaA = gammaA
        self.gammaS = gammaS
        self.muH = muH
        self.muU = muU
        self.xi = xi
        self.omegaU = omegaU
        self.omegaH = omegaH
        self.beta_variavel = beta2
        self.day_mudar = day_mudar
        self.y = y
        self.d = d
        self.x = x
        df = np.array(y)
        dd = np.array(d)
        options = {'c1': c1, 'c2': c2, 'w': w,'k':k,'p':norm}
        optimizer = None
        if bound==None:
            if (beta2) & (day_mudar==None):
                optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=10, options=options)
            elif beta2:
                optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=9, options=options)
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=8, options=options)                
        else:
            if (beta2) & (day_mudar==None):
                if len(bound[0])==8:
                    bound = (bound[0].copy(),bound[1].copy())
                    bound[0].insert(1,bound[0][0])
                    bound[1].insert(1,bound[1][0])
                    bound[0].insert(2,x[4])
                    bound[1].insert(2,x[-5])

                    
                optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=10, options=options,bounds=bound)
            elif beta2:
                if len(bound[0])==8:
                    bound = (bound[0].copy(),bound[1].copy())
                    bound[0].insert(1,bound[0][0])
                    bound[1].insert(1,bound[1][0])
                    
                optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=9, options=options,bounds=bound)
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=8, options=options,bounds=bound)
                
        cost = pos = None
        #__cal_EDO(self,x,beta,gammaH,gammaU,delta,h,ia0,is0,e0)
        #__cal_EDO_2(self,x,beta1,beta2,tempo,gammaH,gammaU,delta,h,ia0,is0,e0)
        if beta2:
            cost, pos = optimizer.optimize(self.objectiveFunction,itera, x = x,y=df,d=dd,stand_error=stand_error,n_processes=self.numeroProcessadores)
        else:
            cost, pos = optimizer.optimize(self.objectiveFunction, itera, x = x,y=df,d=dd,stand_error=stand_error,n_processes=self.numeroProcessadores)
            self.beta = pos[0]
            self.gammaH = pos[1]
            self.gammaU = pos[2]
            self.delta = pos[3]
            self.h = pos[4]
            self.ia0 = pos[5]
            self.is0 = pos[6]
            self.e0 = pos[7]
        if beta2:
            self.beta1 = pos[0]
            self.beta2 = pos[1]
            
            if day_mudar==None:
                self.day_mudar = pos[2]
                self.gammaH = pos[3]
                self.gammaU = pos[4]
                self.delta = pos[5]
                self.h = pos[6]
                self.ia0 = pos[7]
                self.is0 = pos[8]
                self.e0 = pos[9]
            else:
                self.day_mudar = day_mudar
                self.gammaH = pos[2]
                self.gammaU = pos[3]
                self.delta = pos[4]
                self.h = pos[5]
                self.ia0 = pos[6]
                self.is0 = pos[7]
                self.e0 = pos[8]
        self.rmse = cost
        self.optimize = optimizer
            
    def predict(self,x):
        ''' x = dias passados do dia inicial 1'''
        if self.beta_variavel:
            S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(x,self.beta1,self.beta2,self.day_mudar,self.gammaH,self.gammaU,self.delta,self.h,self.ia0,self.is0,self.e0)
        else:
            S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO(x,self.beta,self.gammaH,self.gammaU,self.delta,self.h,self.ia0,self.is0,self.e0)
        self.ypred = Nw
        self.dpred = D
        self.S = S
        self.E = E
        self.IA = IA
        self.IS = IS
        self.H = H
        self.U = U
        self.R = R         
        return self.ypred

#Auxiliary functions to compute R(t)
#(Fjj - Fii)
    def __prod(self, i, F):
        P = 1
        for j in range(0, len(F)):
            if i != j:
                P = P * (F[j] - F[i])
        return P
##compute g(x)
    def __gx(self, x, F):
        g = 0
        for i in range(len(F)):
            if 0 != self.__prod(i, F): 
                g += np.exp(-F[i]*x)/self.__prod(i, F)
        g = np.prod(F) * g
        return g

#Integral b(t-x)g(x) dx
    def __int(self, b, t, F):
        res = 0
        for x in range(t+1):
            res += b[t - x] * self.__gx(x, F)
        return res

#Compute R(t)
    def Rt(self, cummulativeCases, date):
        date = date
        cummulativeCases.index = date
        cummulativeCases = cummulativeCases[np.argsort(cummulativeCases)]

        #using cummulative cases
        cummulativeCases = np.diff(cummulativeCases[:len(cummulativeCases) + 1])
        #Defining the F matrix array
        try:
            F = np.array([self.kappa, self.gammaA, self.gammaS])
            #initiate a empety list to get result
            res = []
            for t in range(0,len(cummulativeCases)):
                res.append(cummulativeCases[t]/self.__int(cummulativeCases, t, F))
            self.rt = pd.Series(np.array(res))
            self.rt.index  = np.sort(date.iloc[1:])
            return(self.rt)
        except:
            return("Model must be fitted before R(t) could be computed")
        

    def getResiduosQuadatico(self):
        y = np.array(self.y)
        d = np.array(self.d)
        ypred = np.array(self.ypred)
        dpred = np.array(self.dpred)
        y = y[0:len(self.x)]
        d = d[0:len(self.x)]
        ypred = ypred[0:len(self.x)]
        dpred = dpred[0:len(self.x)]
        return ((y - ypred)**2)*(1-self.pesoMorte) + ((d-dpred)**2)*self.pesoMorte

    def getReQuadPadronizado(self):
        y = np.array(self.y)
        d = np.array(self.d)
        ypred = np.array(self.ypred)
        dpred = np.array(self.dpred)
        y = y[0:len(self.x)]
        d = d[0:len(self.x)]
        ypred = ypred[0:len(self.x)]
        dpred = dpred[0:len(self.x)]
        return (((y - ypred)**2)/np.sqrt(ypred+1))*(1-self.pesoMorte) + (((d-dpred)**2)/np.sqrt(dpred+1))*self.pesoMorte
    
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

    def plotDeath(self,local):
        self.predict(self.x)
        plt.plot(self.dpred,c='b',label='Predição mortes')
        plt.plot(self.d,c='r',marker='o', markersize=3,label='mortos')
        plt.legend(fontsize=15)
        plt.title('Dinâmica do CoviD19 - {}'.format(local),fontsize=20)
        plt.ylabel('Mortos',fontsize=15)
        plt.xlabel('Dias',fontsize=15)
        plt.show()

    def getCoef(self):
        if self.beta_variavel:
            return ['beta1','beta2','dia_mudanca','gammaH','gammaU', 'delta','h','ia0','is0','e0'],[self.beta1,self.beta2,self.day_mudar,self.gammaH,self.gammaU,self.delta,self.h,self.ia0,self.is0,self.e0]
        return ['beta','gammaH','gammaU', 'delta','h','ia0','is0','e0'],[self.beta,self.gammaH,self.gammaU,self.delta,self.h,self.ia0,self.is0,self.e0]

    def plotFit(self):
        plt.style.use('seaborn-deep')
        fig, axes = plt.subplots(figsize = (18,8))
        try:
            plt.plot(self.x, self.ypred, label = "Fitted", c = "red")
            plt.scatter(self.x, self.y, label = "Observed", c = "blue")
            plt.legend(loc='upper left')
            plt.show()
        except:
            print("There is no predicted value")