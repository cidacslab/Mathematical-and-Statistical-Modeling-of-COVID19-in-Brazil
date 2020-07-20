#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:18:44 2020
@author: Rafael Veiga rafaelvalenteveiga@gmail.com
@author: matheustorquato matheusft@gmail.com
ADICIONAR OS OUTROS AUTORES
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as spi
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import pickle as pk
from numbers import Number
import copy
import matplotlib.gridspec as gridspec
from datetime import date, timedelta
import pyswarms.backend as P
from pyswarms.backend.topology import Ring


class Models:
    def __init__(self,popSize,nCores=None):
        self.isFit=False
        self.BetaChange = 0
        self.isCI = False
        self.isRT = False
        self.N = popSize
        self.nCores = nCores
    
    def __validadeVar(self,var,name):
        if len(var)<3:
            print('\nthe '+name+' variable has les than 3 elements!\n')
            return False
        for n in var:
            if not isinstance(n, Number):
                print('\nthe elemente '+str(n)+' in '+name+' variable is not numeric!\n')
                return False
        if name=='y':
            flag = 0
            i = 1
            while i < len(var): 
                if(var[i] < var[i - 1]): 
                    flag = 1
                i += 1
            if flag:
                print('\nthe y is not sorted!\n')
                return False
        var = sorted(var) 
        if var[0]<0:
            print('the '+ name + ' can not have negative  '+ str(var[0])+' value!')
            return False
        if name=='y' and var[0]==0:
            print('the y can not have 0 value!')
            return False
        return True
               
    def __changeCases(self, y):
        tam = len(y)
        res = np.ones(tam)
        res[0] = y[0]
        for i in range(1,tam):
            res[i] = y[i]-y[i-1]
        return res
    
    def __genBoot(self, series, times = 500):
        series = np.diff(series)
        series = np.insert(series, 0, 1)
        series[series < 0] = 0
        results = []
        for i in range(0,times):
            results.append(np.random.multinomial(n = sum(series), pvals = series/sum(series)))
        return np.array(results)
    
    def __getConfidenceInterval(self, series, level,isVar):
        series = np.array(series)
        if isVar:
            #Compute mean value
            meanValue = np.mean(series) 
            #Compute deltaStar
            deltaStar = meanValue - series
            #Compute lower and uper bound
            q= (1-level)/2
            deltaL = np.quantile(deltaStar, q = q)
            deltaU = np.quantile(deltaStar, q = 1-q) 

        else:
            length = len(series[0])
            #Compute mean value
            meanValue = [np.mean(series[:,i]) for i in range(0,length)]
            #Compute deltaStar
            deltaStar = meanValue - series
            #Compute lower and uper bound
            q= (1-level)/2
            deltaL = [np.quantile(deltaStar[:,i], q = q) for i in range(0,length)]
            deltaU = [np.quantile(deltaStar[:,i], q = 1-q) for i in range(0,length)]


        
        

        #Compute CI
        lowerBound  = np.array(meanValue) + np.array(deltaL)
        UpperBound  = np.array(meanValue) + np.array(deltaU)
        return (lowerBound, UpperBound)
        
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
        if self.isFit:
            plot_cost_history(cost_history=self.cost_history)
            plt.show()
        else:
            print('\nModels is not fitted\n')

    def save(self,fileName):
        file = open(fileName,'wb')
        pk.dump(self,file)
        file.close()
        
    def load(fileName):
        file = open(fileName,'rb')
        model = pk.load(file)
        return model
    def __fitNBeta(self,dim,n_particles,itera,options,objetive_function,BetaChange,bound):
        my_topology = Ring()
        my_swarm = P.create_swarm(n_particles=n_particles, dimensions=dim, options=options,bounds=bound)
        my_swarm.pbest_cost = np.full(n_particles, np.inf)
        my_swarm.best_cost = np.inf
        
        for i in range(itera):
            for a in range(n_particles):
                my_swarm.position[a][0:BetaChange] = sorted(my_swarm.position[a][0:BetaChange])
                for c in range(1,self.BetaChange):
                    if my_swarm.position[a][c-1]+5>=my_swarm.position[a][c]:
                        my_swarm.position[a][c]=my_swarm.position[a][c]+5
            my_swarm.current_cost = objetive_function(my_swarm.position)
            my_swarm.pbest_pos, my_swarm.pbest_cost = P.operators.compute_pbest(my_swarm)
            #my_swarm.current_cost[np.isnan(my_swarm.current_cost)]=np.nanmax(my_swarm.current_cost)
            #my_swarm.pbest_cost = objetive_function(my_swarm.pbest_pos)
            
            
            my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm,options['p'],options['k'])
            if i%20==0:
                print('Iteration: {} | my_swarm.best_cost: {:.4f} | days: {}'.format(i+1, my_swarm.best_cost, str(my_swarm.pbest_pos[my_swarm.pbest_cost.argmin()])))
            my_swarm.velocity = my_topology.compute_velocity(my_swarm,  bounds=bound)
            my_swarm.position = my_topology.compute_position(my_swarm,bounds=bound)
        final_best_cost = my_swarm.best_cost.copy()
        final_best_pos = my_swarm.pbest_pos[my_swarm.pbest_cost.argmin()].copy()
        return final_best_pos,final_best_cost

class SEIRHUD(Models):
    ''' SEIRHU Model'''
    
    def __cal_EDO(self,x,beta,gammaH,gammaU,delta,h,ia0,is0,e0):
            t_range = x
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
        
    def __cal_EDO_2(self,x,tVar,betaVar,gammaH,gammaU,delta,h,ia0,is0,e0):
            t_range = x
            
            def beta(t,tVar,betaVar):
                for i in range(len(tVar)):
                    if t<tVar[i]:
                        return betaVar[i]
                return betaVar[i+1]

            delta = np.array(delta)
            def SIR_diff_eqs(INP, t, tVar, betaVar,gammaH,gammaU, delta,h):
                #Y[0] = - beta(t,t1,beta1,beta2) * V[0] * V[1]                 #S
                Y = np.zeros((9))
                V = INP
                Y[0] = - beta(t,tVar,betaVar)*V[0]*(V[3] + delta*V[2])                    #S
                Y[1] = beta(t,tVar,betaVar)*V[0]*(V[3] + delta*V[2]) -self.kappa * V[1]
                Y[2] = (1-self.p)*self.kappa*V[1] - self.gammaA*V[2]
                Y[3] = self.p*self.kappa*V[1] - self.gammaS*V[3]
                Y[4] = h*self.xi*self.gammaS*V[3] + (1-self.muU + self.omegaU*self.muU)*gammaU*V[5] -gammaH*V[4]
                Y[5] = h*(1-self.xi)*self.gammaS*V[3] +self.omegaH*gammaH*V[4] -gammaU*V[5]
                Y[6] = self.gammaA*V[2] + (1-(self.muH))*(1-self.omegaH)*gammaH*V[4] + (1-h)*self.gammaS*V[3]
                Y[7] = (1-self.omegaH)*self.muH*gammaH*V[4] + (1-self.omegaU)*self.muU*gammaU*V[5]#R
                Y[8] = self.p*self.kappa*V[1]                      #R
                
                return Y
            result_fit = spi.odeint(SIR_diff_eqs, (1-ia0-is0-e0,e0 ,ia0,is0,0,0,0,0,0), t_range,
                                    args=(tVar,betaVar,gammaH,gammaU, delta,h))
            
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
        
    def __residuals(self,coef):
        if (self.BetaChange!=0) & (self.dayBetaChange==None):
            tVar = np.ones(self.BetaChange)
            betaVar = np.ones(self.BetaChange+1)
            for i in range(self.BetaChange):
                tVar[i] = coef[i]
                betaVar[i] = coef[self.BetaChange+i]
            betaVar[self.BetaChange] = coef[self.BetaChange*2]
            S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(self.x,tVar,betaVar,coef[self.BetaChange*2+1],coef[self.BetaChange*2+2],coef[self.BetaChange*2+3],coef[self.BetaChange*2+4],coef[self.BetaChange*2+5],coef[self.BetaChange*2+6],coef[self.BetaChange*2+7])
        elif self.dayBetaChange!=None:
            tVar = self.dayBetaChange
            betaVar = np.ones(self.BetaChange+1)
            for i in range(self.BetaChange+1):
                betaVar[i] = coef[i]      
            S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(self.x,tVar,betaVar,coef[self.BetaChange+1],coef[self.BetaChange+2],coef[self.BetaChange+3],coef[self.BetaChange+4],coef[self.BetaChange+5],coef[self.BetaChange+6],coef[self.BetaChange+7])
        else:
            S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO(self.x,coef[0],coef[1],coef[2],coef[3],coef[4],coef[5],coef[6],coef[7])
        
        
        aux1 = Nw
        aux2 = self.d
        auxT2 = aux2 - D
        aux3 = self.hos
        auxT3 = aux3 - H if self.hos else None
        aux4 = self.u
        auxT4 = aux4 - U if self.u else False
        
        if self.fittingByCumulativeCases:
            auxT1 = self.y-aux1 
        else:
            aux1 = self._Models__changeCases(aux1) 
            auxT1 = self.NC - aux1
            
        if self.stand_error:
            try:
                auxT1 = auxT1 / np.sqrt(aux1+1)
                auxT2 = auxT2/ np.sqrt(aux2+1)
                if self.hos:
                    auxT3 = auxT3 / np.sqrt(aux3+1)
                if self.u:
                    auxT4 = auxT4 / np.sqrt(aux4+1)
            except:
                pass
            
        return (auxT1,auxT2,auxT3,auxT4)        
    
    def _objectiveFunction(self,coef):
        tam2 = len(coef[:,0])
        soma = np.zeros(tam2)
        for i in range(tam2):
            res = self.__residuals(coef[i])
            soma[i] = ((res[0])**2).mean()*self.yWeight + ((res[1])**2).mean()*self.dWeight
            soma[i] = (soma[i] + ((res[2])**2).mean()*self.hosWeight) if self.hos else soma[i]
            soma[i] = (soma[i] + ((res[3])**2).mean()*self.hosWeight) if self.hos else soma[i]
        return soma
    
    def __validateBound(self, bound):
        if bound==None:
            self.bound = bound
            return True
        if len(bound)!=2:
           raise ValueError("Bound of Incorrect size")
           return False
        if (self.BetaChange!=0) & (self.dayBetaChange==None):
            if len(bound[0])==5:
                #beta,gammaH,gammaU,delta,h
                b1 = bound[0][0]
                b2 = bound[1][0]
                gh1 = bound[0][1]
                gh2 = bound[1][1]
                gu1 = bound[0][2]
                gu2 = bound[1][2]
                d1 = bound[0][3]
                d2 = bound[1][3] 
                h1 = bound[0][4]
                h2 = bound[1][4] 
                bound2 = ([],[])
                for i in range(self.BetaChange+1):
                    bound2[0].insert(0,b1)
                    bound2[1].insert(0,b2)
                for i in range(self.BetaChange):
                    bound2[0].insert(0,self.x[1]+(i+1)*5)
                    bound2[1].insert(0,self.x[-2]-(self.BetaChange-i)*5)
                bound2[0].append(gh1)
                bound2[1].append(gh2)
                bound2[0].append(gu1)
                bound2[1].append(gu2)
                bound2[0].append(d1)
                bound2[1].append(d2)
                bound2[0].append(h1)
                bound2[1].append(h2)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
                self.bound = bound2
                return True
                
            elif len(bound[0]) == (self.BetaChange*2)+8:
                self.bound = bound
                return True
            else:
                raise ValueError("Bound of Incorrect size")
                return False
        elif self.BetaChange!=0:
            if len(bound[0])==5:
                b1 = bound[0][0]
                b2 = bound[1][0]
                gh1 = bound[0][1]
                gh2 = bound[1][1]
                gu1 = bound[0][2]
                gu2 = bound[1][2]
                d1 = bound[0][3]
                d2 = bound[1][3] 
                h1 = bound[0][4]
                h2 = bound[1][4]
                bound2 = ([],[])
                for i in range(self.BetaChange+1):
                    bound2[0].insert(0,b1)
                    bound2[1].insert(0,b2)
                bound2[0].append(gh1)
                bound2[1].append(gh2)
                bound2[0].append(gu1)
                bound2[1].append(gu2)
                bound2[0].append(d1)
                bound2[1].append(d2)
                bound2[0].append(h1)
                bound2[1].append(h2)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
                self.bound = bound2
                return True
            
            elif len(bound[0]) != self.BetaChange+5:
                raise ValueError("Bound of Incorrect size")
                return False
        else:
            if len(bound[0])==5:
                b1 = bound[0][0]
                b2 = bound[1][0]
                gh1 = bound[0][1]
                gh2 = bound[1][1]
                gu1 = bound[0][2]
                gu2 = bound[1][2]
                d1 = bound[0][3]
                d2 = bound[1][3] 
                h1 = bound[0][4]
                h2 = bound[1][4]
                bound2 = ([],[])
                
                bound2[0].append(b1)
                bound2[1].append(b2)
                bound2[0].append(gh1)
                bound2[1].append(gh2)
                bound2[0].append(gu1)
                bound2[1].append(gu2)
                bound2[0].append(d1)
                bound2[1].append(d2)
                bound2[0].append(h1)
                bound2[1].append(h2)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
                bound2[0].append(0)
                bound2[1].append(self.y[0]/self.N)
        self.bound = bound
        return True

    def __fit(self,x, y ,d,hos=None,u=None,c1= 0.6, c2= 0.5, w = 0.9):
        options = {'c1': c1, 'c2': c2, 'w': w}
        self.x = x
        self.y = np.array(y)
        self.NC = self._Models__changeCases(self.y)
        self.d = np.array(d)
        if self.hos:
            self.hos=hos
        if self.u:
            self.u=u
            
        if self.BetaChange!=0:
            if self.isFitDayBetaChange == True:
                pos = []
                for i in range(self.BetaChange):
                    pos.append(self.dayBetaChange[i])
                for i in range(self.BetaChange+1):
                    pos.append(self.beta[i])
                pos.append(self.gammaH)
                pos.append(self.gammaU)
                pos.append(self.delta)
                pos.append(self.h)
                pos.append(self.ia0)
                pos.append(self.is0)
                pos.append(self.e0)
                
                pos = np.array(pos)
                par=ps.backend.generators.create_swarm(10,dimensions=((self.BetaChange+1)*2)+6,bounds=self.bound)
                for i in range(5):
                    par.position[i]=pos
                
                optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=((self.BetaChange+1)*2)+6, options=options,bounds=self.bound,init_pos=par.position)
                cost, pos = optimizer.optimize(self._objectiveFunction, 1500,n_processes=self.nCores)
                
                beta = []
                dayBetaChange = []
                for i in range(self.BetaChange):
                    dayBetaChange.append(pos[i])
                    beta.append(pos[self.BetaChange+i])
                beta.append(pos[self.BetaChange*2])
                #gammaH,gammaU,delta,h,ia0,is0,e0
                self.gammaH = pos[self.BetaChange*2+1]
                self.gammaU = pos[self.BetaChange*2+2]
                self.delta = pos[self.BetaChange*2+3]
                self.h = pos[self.BetaChange*2+4]
                self.ia0 = pos[self.BetaChange*2+5]
                self.is0 = pos[self.BetaChange*2+6]
                self.e0 = pos[self.BetaChange*2+7]
                self.dayBetaChange = np.array(dayBetaChange)
                self.beta = np.array(beta)
                self.rmse = cost
                #self.cost_history = optimizer.cost_history
                
            else:
                optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=self.BetaChange+7, options=options,bounds=self.bound)
                cost, pos = optimizer.optimize(self._objectiveFunction, 1500,n_processes=self.nCores)
                beta = []
                for i in range(len(self.BetaChange)+1):
                    beta.append(pos[i])
                
                self.gammaH = pos[self.BetaChange+1]
                self.gammaU = pos[self.BetaChange*+2]
                self.delta = pos[self.BetaChange+3]
                self.h = pos[self.BetaChange+4]
                self.ia0 = pos[self.BetaChange+5]
                self.is0 = pos[self.BetaChange+6]
                self.e0 = pos[self.BetaChange+7]
                self.dayBetaChange = np.array(dayBetaChange)
                self.beta = np.array(beta)
                self.rmse = cost
                self.cost_history = optimizer.cost_history
        else:
            optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=8, options=options,bounds=self.bound)
            cost, pos = optimizer.optimize(self._objectiveFunction, 1500,n_processes=self.nCores)
            beta = [pos[0]]
            self.beta = np.array(beta)
            self.gammaH = pos[1]
            self.gammaU = pos[2]
            self.delta = pos[3]
            self.h = pos[4]
            self.ia0 = pos[5]
            self.is0 = pos[6]
            self.e0 = pos[7]
        self.isFit=True
        self.predict(0)
    
    def fit(self, x, y, d, fittingByCumulativeCases=True, hos=None,u=None,yWeight=1,dWeight = 1,hosWeight=1,uWeight=1, kappa = 1/4,p = 0.2,gammaA = 1/3.5, gammaS = 1/4.001, muH = 0.15,
            muU = 0.4,xi = 0.53,omegaU = 0.29,omegaH=0.14 , bound = [[0,1/8,1/12,0,0],[2,1/4,1/3,0.7,0.35]],
            stand_error = True, BetaChange = 1, dayBetaChange = None, particles = 300, itera = 1000, c1 = 1, c2 = 0.5, w = 0.9, k = 5, norm = 1):
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound)
        '''
        self.BetaChange = BetaChange
        if dayBetaChange==None:   
            self.dayBetaChange = dayBetaChange
            self.isFitDayBetaChange = True
        else:
            self.isFitDayBetaChange = False
            self.dayBetaChange = np.array(dayBetaChange)
        if not self._Models__validadeVar(y,'y'):
            return
        if not self._Models__validadeVar(d,'d'):
            return
        if hos:
            if not self._Models__validadeVar(hos,'hos'):
                return
        if u:
            if not self._Models__validadeVar(u,'u'):
                return
        self.yWeight = yWeight
        self.dWeight = dWeight
        self.hosWeight = hosWeight
        self.uWeight = uWeight
        self.kappa = kappa
        self.p = p
        self.gammaA = gammaA
        self.gammaS = gammaS
        self.muH = muH
        self.muU = muU
        self.xi = xi
        self.omegaU = omegaU
        self.omegaH = omegaH
        self.BetaChange = BetaChange
        self.dayBetaChange = dayBetaChange
        self.y = np.array(y)
        self.d = np.array(d)
        self.x = np.array(x)
        if hos:
            self.hos = np.array(hos)
        else:
            self.hos = hos
        if u:
            self.u = np.array(u)
        else:
           self.u = u 
        self.fittingByCumulativeCases = fittingByCumulativeCases
        self.stand_error = stand_error
        self.particles = particles
        self.itera = itera
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.k = k
        self.norm = norm
        self.NC = self._Models__changeCases(self.y)
        if not self.__validateBound(bound):
            return
        #self.bound=bound
        options = {'c1': c1, 'c2': c2, 'w': w,'k':k,'p':norm}
        if self.BetaChange!=0:
            if self.dayBetaChange==None:
                if self.BetaChange>=2:
                #optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=self.BetaChange*2+2, options=options,bounds=self.bound)
                #cost, pos = optimizer.optimize(self._objectiveFunction, itera,n_processes=self.nCores)
                #__fitNBeta(self,dim,n_particles,itera,options,objetive_function,**kwargs)
                    pos,cost = self._Models__fitNBeta(dim=((self.BetaChange+1)*2)+6,n_particles=self.particles,itera=self.itera,options=options,objetive_function=  self._objectiveFunction,BetaChange= self.BetaChange,bound=self.bound)
                else:
                    optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=((self.BetaChange+1)*2)+6, options=options,bounds=self.bound)
                    cost, pos = optimizer.optimize(self._objectiveFunction, itera,n_processes=self.nCores)

                print('posi='+str(pos)+'\n'+str(cost))
                beta = []
                dayBetaChange = []
                for i in range(self.BetaChange):
                    dayBetaChange.append(pos[i])
                    beta.append(pos[self.BetaChange+i])
                beta.append(pos[self.BetaChange*2])
                #gammaH,gammaU,delta,h,ia0,is0,e0
                self.gammaH = pos[self.BetaChange*2+1]
                self.gammaU = pos[self.BetaChange*2+2]
                self.delta = pos[self.BetaChange*2+3]
                self.h = pos[self.BetaChange*2+4]
                self.ia0 = pos[self.BetaChange*2+5]
                self.is0 = pos[self.BetaChange*2+6]
                self.e0 = pos[self.BetaChange*2+7]
                self.dayBetaChange = np.array(dayBetaChange)
                self.beta = np.array(beta)
                self.rmse = cost
                #self.cost_history = optimizer.cost_history
                
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=self.BetaChange+3, options=options,bounds=self.bound)
                cost, pos = optimizer.optimize(self._objectiveFunction, itera,n_processes=self.nCores)
                beta = []
                for i in range(len(self.BetaChange)+1):
                    beta.append(pos[i])
                beta.append(pos[self.BetaChange*2])
                self.gammaH = pos[self.BetaChange*2+1]
                self.gammaU = pos[self.BetaChange*2+2]
                self.delta = pos[self.BetaChange*2+3]
                self.h = pos[self.BetaChange*2+4]
                self.ia0 = pos[self.BetaChange*2+5]
                self.is0 = pos[self.BetaChange*2+6]
                self.e0 = pos[self.BetaChange*2+7]
                self.dayBetaChange = np.array(dayBetaChange)
                self.beta = np.array(beta)
                self.rmse = cost
                self.cost_history = optimizer.cost_history
        else:
            optimizer = ps.single.LocalBestPSO(n_particles=particles, dimensions=8, options=options,bounds=self.bound)
            cost, pos = optimizer.optimize(self._objectiveFunction, itera,n_processes=self.nCores)
            beta = [pos[0]]
            self.beta = np.array(beta)
            self.gammaH = pos[1]
            self.gammaU = pos[2]
            self.delta = pos[3]
            self.h = pos[4]
            self.ia0 = pos[5]
            self.is0 = pos[6]
            self.e0 = pos[7]
        self.isFit=True
        self.predict(0)
    
    

    def predict(self,numDays):
        ''' x = dias passados do dia inicial 1'''
        if numDays<0:
            print('\nnumDays must be a positive number!\n')
            return
        if self.isFit==False:
            print('\nModels is not fitted\n')
            return None
        x = np.arange(self.x[0], self.x[-1] + 1+numDays) 
        self.predictNumDays = numDays
        if self.BetaChange==0:
            S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO(x,self.beta,self.gammaH,self.gammaU,self.delta,self.h,self.ia0,self.is0,self.e0)
        else:
            S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(x,self.dayBetaChange,self.beta,self.gammaH,self.gammaU,self.delta,self.h,self.ia0,self.is0,self.e0)
        self.ypred = Nw
        self.NCpred = self._Models__changeCases(self.ypred)
        self.xpred = x
        self.S = S
        self.E = E
        self.IA = IA
        self.IS = IS
        self.H = H
        self.U = U
        self.R = R
        self.D = D
        return self.ypred

#Compute R(t)
    def Rt(self, cutoof):
        #Auxiliary functions to compute R(t)
        #(Fjj - Fii)
        if self.isFit==False:
            print('\nModels is not fitted\n')
            return None
        
        def _prod(i, F):
            P = 1
            for j in range(0, len(F)):
                    if i != j:
                        P = P * (F[j] - F[i])
            return P
        ##compute g(x)
        def _gx(x, F):
            g = 0
            for i in range(len(F)):
                if 0 != _prod(i, F): 
                    g += np.exp(-F[i]*x)/_prod(i, F)
            g = np.prod(F) * g
            return g
        #Integral b(t-x)g(x) dx
        def _int( b, t, F):
            res = 0
            for x in range(t+1):
                res += b[t - x] * _gx(x, F)
            return res
        
        if self.isFit==False:
            print('\nModels is not fitted\n')
            return None
        

        cummulativeCases = np.array(self.y)
        #using cummulative cases
        cummulativeCases = np.diff(cummulativeCases[:len(cummulativeCases) + 1])
        #Defining the F matrix array
    #try:
        F = np.array([self.kappa, self.gammaA, self.gammaS])
        #initiate a empety list to get result
        res = []
        for t in range(0,len(cummulativeCases)):
            res.append(cummulativeCases[t]/_int(cummulativeCases, t, F))
        self.rt = pd.Series(np.array(res))
        idx_start = np.searchsorted(np.cumsum(cummulativeCases),cutoof)
        self.isRT=True
        return(self.rt.iloc[idx_start:])
    #except:
        #return("Model must be fitted before R(t) could be computed")
        

    def plot(self,local):
        ypred = self.predict(self.x)
        plt.plot(ypred,c='b',label='Predição Infectados')
        plt.plot(self.y,c='r',marker='o', markersize=3,label='Infectados')
        plt.legend(fontsize=15)
        plt.title('Dinâmica do CoviD19 - {}'.format(local),fontsize=20)
        plt.ylabel('Casos COnfirmados',fontsize=15)
        plt.xlabel('Dias',fontsize = 15)
        plt.show()

    def plotDeath(self,local):
        self.predict(self.x)
        plt.plot(self.dpred,c='b',label='Predição mortes')
        plt.plot(self.d,c='r',marker='o', markersize=3,label='mortos')
        plt.legend(fontsize = 15)
        plt.title('Dinâmica do CoviD19 - {}'.format(local),fontsize=20)
        plt.ylabel('Mortos',fontsize=15)
        plt.xlabel('Dias',fontsize=15)
        plt.show()

        
    def getCoef(self):
        if self.BetaChange>0:
            var = []
            valor = []
            for i in range(self.BetaChange+1):
                var.append('beta'+str(i+1))
                valor.append(self.beta[i])
            for i in range(self.BetaChange):
                var.append('dayBetaChange'+str(i+1))
                valor.append(self.dayBetaChange[i])
            var.append('gammaH')
            var.append('gammaU')
            var.append('delta')
            var.append('h')
            var.append('ia0')
            var.append('is0')
            var.append('e0')
            valor.append(self.gammaH)
            valor.append(self.gammaU)
            valor.append(self.delta)
            valor.append(self.h)
            valor.append(self.ia0)
            valor.append(self.is0)
            valor.append(self.e0)
            return dict(zip(var, valor))
        return dict(zip(['beta','gammaH','gammaU','delta','h','ia0','is0'], [self.beta,self.gammaH,self.gammaU,self.delta,self.h,self.ia0,self.is0]))

    
    def getEstimation(self):
        return dict(zip(['S','E','IA','IS','H','U','R','D','Cumulative_cases_predict','new_cases_predict'],[self.S,self.E,self.IA,self.IS,self.H,self.U,self.R,self.D,self.ypred,self.NCpred]))
    
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
    
    def computeCI(self, times=500, level=0.95):
        if self.isFit==False:
            print('\nModels is not fitted\n')
            return None
        if self.isCI:
            self.lypred = self._Models__getConfidenceInterval(self.__bypred, level,False)
            self.lS = self._Models__getConfidenceInterval(self.__bS, level,False)
            self.lE = self._Models__getConfidenceInterval(self.__bE, level,False)
            self.lIA = self._Models__getConfidenceInterval(self.__bIA, level,False)
            self.lIS = self._Models__getConfidenceInterval(self.__bIS, level,False)
            self.lH = self._Models__getConfidenceInterval(self.__bH, level,False)
            self.lU = self._Models__getConfidenceInterval(self.__bU, level,False)
            self.lR = self._Models__getConfidenceInterval(self.__bR, level,False)
            self.lD = self._Models__getConfidenceInterval(self.__bD, level,False)
            self.lBeta=[]
            self.lDayBetaChange=[]
            if self.BetaChange>0:
                for i in range(self.BetaChange):
                    self.lBeta.append(self._Models__getConfidenceInterval(self.__bBeta[i], level,True))
                    self.lDayBetaChange.append(self._Models__getConfidenceInterval(self.__bDayBetaChange[i], level,True))
                self.lBeta.append(self._Models__getConfidenceInterval(self.__bBeta[self.BetaChange], level,True))
            else:
                self.lBeta = self._Models__getConfidenceInterval(self.__bBeta, level,True)
            self.lGammaH = self._Models__getConfidenceInterval(self.__bGammaH, level,True)
            self.lGammaU = self._Models__getConfidenceInterval(self.__bGammaU, level,True)
            self.lDelta = self._Models__getConfidenceInterval(self.__bDelta, level,True)
            self.lh = self._Models__getConfidenceInterval(self.__bh, level,True)
            self.lIa0 = self._Models__getConfidenceInterval(self.__bIa0, level,True)
            self.lIs0 = self._Models__getConfidenceInterval(self.__bIs0, level,True)
            self.lE0 = self._Models__getConfidenceInterval(self.__bE0, level,True)
        #Define empty lists to recive results
        self.__bypred = []
        self.__bS = []
        self.__bE = []
        self.__bIA = []
        self.__bIS = []
        self.__bH = []
        self.__bU = []
        self.__bR = []
        self.__bD = []
        
        self.__bBeta = []
        if self.BetaChange>0:
            self.__bDayBetaChange=[]
        for i in range(self.BetaChange):   
            self.__bDayBetaChange.append([])
            self.__bBeta.append([])
        self.__bBeta.append([])
        #gammaH,gammaU,delta,h,ia0,is0,e0
        self.__bGammaH = []
        self.__bGammaU = []
        self.__bDelta = []
        self.__bh = []
        self.__bIa0 = []
        self.__bIs0 = []
        self.__bE0 = []
        
        casesSeries = self._Models__genBoot(self.y, times)
        casesDeath = self._Models__genBoot(self.d, times)
        if self.hos:
            casesHos = self._Models__genBoot(self.hos, times)
        if self.u:
            casesU = self._Models__genBoot(self.u, times)
        
        for i in range(0,len(casesSeries)):
            copia = copy.deepcopy(self)
            print("\n"+str(i)+'\n')
            copia.__fit(x=self.x,y = casesSeries[i],d=casesDeath[i],hos= (casesHos[i] if self.hos else None),u= (casesU[i] if self.u else None))
            

            self.__bypred.append(copia.ypred)
            self.__bS.append(copia.S)
            self.__bE.append(copia.E)
            self.__bIA.append(copia.IA)
            self.__bIS.append(copia.IS)
            self.__bH.append(copia.H)
            self.__bU.append(copia.U)
            self.__bR.append(copia.R)
            self.__bD.append(copia.D)
            for a in range(self.BetaChange):
                self.__bBeta[a].append(copia.beta[a])
                self.__bDayBetaChange[a].append(copia.dayBetaChange[a])
            self.__bBeta[self.BetaChange].append(copia.beta[self.BetaChange])
            self.__bGammaH.append(copia.gammaH)
            self.__bGammaU.append(copia.gammaU)            
            self.__bDelta.append(copia.delta)
            self.__bh.append(copia.h)
            self.__bIa0.append(copia.ia0)
            self.__bIs0.append(copia.is0)
            self.__bE0.append(copia.e0)
        #'S','E','IA','IS','H','U','R','D'
        self.lypred = self._Models__getConfidenceInterval(self.__bypred, level,False)
        self.lS = self._Models__getConfidenceInterval(self.__bS, level,False)
        self.lE = self._Models__getConfidenceInterval(self.__bE, level,False)
        self.lIA = self._Models__getConfidenceInterval(self.__bIA, level,False)
        self.lIS = self._Models__getConfidenceInterval(self.__bIS, level,False)
        self.lH = self._Models__getConfidenceInterval(self.__bH, level,False)
        self.lU = self._Models__getConfidenceInterval(self.__bU, level,False)
        self.lR = self._Models__getConfidenceInterval(self.__bR, level,False)
        self.lD = self._Models__getConfidenceInterval(self.__bD, level,False)
        
        if self.BetaChange>0:
            self.lDayBetaChange=[]
            self.lBeta = []
            for i in range(self.BetaChange):
                self.lDayBetaChange.append(self._Models__getConfidenceInterval(self.__bDayBetaChange[i], level,True))
                self.lBeta.append(self._Models__getConfidenceInterval(self.__bBeta[i], level,True))
            self.lBeta.append(self._Models__getConfidenceInterval(self.__bBeta[self.BetaChange], level,True))
        else:
            self.lBeta = self._Models__getConfidenceInterval(self.__bBeta[0], level,True)
                #gammaH,gammaU,delta,h,ia0,is0,e0
        self.lGammaH = self._Models__getConfidenceInterval(self.__bGammaH, level,True)
        self.lGammaU = self._Models__getConfidenceInterval(self.__bGammaU, level,True)
        self.lDelta = self._Models__getConfidenceInterval(self.__bDelta, level,True)
        self.lh = self._Models__getConfidenceInterval(self.__bh, level,True)
        self.lIa0 = self._Models__getConfidenceInterval(self.__bIa0, level,True)
        self.lIs0 = self._Models__getConfidenceInterval(self.__bIs0, level,True)
        self.lE0 = self._Models__getConfidenceInterval(self.__bE0, level,True)
        self.isCI=True
        
    