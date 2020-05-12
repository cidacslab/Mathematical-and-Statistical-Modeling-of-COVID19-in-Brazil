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
from pyswarms.single.global_best import GlobalBestPSO
import pyswarms as ps
from pyswarms.backend.topology import Star
from pyswarms.utils.plotters import plot_cost_history
from itertools import repeat
import multiprocessing as mp



logging.disable()
def ler_banco_municipios():
    try:
        url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv'
        banco = pd.read_csv(url)
    except:
        return None,None
    
    
    banco =banco[banco['ibgeID'].notnull()]
    banco = banco[banco.city!='TOTAL']
    nome_local =list(banco['ibgeID'].unique())
    data = []
    for i in banco.index:
        print(str(i)+'\n')
        data.append( dt.datetime.strptime(banco.date[i], '%Y-%m-%d').date())
    banco.date = data
    local = []
    for est in nome_local:
        
    
        aux = banco[banco['ibgeID']==est].sort_values('date')
        data_ini = aux.date.iloc[0]
        data_fim = aux.date.iloc[-1]
        dias = (data_fim-data_ini).days + 1
        d = [(data_ini + dt.timedelta(di)) for di in range(dias)]
        
        estado = [aux.state.iloc[0] for di in range(dias)]
        city = [aux.city.iloc[0] for di in range(dias)]
        ibgeID = [est for di in range(dias)]
        df = pd.DataFrame({'date':d,'UF':estado,'city':city,'ibgeID':ibgeID})
        
        casos = []
        mortes = []
        morte = 0
        caso = 0
        i_aux = 0
        for i in range(dias):
            if (d[i]-aux.date.iloc[i_aux]).days==0:
                caso = aux['totalCases'].iloc[i_aux]
                morte = aux['deaths'].iloc[i_aux]
                casos.append(caso)
                mortes.append(morte)
                i_aux=i_aux+1
            else:
                casos.append(caso)
                mortes.append(morte)
        new = [casos[0]]        
        for i in range(1,dias):
            new.append(casos[i]-casos[i-1])
        df['mortes'] = mortes
        df['newCases'] = new
        df['TOTAL'] = casos
        local.append(df)
    return nome_local, local    


def ler_banco_estados():
    try:
        url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
        banco = pd.read_csv(url)
    except:
        return None,None
    
    

    nome_local =list(banco['state'].unique())
    nome_local.remove('TOTAL')
    nome_local.insert(0,'TOTAL')
    for i in banco.index:
        banco.date[i] = dt.datetime.strptime(banco.date[i], '%Y-%m-%d').date()
    local = []
    for est in nome_local:
        
    
        aux = banco[banco['state']==est].sort_values('date')
        data_ini = aux.date.iloc[0]
        data_fim = aux.date.iloc[-1]
        dias = (data_fim-data_ini).days + 1
        d = [(data_ini + dt.timedelta(di)) for di in range(dias)]
        
        estado = [est for di in range(dias)]
        df = pd.DataFrame({'date':d,'state':estado})
        
        casos = []
        mortes = []
        caso = 0
        morte=0
        i_aux = 0
        for i in range(dias):
            if (d[i]-aux.date.iloc[i_aux]).days==0:
                caso = aux['totalCases'].iloc[i_aux]
                morte = aux.deaths.iloc[i_aux]
                casos.append(caso)
                mortes.append(morte)
                i_aux=i_aux+1
            else:
                casos.append(caso)
                mortes.append(morte)
        new = [casos[0]]        
        for i in range(1,dias):
            new.append(casos[i]-casos[i-1])
        df['newCases'] = new
        df['mortes'] = mortes
        df['TOTAL'] = casos
        local.append(df)
        nome_local[0]='TOTAL'
        local[0].state='TOTAL'
    return nome_local, local    
            
class EXP:
    ''' f(x) = a*exp(b*x) '''
    def __init__(self, N_inicial,numeroProcessadores=None):
        self.N=N_inicial
        self.a = None
        self.b = None
        self.numeroProcessadores = numeroProcessadores
         
        
    
    def objectiveFunction(self,coef,x ,y,stand_error):
        tam = len(coef[:, 0])
        res = np.zeros(tam)
        if stand_error:
            for i in range(tam):
                res[i] = ((((coef[i, 0]*np.exp(x*coef[i, 1]))-y)/y)**2).mean()
        else:
            for i in range(tam):
                res[i] = (((coef[i, 0]*np.exp(x*coef[i, 1]))-y)**2).mean()                
        return res
            


    def fit(self, x,y , bound = None, stand_error=True):        
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound) '''
        df = np.array(y)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        if bound==None:
            optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options)
            cost, pos = optimizer.optimize(self.objectiveFunction, 500, x = x,y=df,stand_error=stand_error,n_processes=self.numeroProcessadores)
            self.a = pos[0]
            self.b = pos[1]
            self.x = x
            self.y = df
            self.rmse = cost
            self.optimize = optimizer
        else:
            optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=bound)
            cost, pos = optimizer.optimize(self.objectiveFunction, 500, x = x,y=df,stand_error=stand_error,n_processes=self.numeroProcessadores)
            self.a = pos[0]
            self.b = pos[1]
            self.x = x
            self.y = df
            self.rmse = cost
            self.optimize = optimizer
            
    def predict(self,x):
        ''' x = dias passados do dia inicial 1'''
        res = [self.a*np.exp(self.b*v) for v in x]
        self.ypred = res
         
        return res
    
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
        return ['a','b'], [self.a,self.b]
   
class SIR:
    ''' SIR Model'''
    def __init__(self,tamanhoPop,numeroProcessadores=None):
        self.N = tamanhoPop
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
        
    def __cal_EDO_2(self,x,beta1,gamma,beta2,tempo):
            ND = len(x)-1
            
            t_start = 0.0
            t_end = ND
            t_inc = 1
            t_range = np.arange(t_start, t_end + t_inc, t_inc)
            def H(t):
                h = 1.0/(1.0+ np.exp(-2.0*50*t))
                return h
            def beta(t,t1,b,b1):
                beta = b*H(t1-t) + b1*H(t-t1) 
                return beta

            gamma = np.array(gamma)
            def SIR_diff_eqs(INP, t, beta1, gamma,beta2,t1):
                Y = np.zeros((3))
                V = INP
                Y[0] = - beta(t,t1,beta1,beta2) * V[0] * V[1]                 #S
                Y[1] = beta(t,t1,beta1,beta2) * V[0] * V[1] - gamma * V[1]    #I
                Y[2] = gamma * V[1]                         #R
                
                return Y
            result_fit = spi.odeint(SIR_diff_eqs, (self.S0, self.I0,self.R0), t_range,
                                    args=(beta1, gamma,beta2,tempo))
            
            S=result_fit[:, 0]*self.N
            R=result_fit[:, 2]*self.N
            I=result_fit[:, 1]*self.N
            
            return S,I,R
    
    def objectiveFunction(self,coef,x ,y,stand_error):
        tam2 = len(coef[:,0])
        soma = np.zeros(tam2)
        y = y*self.N
        if stand_error:
            if (self.beta_variavel) & (self.day_mudar==None):
                for i in range(tam2):
                    S,I,R = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3])
                    soma[i]= (((y-(I+R))/y)**2).mean()
            elif self.beta_variavel:
                for i in range(tam2):
                    S,I,R = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],self.day_mudar)
                    soma[i]= (((y-(I+R))/y)**2).mean()
            else:
                for i in range(tam2):
                    S,I,R = self.__cal_EDO(x,coef[i,0],coef[i,1])
                    soma[i]= (((y-(I+R))/y)**2).mean()
        else:
            if (self.beta_variavel) & (self.day_mudar==None):
                for i in range(tam2):
                    S,I,R = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3])
                    soma[i]= (((y-(I+R)))**2).mean()
            elif self.beta_variavel:
                for i in range(tam2):
                    S,I,R = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],self.day_mudar)
                    soma[i]= (((y-(I+R)))**2).mean()
            else:
                for i in range(tam2):
                    S,I,R = self.__cal_EDO(x,coef[i,0],coef[i,1])
                    soma[i]= (((y-(I+R)))**2).mean()
        return soma
    def fit(self, x,y , bound = ([0,1/21],[1,1/5]),stand_error=False, beta2=True,day_mudar = None):
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound)
        '''
        self.beta_variavel = beta2
        self.day_mudar = day_mudar
        self.y = y
        self.x = x
        df = np.array(y)/self.N
        self.I0 = df[0]
        self.S0 = 1-self.I0
        self.R0 = 0
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9,'k':3,'p':1}
        optimizer = None
        if bound==None:
            if (beta2) & (day_mudar==None):
                optimizer = ps.single.LocalBestPSO(n_particles=80, dimensions=4, options=options)
            elif beta2:
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=3, options=options)
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=2, options=options)                
        else:
            if (beta2) & (day_mudar==None):
                if len(bound[0])==2:
                    bound = (bound[0].copy(),bound[1].copy())
                    bound[0].append(bound[0][0])
                    bound[1].append(bound[1][0])
                    bound[0].append(x[4])
                    bound[1].append(x[-5])
                    bound[0][3] = x[4]
                    bound[1][3] = x[-5]
                    
                optimizer = ps.single.LocalBestPSO(n_particles=80, dimensions=4, options=options,bounds=bound)
            elif beta2:
                if len(bound[0])==2:
                    bound = (bound[0].copy(),bound[1].copy())
                    bound[0].append(bound[0][1])
                    bound[1].append(bound[1][1])
                    
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=3, options=options,bounds=bound)
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=2, options=options,bounds=bound)
                
        cost = pos = None
        if beta2:
            cost, pos = optimizer.optimize(self.objectiveFunction, 500, x = x,y=df,stand_error=stand_error,n_processes=self.numeroProcessadores)
        else:
            cost, pos = optimizer.optimize(self.objectiveFunction, 500, x = x,y=df,stand_error=stand_error,n_processes=self.numeroProcessadores)
            self.beta = pos[0]
            self.gamma = pos[1]
        if beta2:
            self.beta1 = pos[0]
            self.gamma = pos[1]
            self.beta2 = pos[2]
            if day_mudar==None:
                self.day_mudar = pos[3]
            else:
                self.day_mudar = day_mudar
        self.rmse = cost
        self.optimize = optimizer
            
    def predict(self,x):
        ''' x = dias passados do dia inicial 1'''
        if self.beta_variavel:
            S,I,R = self.__cal_EDO_2(x,self.beta1,self.gamma,self.beta2,self.day_mudar)
        else:
            S,I,R = self.__cal_EDO(x,self.beta,self.gamma)
        self.ypred = I+R
        self.S = S
        self.I = I
        self.R = R         
        return self.ypred

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
        if self.beta_variavel:
            return ['beta1','beta2','gamma','dia_mudanca'],[self.beta1,self.beta2,self.gamma,self.day_mudar]
        return ['beta','gamma'], [self.beta,self.gamma]

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
                    soma[i]= (((y-(Nw))/y)**2).mean()*(1-self.pesoMorte)+(((d-(D))/d)**2).mean()*self.pesoMorte
            elif self.beta_variavel:
                for i in range(tam2):
                    S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],self.day_mudar,coef[i,3],coef[i,4],coef[i,5],coef[i,6],coef[i,7],coef[i,8])
                    soma[i]= (((y-(Nw))/y)**2).mean()*(1-self.pesoMorte)+(((d-(D))/d)**2).mean()*self.pesoMorte
            else:
                for i in range(tam2):
                    S,E,IA,IS,H,U,R,D,Nw = self.__cal_EDO(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3],coef[i,4],coef[i,5],coef[i,6],coef[i,7])
                    soma[i]= (((y-(Nw))/y)**2).mean()*(1-self.pesoMorte)+(((d-(D))/d)**2).mean()*self.pesoMorte
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
    def fit(self, x,y,d,pesoMorte = 0.5, kappa = 1/4,p=0.2,gammaA=1/3.5,gammaS=1/4,muH = 0.15,muU=0.4,xi = 0.53,omegaU = 0.29,omegaH=0.14 , bound = [[0,1/8,1/12,0,0.05],[2,1/4,1/3,0.7,0.25]],stand_error=False, beta2=True,day_mudar = None,paramPSO = {'options':{'c1': 0.1, 'c2': 0.3, 'w': 0.9,'k':5,'p':2},'n_particles':300,'iter':1000}):
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

        optimizer = None
        if bound==None:
            if (beta2) & (day_mudar==None):
                optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=10, options=paramPSO['options'])
            elif beta2:
                optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=9, options=paramPSO['options'])
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=8, options=paramPSO['options'])                
        else:
            if (beta2) & (day_mudar==None):
                if len(bound[0])==8:
                    bound = (bound[0].copy(),bound[1].copy())
                    bound[0].insert(1,bound[0][0])
                    bound[1].insert(1,bound[1][0])
                    bound[0].insert(2,x[4])
                    bound[1].insert(2,x[-5])

                    
                optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=10, options=paramPSO['options'],bounds=bound)
            elif beta2:
                if len(bound[0])==8:
                    bound = (bound[0].copy(),bound[1].copy())
                    bound[0].insert(1,bound[0][0])
                    bound[1].insert(1,bound[1][0])
                    
                optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=9, options=paramPSO['options'],bounds=bound)
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=8, options=paramPSO['options'],bounds=bound)
                
        cost = pos = None
        #__cal_EDO(self,x,beta,gammaH,gammaU,delta,h,ia0,is0,e0)
        #__cal_EDO_2(self,x,beta1,beta2,tempo,gammaH,gammaU,delta,h,ia0,is0,e0)
        if beta2:
            cost, pos = optimizer.optimize(self.objectiveFunction,paramPSO['iter'], x = x,y=df,d=dd,stand_error=stand_error,n_processes=self.numeroProcessadores)
        else:
            cost, pos = optimizer.optimize(self.objectiveFunction, paramPSO['iter'], x = x,y=df,d=dd,stand_error=stand_error,n_processes=self.numeroProcessadores)
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
        return (((y - ypred)**2)/y)*(1-self.pesoMorte) + (((d-dpred)**2)/(d))*self.pesoMorte
    
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


class SEIR:
    ''' SIR Model'''
    def __init__(self,tamanhoPop,numeroProcessadores=None):
        self.N = tamanhoPop
        self.beta = None
        self.gamma = None
        self.mu = None
        self.sigma = None
        self.numeroProcessadores = numeroProcessadores
    
    def __cal_EDO(self,x,beta,gamma,mu,sigma):
            ND = len(x)-1
            t_start = 0.0
            t_end = ND
            t_inc = 1
            t_range = np.arange(t_start, t_end + t_inc, t_inc)
            #beta = np.array(beta)
            #gamma = np.array(gamma)
            #mu = np.array(mu)
            #sigma = np.array(sigma)
            
            def SEIR_diff_eqs(INP, t, beta, gamma,mu,sigma):
                Y = np.zeros((4))
                V = INP
                Y[0] = mu - beta * V[0] * V[2] - mu * V[0]  # Susceptile
                Y[1] = beta * V[0] * V[2] - sigma * V[1] - mu * V[1] # Exposed
                Y[2] = sigma * V[1] - gamma * V[2] - mu * V[2] # Infectious
                Y[3] = gamma * V[2] #recuperado
                return Y   # For odeint

                return Y
            result_fit = spi.odeint(SEIR_diff_eqs, (self.S0,self.E0, self.I0,self.R0), t_range,
                                    args=(beta, gamma,mu,sigma))
            
            S=result_fit[:, 0]*self.N
            E=result_fit[:, 1]*self.N
            I=result_fit[:, 2]*self.N
            R=result_fit[:, 3]*self.N
            
            return S,E,I,R
      
    def __cal_EDO_2(self,x,beta1,beta2,day_mudar,gamma,mu,sigma):
            ND = len(x)-1
            t_start = 0.0
            t_end = ND
            t_inc = 1
            t_range = np.arange(t_start, t_end + t_inc, t_inc)
            #beta1 = np.array(beta1)
            #beta2 = np.array(beta2)
            #gamma = np.array(gamma)
            #mu = np.array(mu)
            #sigma = np.array(sigma)
            def Hf(t):
                h = 1.0/(1.0+ np.exp(-2.0*50*t))
                return h
            def beta(t,t1,b,b1):
                beta = b*Hf(t1-t) + b1*Hf(t-t1) 
                return beta
            def SEIR_diff_eqs(INP, t, beta1,beta2,t1, gamma,mu,sigma):
                Y = np.zeros((4))
                V = INP
                Y[0] = mu - beta(t,t1,beta1,beta2) * V[0] * V[2] - mu * V[0]  # Susceptile
                Y[1] = beta(t,t1,beta1,beta2) * V[0] * V[2] - sigma * V[1] - mu * V[1] # Exposed
                Y[2] = sigma * V[1] - gamma * V[2] - mu * V[2] # Infectious
                Y[3] = gamma * V[2] #recuperado
                return Y   # For odeint

                return Y
            result_fit = spi.odeint(SEIR_diff_eqs, (self.S0,self.E0, self.I0,self.R0), t_range,
                                    args=(beta1,beta2,day_mudar, gamma,mu,sigma))
            
            S=result_fit[:, 0]*self.N
            E=result_fit[:, 1]*self.N
            I=result_fit[:, 2]*self.N
            R=result_fit[:, 3]*self.N
            
            return S,E,I,R
    def __objectiveFunction(self,coef,x ,y,stand_error):
        tam2 = len(coef[:,0])
        soma = np.zeros(tam2)
        #__cal_EDO(self,x,beta,gamma,mu,sigma)
        #__cal_EDO2(self,x,beta1,beta2,day_mudar,gamma,mu,sigma)
        if stand_error:
            if (self.beta_variavel) & (self.day_mudar==None):
                for i in range(tam2):
                    S,E,I,R = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3],self.mu,coef[i,4])
                    soma[i]= (((y-(I+R))/y)**2).mean()
            elif self.beta_variavel:
                for i in range(tam2):
                    S,E,I,R = self.__cal_EDO_2(x,coef[i,0],coef[i,1],self.day_mudar,coef[i,2],self.mu,coef[i,3])
                    soma[i]= (((y-(I+R))/y)**2).mean()
            else:
                for i in range(tam2):
                    S,E,I,R = self.__cal_EDO(x,coef[i,0],coef[i,1],self.mu,coef[i,2])
                    soma[i]= (((y-(I+R))/y)**2).mean()
        else:
            if (self.beta_variavel) & (self.day_mudar==None):
                for i in range(tam2):
                    S,E,I,R = self.__cal_EDO_2(x,coef[i,0],coef[i,1],coef[i,2],coef[i,3],self.mu,coef[i,4])
                    soma[i]= (((y-(I+R)))**2).mean()
            elif self.beta_variavel:
                for i in range(tam2):
                    S,E,I,R = self.__cal_EDO_2(x,coef[i,0],coef[i,1],self.day_mudar,coef[i,2],self.mu,coef[i,3])
                    soma[i]= (((y-(I+R)))**2).mean()
            else:
                for i in range(tam2):
                    S,E,I,R = self.__cal_EDO(x,coef[i,0],coef[i,1],self.mu,coef[i,2])
                    soma[i]= (((y-(I+R)))**2).mean()
        return soma
    

    def fit(self, x,y , bound = ([0,1/7,1/6],[1.5,1/4,1/4]) ,stand_error=False, beta2=True,day_mudar = None):
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound)
        '''
        
        self.y = y
        self.I0 = np.array(y[0])/self.N
        self.S0 = 1-self.I0
        self.R0 = 0
        self.E0 = 0
        self.mu = 1/(75.51*365)
        # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # if bound==None:
        #     optimizer = ps.single.GeneralOptimizerPSO(n_particles=50, dimensions=3, options=options,topology=Star())
        #     cost, pos = optimizer.optimize(self.__objectiveFunction, 500, x = x,y=y,mu=1/(75.51*365),n_processes=self.numeroProcessadores)
        #     self.beta = pos[0]
        #     self.gamma = pos[1]
        #     self.mu = 1/(75.51*365)
        #     self.sigma = pos[2]
        #     self.x = x
        #     self.rmse = cost
        #     self.optimize = optimizer
            
        # else:
        #     optimizer = ps.single.GeneralOptimizerPSO(n_particles=50, dimensions=3, options=options,bounds=bound,topology=Star())
        #     cost, pos = optimizer.optimize(self.__objectiveFunction, 500, x = x,y=y,mu=1/(75.51*365),n_processes=self.numeroProcessadores)
        #     self.beta = pos[0]
        #     self.gamma = pos[1]
        #     self.mu = 1/(75.51*365)
        #     self.sigma = pos[2]
        #     self.x = x
        #     self.rmse = cost
        #     self.optimize = optimizer
        self.beta_variavel = beta2
        self.day_mudar = day_mudar
        self.y = y
        self.x = x

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9,'k':5,'p':1}
        optimizer = None
        
        if bound==None:
            if (beta2) & (day_mudar==None):
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=5, options=options)
            elif beta2:
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=4, options=options)
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=3, options=options)                
        else:
            if (beta2) & (day_mudar==None):
                if len(bound[0])==3:
                    bound = (bound[0].copy(),bound[1].copy())
                    bound[0].insert(1,bound[0][0])
                    bound[1].insert(1,bound[1][0])
                    bound[0].insert(2,x[4])
                    bound[1].insert(2,x[-5])

                    
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=5, options=options,bounds=bound)
            elif beta2:
                if len(bound[0])==3:
                    bound = (bound[0].copy(),bound[1].copy())
                    bound[0].insert(1,bound[0][0])
                    bound[1].insert(1,bound[1][0])
                    
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=4, options=options,bounds=bound)
            else:
                optimizer = ps.single.LocalBestPSO(n_particles=50, dimensions=3, options=options,bounds=bound)
                
        cost = pos = None
        if beta2:
            cost, pos = optimizer.optimize(self.__objectiveFunction, 500, x = x,y=y,stand_error=stand_error,n_processes=self.numeroProcessadores)
        else:
            cost, pos = optimizer.optimize(self.__objectiveFunction, 500, x = x,y=y,stand_error=stand_error,n_processes=self.numeroProcessadores)
            self.beta = pos[0]
            self.gamma = pos[1]
            self.sigma = pos[2]
            
        if beta2:
            self.beta1 = pos[0]
            self.beta2 = pos[1]
            
            if day_mudar==None:
                self.day_mudar = pos[2]
                self.gamma = pos[3]
                self.sigma = pos[4]
            else:
                self.day_mudar = day_mudar
                self.gamma = pos[2]
                self.sigma = pos[3]

        self.rmse = cost
        self.optimize = optimizer
            
    def predict(self,x):
        ''' x = dias passados do dia inicial 1'''
        
        if self.beta_variavel:
            S,E,I,R = self.__cal_EDO_2(x,self.beta1,self.beta2,self.day_mudar,self.gamma,self.mu,self.sigma)
        else:
            S,E,I,R = self.__cal_EDO(x,self.beta,self.gamma,self.mu,self.sigma)
        self.ypred = I+R
        self.S = S
        self.E = E
        self.I = I
        self.R = R         
        return self.ypred
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
        #__cal_EDO(self,x,beta,gamma,mu,sigma)
        #__cal_EDO2(self,x,beta1,beta2,day_mudar,gamma,mu,sigma)
        if self.beta_variavel:
            return ['beta1','beta2','dia_mudanca','gamma','mu','sigma'],[self.beta1,self.beta2,self.day_mudar,self.gamma,self.mu,self.sigma]
        return ['beta','gamma','mu','sigma'],[self.beta,self.gamma,self.mu,self.sigma]

        
class SEIR_GA:
    
    def __init__(self,N):
        """
        Parameters
        ----------
        N : int
            População Inicial
        """
        self.N = N
        self.R = 0     # 5  R = removed
        self.D = 0     # 3  D = publicPerception
        self.C = 0     # 4  C = cumulativeCases
        self.I = 1     # 1  I = infectious
        self.E = 2*self.I   # 2  E = exposed
        self.S = 0.9*self.N # 0  S = susceptible
        self.R = 0     # removed


    def SEIR_diff_eqs(self,INP,t, beta0, alpha, kappa, gamma, sigma, lamb,mu,d):
        '''
        The main set of equations
        '''
        Y=np.zeros((7))
        V = INP    
        beta = beta0*(1-alpha)*(1 -self.D/self.N)**kappa
        Y[0] = - beta * V[0] * V[1]/self.N  - mu* V[0] * V[1]  #Susceptibles
        Y[1] = sigma * V[2] - (gamma + mu)*V[1]            #Infectious 
        Y[2] = beta * V[0] * V[1]/self.N  - (sigma + mu) * V[2] #exposed
        Y[3] = d*gamma * V[1] - lamb * V[3]                #publicPerception
        Y[4] = -sigma * V[2]                               #cumulativeCases
        Y[5] = gamma * V[1] - mu*V[5]                      #Removed
        Y[6] = mu * V[6]                                  #Population size
      
        self.Y = Y
      
        return Y   # For odeint


    def fitness_function(self, x, y, Model_Input, t_range):
      
        beta0 = x[0]     
        alpha = x[1]     
        kappa = x[2]     
        gamma = x[3]     
        sigma = x[4]     
        lamb  = x[5]     
        mu = x[6]     
        d  = x[7]        

        result = spi.odeint(self.SEIR_diff_eqs,Model_Input,
                           t_range,args=(beta0, alpha, kappa ,gamma, sigma, lamb,mu,d))


        mean_squared_error = ((np.array(y)-result[:,1])**2).mean()    

        return [mean_squared_error]

    def fit(self, x,y ,bound = None,name = None):
        self.name = name
        self.y=np.array(y)
        self.x = x
        
        TS = 1
        ND = len(y) - 1
    
        t_start = 0.0
        t_end = ND
        t_inc = TS
        t_range = np.arange(t_start, t_end + t_inc, t_inc)
    
        INPUT = (self.S, self.I, self.E, self.D, self.C, self.R, self.N)

        input_variables = ['beta0', 'alpha', 'kappa', 'gamma', 'sigma',
                           'lamb', 'mu','d']
    
        # GA Parameters
        number_of_generations = 1000
        ga_population_size = 100
        number_of_objective_targets = 1
        number_of_constraints = 0
        number_of_input_variables = len(input_variables)

        problem = Problem(number_of_input_variables,number_of_objective_targets,number_of_constraints)

        problem.types[0] = Real(0, 2)           #beta0
        problem.types[1] = Real(0, 2)           #alpha
        problem.types[2] = Real(0, 2000)           #kappa
        problem.types[3] = Real(0, 2)           #gamma
        problem.types[4] = Real(0, 2)           #sigma
        problem.types[5] = Real(0, 2)           #lamb
        problem.types[6] = Real(0, 2)           #mu
        problem.types[7] = Real(0, 2)           #d


        problem.function = functools.partial(self.fitness_function,
                                             y=y, Model_Input=INPUT,
                                             t_range=t_range)
        algorithm = NSGAII(problem, population_size = ga_population_size)
        algorithm.run(number_of_generations)

        feasible_solutions = [s for s in algorithm.result if s.feasible]


        self.beta0 = feasible_solutions[0].variables[0]      
        self.alpha = feasible_solutions[0].variables[1]         
        self.kappa = feasible_solutions[0].variables[2]    
        self.gamma = feasible_solutions[0].variables[3]         
        self.sigma = feasible_solutions[0].variables[4]     
        self.lamb  = feasible_solutions[0].variables[5] 
        self.mu = feasible_solutions[0].variables[6]       
        self.d  = feasible_solutions[0].variables[7]  

        file_address = 'optimised_coefficients/'
        filename = "ParametrosAjustados_Modelo_{}_{}_{}_Dias.txt".format('SEIR_EDO',name,len(x))
        if not os.path.exists(file_address):
            os.makedirs(file_address)
        file_optimised_parameters = open(file_address+filename, "w")
        file_optimised_parameters.close()
       
        with open(file_address+filename, "a") as file_optimised_parameters:
            for i in range(len(input_variables)):
                message ='{}:{:.4f}\n'.format(input_variables[i],feasible_solutions[0].variables[i])    
                file_optimised_parameters.write(message)
                
        result_fit = spi.odeint(self.SEIR_diff_eqs,
                                    (self.S, self.I, self.E, self.D, self.C, 
                                     self.R, self.N),t_range,
                                args=(self.beta0, self.alpha, self.kappa,
                                      self.gamma, self.sigma, self.lamb,
                                      self.mu,self.d))
                                    
        plt.plot(result_fit[:, 1],c='b',label='Predição Infectados')
        plt.plot(self.y,c='r',marker='o', markersize=3,label='Infectados')
        plt.legend(fontsize=15)
        plt.ylabel('Casos COnfirmados',fontsize=15)
        plt.xlabel('Dias',fontsize=15)
        plt.show()

            
    def predict(self,x):
        """
        Parameters
        ----------
        x : int
            Número de dias para a predição desde o primeiro caso (Dia 1)
        """
        
        if (self.beta0 == None or self.lamb == None):
            
            print('The model needs to be fitted before predicting\n\n')
            return 0
            
        else:
        
            ND = len(x)+1
            t_start = 0.0
            t_end = ND
            t_inc = 1
            t_range = np.arange(t_start, t_end + t_inc, t_inc)
            result_fit = spi.odeint(self.SEIR_diff_eqs,
                                    (self.S, self.I, self.E, self.D, self.C, 
                                     self.R, self.N),t_range,
                                args=(self.beta0, self.alpha, self.kappa,
                                      self.gamma, self.sigma, self.lamb,
                                      self.mu,self.d))

            self.ypred = result_fit[:, 1]

            return result_fit[:, 1]
        
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
        
    def plot(self,local):
        plt.plot(self.ypred,c='b',label='Predição Infectados')
        plt.plot(self.y,c='r',marker='o', markersize=3,label='Infectados')
        plt.legend(fontsize=15)
        plt.title('Dinâmica do CoviD19 - {}'.format(local),fontsize=20)
        plt.ylabel('Casos COnfirmados',fontsize=15)
        plt.xlabel('Dias',fontsize=15)
        plt.show()
        
    def getCoef(self):
        
        return ['beta0','alpha', 'kappa', 'gamma','sigma','lamb', 'mu',
                         'd',['Susceptibles','Infectious','exposed','publicPerception','cumulativeCases','Removed','Population_size']],[self.beta0, self.alpha, self.kappa,self.gamma, 
                          self.sigma, self.lamb,self.mu,self.d,self.Y]
        


class SEQIJR_GA:

   def __init__(self,N):
       """
       Parameters
       ----------
       N : int
           População Inicial
       """

       self.N = N
       
       self.I0 = 1  # Infectious
       # An infectious person is symptomatic
   
       self.E0 = self.I0*1.5  # Exposed
       # An exposed person is someone who has come into contact
       # with an infectious person but is asymptomatic
   
       self.S0 = N-self.I0  # Susceptible
       # A susceptible person is an uninfected person who can
       # be infected through contact with an infectious or exposed
       # person
   
       self.Q0 = 0  # Quarantined
       # A quarantined person is an exposed person who is removed
       # from contact with the general population
   
       self.J0 = 0  # Isolated
       # an isolated person is an infectious person who
       # is removed from contact with the general population,
       # usually by being admitted to a hospital.
   
       self.R0 = 0  # Recovered
       # A recovered person is someone who has recovered
       # from the disease
   
       self.N0 = 0  # Population size ???
   
       self.D0 = 0  # Death Rate ???


   def SEQIJR_diff_eqs(self,INP, t, beta, epsilon_E, epsilon_Q, epsilon_J, Pi,
                       mu, v, gamma1, gamma2, kappa1, kappa2, d1, d2, sigma1, 
                       sigma2, DS, DE, DI, DJ, DQ):
       '''The main set of equations'''
       Y = np.zeros((8))
       V = INP
       L = beta * (V[3] + epsilon_E * V[1] + epsilon_Q * V[2] + epsilon_J * V[4])
       Y[0] = Pi - L * V[0] / self.N - DS * V[0]  # (1) 
       Y[1] = L * V[0] / self.N - DE * V[1]  # (2)
       Y[2] = gamma1 * V[1] - DQ * V[2]  # (3)
       Y[3] = kappa1 * V[1] - DI * V[3]  # (4)
       Y[4] = gamma2 * V[3] + kappa2 * V[2] - DJ * V[4]  # (5)
       Y[5] = v * V[0] + sigma1 * V[3] + sigma2 * V[4] - mu * V[5]  # (6)
       Y[6] = Pi - d1 * V[3] - d2 * V[4] - mu * V[6]  # (7)
       Y[7] = d1 * V[3] + d2 * V[4]
       return Y


   def fitness_function(self, x, y, Model_Input, t_range):
   
       beta = x[0]  # Infectiousness and contact rate between a susceptible and an infectious individual
       epsilon_E = x[1]  # Modification parameter associated with infection from an exposed asymptomatic individual
       epsilon_Q = x[2]  # Modification parameter associated with infection from a quarantined individual
       epsilon_J = x[3]  # Modification parameter associated with infection from an isolated individual
       Pi = x[4]  # Rate of inflow of susceptible individuals into a region or community through birth or migration.
       mu = x[5]  # The natural death rate for disease-free individuals
       v = x[6]  # Rate of immunization of susceptible individuals
       gamma1 = x[7]  # Rate of quarantine of exposed asymptomatic individuals
       gamma2 = x[8]  # Rate of isolation of infectious symptomatic individuals
       kappa1 = x[9]  # Rate of development of symptoms in asymptomatic individuals
       kappa2 = x[10]  # Rate of development of symptoms in quarantined individuals
       d1 = x[11]  # Rate of disease-induced death for symptomatic individuals
       d2 = x[12]  # Rate of disease-induced death for isolated individuals
       sigma1 = x[13]  # Rate of recovery of symptomatic individuals
       sigma2 = x[14]  # Rate of recovery of isolated individuals

       DS = mu + v
       DE = gamma1 + kappa1 + mu
       DI = gamma2 + d1 + sigma1 + mu
       DJ = sigma2 + d2 + mu
       DQ = mu + kappa2

       result = spi.odeint(self.SEQIJR_diff_eqs, Model_Input, t_range, 
                           args=(beta, epsilon_E, epsilon_Q, epsilon_J,
                            Pi, mu, v, gamma1, gamma2, kappa1, kappa2, d1,
                            d2, sigma1, sigma2, DS, DE, DI, DJ, DQ))

       mean_squared_error = ((np.array(y) - result[:, 3]) ** 2).mean()
       
#       print(mean_squared_error)
#       plt.plot(result[:, 3]/self.N,c='r')
#       plt.plot(np.array(y),c='b')
#       plt.show()

       return [mean_squared_error]

   def fit(self, x,y ,bound = None,name = None):
       self.name = name
       self.y=np.array(y)
       self.x = x
        
       TS = 1
       ND = len(y) - 1
       
   
       t_start = 0.0
       t_end = ND
       t_inc = TS
       t_range = np.arange(t_start, t_end + t_inc, t_inc)
   
       Model_Input = (self.S0,self.E0,self.Q0,self.I0,self.J0, \
                      self.R0,self.N0,self.D0)
   
       input_variables = ['beta', 'epsilon_E', 'epsilon_Q', 'epsilon_J',
                          'Pi', 'mu', 'v', 'gamma1', 'gamma2', 'kappa1',
                          'kappa2', 'd1', 'd2', 'sigma1', 'sigma2']

       number_of_generations = 1000
       ga_population_size = 100
       number_of_objective_targets = 1
       number_of_constraints = 0
       number_of_input_variables = len(input_variables)
   
       problem = Problem(number_of_input_variables, number_of_objective_targets, number_of_constraints)
   
       problem.types[0] = Real(0,0.4)  # beta      - Infectiousness and contact rate between a susceptible and an infectious individual
       problem.types[1] = Real(0,0.5)  # epsilon_E - Modification parameter associated with infection from an exposed asymptomatic individual
       problem.types[2] = Real(0,0.5)  # epsilon_Q - Modification parameter associated with infection from a quarantined individual
       problem.types[3] = Real(0,1)  # epsilon_J - Modification parameter associated with infection from an isolated individual
       problem.types[4] = Real(0,500)  # Pi        - Rate of inflow of susceptible individuals into a region or community through birth or migration.
       problem.types[5] = Real(0, 0.00005)  # mu        - The natural death rate for disease-free individuals
       problem.types[6] = Real(0, 0.1)  # v         - Rate of immunization of susceptible individuals
       problem.types[7] = Real(0, 0.3)  # gamma1    - Rate of quarantine of exposed asymptomatic individuals
       problem.types[8] = Real(0, 0.7)  # gamma2    - Rate of isolation of infectious symptomatic individuals
       problem.types[9] = Real(0, 0.3)  # kappa1    - Rate of development of symptoms in asymptomatic individuals
       problem.types[10] = Real(0, 0.3)  # kappa2    - Rate of development of symptoms in quarantined individuals
       problem.types[11] = Real(0, 0.1)  # d1        - Rate of disease-induced death for symptomatic individuals
       problem.types[12] = Real(0, 0.1)  # d2        - Rate of disease-induced death for isolated individuals
       problem.types[13] = Real(0, 0.1)  # sigma1    - Rate of recovery of symptomatic individuals
       problem.types[14] = Real(0, 0.1)  # sigma2    - Rate of recovery of isolated individuals
   
       problem.function = functools.partial(self.fitness_function,        
                                            y=y,                           
                                            Model_Input=Model_Input,  
                                            t_range=t_range)          
       algorithm = NSGAII(problem, population_size=ga_population_size)
       algorithm.run(number_of_generations)
       
       feasible_solutions = [s for s in algorithm.result if s.feasible]    
       
       self.beta = feasible_solutions[0].variables[0]  # Infectiousness and contact rate between a susceptible and an infectious individual
       self.epsilon_E = feasible_solutions[0].variables[1]  # Modification parameter associated with infection from an exposed asymptomatic individual
       self.epsilon_Q = feasible_solutions[0].variables[2]  # Modification parameter associated with infection from a quarantined individual
       self.epsilon_J = feasible_solutions[0].variables[3]  # Modification parameter associated with infection from an isolated individual
       self.Pi = feasible_solutions[0].variables[4]  # Rate of inflow of susceptible individuals into a region or community through birth or migration.
       self.mu = feasible_solutions[0].variables[5]  # The natural death rate for disease-free individuals
       self.v = feasible_solutions[0].variables[6]  # Rate of immunization of susceptible individuals
       self.gamma1 = feasible_solutions[0].variables[7]  # Rate of quarantine of exposed asymptomatic individuals
       self.gamma2 = feasible_solutions[0].variables[8]  # Rate of isolation of infectious symptomatic individuals
       self.kappa1 = feasible_solutions[0].variables[9]  # Rate of development of symptoms in asymptomatic individuals
       self.kappa2 = feasible_solutions[0].variables[10]  # Rate of development of symptoms in quarantined individuals
       self.d1 = feasible_solutions[0].variables[11]  # Rate of disease-induced death for symptomatic individuals
       self.d2 = feasible_solutions[0].variables[12]  # Rate of disease-induced death for isolated individuals
       self.sigma1 = feasible_solutions[0].variables[13]  # Rate of recovery of symptomatic individuals
       self.sigma2 = feasible_solutions[0].variables[14]  # Rate of recovery of isolated individuals
       
       self.DS = self.mu + self.v
       self.DE = self.gamma1 + self.kappa1 + self.mu
       self.DI = self.gamma2 + self.d1 + self.sigma1 + self.mu
       self.DJ = self.sigma2 + self.d2 + self.mu
       self.DQ = self.mu + self.kappa2
       
       file_address = 'optimised_coefficients/'
       filename = "ParametrosAjustados_Modelo_{}_{}_{}_Dias.txt".format('SEQIJR_EDO',name,len(x))
       if not os.path.exists(file_address):
           os.makedirs(file_address)
       file_optimised_parameters = open(file_address+filename, "w")
       file_optimised_parameters.close()
       
       with open(file_address+filename, "a") as file_optimised_parameters:
           for i in range(len(input_variables)):
               message ='{}:{:.4f}\n'.format(input_variables[i],feasible_solutions[0].variables[i])    
               file_optimised_parameters.write(message)
       
       

#       
#       result_fit = spi.odeint(self.SEQIJR_diff_eqs, (self.S0,self.E0,
#                                                          self.Q0,self.I0,self.J0,
#                      self.R0,self.N0,self.D0), t_range,args=(self.beta, 
#                       self.epsilon_E, self.epsilon_Q, self.epsilon_J, 
#                       self.Pi, self.mu, self.v, self.gamma1, self.gamma2,
#                       self.kappa1, self.kappa2, self.d1, self.d2, 
#                       self.sigma1, self.sigma2, self.DS, self.DE,
#                       self.DI, self.DJ, self.DQ))
#       
#       alphaE = self.DI / self.kappa1
#       alphaS = self.DE * alphaE
#       alphaQ = (self.gamma1 / self.DQ) * alphaE
#       alphaJ = (self.gamma2 + self.kappa2 * alphaQ) / self.DJ
##       alphaN = self.d1 + self.d2 * alphaJ
##        alphaR = (1 / self.mu) * ((self.v * alphaS / self.DS) - 
##                  self.sigma1 - self.sigma2 * alphaJ)
#       alphaL = self.beta * (1 + self.epsilon_E * alphaE + \
#                             self.epsilon_Q * alphaQ + self.epsilon_J  * \
#                             alphaJ)
   
#        I2 = (self.Pi * (self.mu * alphaL - self.DS * 
#                        alphaS)) / (self.alphaS * (self.mu *
#                                     alphaL - self.DS * alphaN))
   
#        S2 = (1 / self.DS) * (self.Pi - alphaS * I2)
   
#        E2 = alphaE * I2
   
#        J2 = alphaJ * I2
   
#        N2 = (1 / self.mu) * (self.Pi - alphaN * I2)
   
#        Q2 = alphaQ * I2
   
#        R2 = ((self.v * self.Pi) / (self.mu * self.DS)) - alphaR * I2
   
#       Rdf = (self.mu * alphaL) / (self.DS * alphaS)
   
#       R0 = alphaL / alphaS
       
#       print('Rdf = {:.4f}, R0 = {:.4f}'.format(Rdf, R0))
       

           
   def predict(self,x):
       """
       Parameters
       ----------
       x : int
           Número de dias para a predição desde o primeiro caso (Dia 1)
       """
       
       if (self.beta == None):
           
           print('The model needs to be fitted before predicting\n\n')
           return 0
           
       else:
       
           ND = len(x)+1
           t_start = 0.0
           t_end = ND
           t_inc = 1
           t_range = np.arange(t_start, t_end + t_inc, t_inc)
           result_fit = spi.odeint(self.SEQIJR_diff_eqs, (self.S0,self.E0,
                                                          self.Q0,self.I0,self.J0,
                      self.R0,self.N0,self.D0), t_range,args=(self.beta, 
                       self.epsilon_E, self.epsilon_Q, self.epsilon_J, 
                       self.Pi, self.mu, self.v, self.gamma1, self.gamma2,
                       self.kappa1, self.kappa2, self.d1, self.d2, 
                       self.sigma1, self.sigma2, self.DS, self.DE,
                       self.DI, self.DJ, self.DQ))
           
           self.ypred = result_fit[:, 3]

           return result_fit[:, 3]
       
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
       
   def plot(self,local):
        plt.plot(self.ypred,c='b',label='Predição Infectados')
        plt.plot(self.y,c='r',marker='o', markersize=3,label='Infectados')
        plt.legend(fontsize=15)
        plt.title('Dinâmica do CoviD19 - {}'.format(local),fontsize=20)
        plt.ylabel('Casos COnfirmados',fontsize=15)
        plt.xlabel('Dias',fontsize=15)
        plt.show()
        
   def getCoef(self):
        
       
        return ['beta','epsilon_E', 'epsilon_Q', 'epsilon_J','Pi', 'mu', 'v',
                'gamma1','gamma2', 'kappa1','kappa2', 'd1', 'd2', 'sigma1', 
                'sigma2'],[self.beta,self.epsilon_E, self.epsilon_Q, self.epsilon_J, 
                self.Pi, self.mu, self.v, self.gamma1, self.gamma2,self.kappa1,
                self.kappa2, self.d1, self.d2,self.sigma1, self.sigma2]
 
                           
