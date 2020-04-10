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
import multiprocessing as mp
import logging

logging.disable()

class SIR_GA:

    def __init__(self,N):
        """
        Parameters
        ----------
        N : int
            População Inicial
        """
        self.N = N


    
    def SIR_diff_eqs(self,INP, t, beta, gamma):
        '''The main set of equations'''
        
        
        Y = np.zeros((3))
        V = INP
        Y[0] = - beta * V[0] * V[1]                 #S
        Y[1] = beta * V[0] * V[1] - gamma * V[1]    #I
        Y[2] = gamma * V[1]                         #R
        self.Y = Y
        return Y


    def fitness_function(self, x, y, Model_Input, t_range):
    
        mean_squared_error = 0

        beta = x[0]
        gamma = x[1]
        
        result = spi.odeint(self.SIR_diff_eqs, Model_Input, t_range,
                            args=(beta, gamma))

        mean_squared_error = ((np.array(y) - (result[:, 1] + result[:, 2])) ** 2).mean()
    
        return [mean_squared_error]

    def fit(self, x,y ,bound = ([0,1/21-0.0001],[1,1/5+0.0001]),name = None):
        
        self.y = np.array(y)
        self.x = x
        
        TS = 1
        ND = len(y) - 1
        
        y = self.y/self.N
    
        t_start = 0.0
        t_end = ND
        t_inc = TS
        t_range = np.arange(t_start, t_end + t_inc, t_inc)
        self.I0 = y[0]
        self.S0 = 1-self.I0
        self.R0 = 0
        self.beta = None
        self.gamma = None
    
        Model_Input = (self.S0, self.I0, self.R0)
    
        # GA Parameters
        number_of_generations = 10000
        ga_population_size = 300
        number_of_objective_targets = 1  # The MSE
        number_of_constraints = 0
        number_of_input_variables = 2  # beta and gamma
        problem = Problem(number_of_input_variables, 
                          number_of_objective_targets, number_of_constraints)
        problem.function = functools.partial(self.fitness_function,
                                             y=y, Model_Input=Model_Input,
                                             t_range=t_range)
    
        algorithm = NSGAII(problem, population_size=ga_population_size)
        
        problem.types[0] = Real(bound[0][0], bound[1][0])  # beta initial Range
        problem.types[1] = Real(bound[0][1], bound[1][1])  # gamma initial Range
    
        # Running the GA
        algorithm.run(number_of_generations)

        feasible_solutions = [s for s in algorithm.result if s.feasible]    
        
        self.beta = feasible_solutions[0].variables[0]
        self.gamma = feasible_solutions[0].variables[1]
        
        input_variables = ['beta','gamma']
        file_address = 'optimised_coefficients/'
        filename = "ParametrosAjustados_Modelo_{}_{}_{}_Dias.txt".format('SIR_EDO',name,len(x))        

        if not os.path.exists(file_address):
            os.makedirs(file_address)
        

        file_optimised_parameters = open(file_address+filename, "w")
        file_optimised_parameters.close()
        if not os.path.exists(file_address):
            os.makedirs(file_address)
        with open(file_address+filename, "a") as file_optimised_parameters:
            for i in range(len(input_variables)):
                message ='{}:{:.4f}\n'.format(input_variables[i],feasible_solutions[0].variables[i])    
                file_optimised_parameters.write(message)
        
            
    def predict(self,x, ci = False):
        """
        Parameters
        ----------
        x : int
            Número de dias para a predição desde o primeiro caso (Dia 1)
        """
        
        if (self.beta == None or self.gamma == None):
            
            print('The model needs to be fitted before predicting\n\n')
            return 0
            
        else:
        
            ND = len(x)+1
            t_start = 0.0
            t_end = ND
            t_inc = 1
            t_range = np.arange(t_start, t_end + t_inc, t_inc)
            result_fit = spi.odeint(self.SIR_diff_eqs, (self.S0, self.I0,
                            self.R0), t_range, args=(self.beta, self.gamma))
            
            self.ypred = (result_fit[:, 1] + result_fit[:, 2])*self.N
            self.S=result_fit[:, 0]*self.N
            self.R=result_fit[:, 2]*self.N
            self.I=result_fit[:, 1]*self.N
            self.rmse = ((self.y-self.ypred[0:len(self.y)])**2).mean()
        if ci == False:
            return self.ypred
        else:
            self.res = {"pred": self.ypred, "I": self.I, "R": self.R, "S":self.S}
            return pd.DataFrame.from_dict(self.res)
    
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
        return ['beta','gamma','R0',('S','I','R')], [self.beta,self.gamma,self.beta/self.gamma,(self.S,self.ypred,self.R)]
    
    
    
    def __bootstratpTS(self,npArray, replicate):
        simList = []
        def poissonGen(npArray, replicate = None):
            simSeries = []
            for i in range(0,len(npArray)):
                if i == 0:
                    simSeries.append(npArray[i]) 
                else:
                    simSeries.append(np.random.poisson(lam = npArray[i] - npArray[i-1], size = 1)[0])
            return np.cumsum(np.array(simSeries))
        for i in range(0,replicate):
            simList.append(poissonGen(npArray))
        return np.array(simList)

    
    
    def runSir(self, y, x, ndays):
        self.y = y
        self.x = x
        newx = range(0, len(x) + ndays) 
        self.fit(y = self.y, x = self.x)
        return self.predict(newx, ci = True)
        
        
    def predictCI(self, y, x, start, ndays, bootstrap, n_jobs):
        """
        This function fits diffent models to data to get confidence interval for I + R.
        y = an array with the series of cases
        x = an range object with the first and last day of cases
        start =  a date in format "YYYY-mm-dd" indicating the day of the first case reported
        ndays = number of days to be predicted
        bootstrap = number of times that the model will run
        n_jobs = number of core to be used to fit the models
        
        """
        
        #Make some parameters avaliable for returnDF
        self.start = start
        self.ndays = len(x) + ndays
        
        
        #Create a lol with data for run the model
        lists = self.__bootstratpTS(npArray = self.y, replicate = bootstrap)
        
        #Model will be fitted and predicted so R) using ci is not consisent
        #Make cores avalible to the process
        pool =  mp.Pool(processes = n_jobs)
        
        #Run the model
        results = pool.starmap(self.runSir, [(lists[i], x, ndays) for i in range(0,len(lists))])
        
        #Create data frames for models
        pred = [results[i]["pred"] for i in range(0,len(results))]
        I = [results[i]["I"] for i in range(0,len(results))]
        S = [results[i]["S"] for i in range(0,len(results))]
        R = [results[i]["R"] for i in range(0,len(results))]
        
        pred = self.__returnDF(pred,"Pred")
        I = self.__returnDF(I,"I")
        R = self.__returnDF(R,"R")
        S = self.__returnDF(S,"S")
        
                        
        self.dfs = reduce(lambda df1, df2: df1.merge(df2, "left"), [pred,I,S,R])
        return self.dfs
    
    
    def __returnDF(self,lol, parName):
        df = pd.DataFrame.from_dict({"date": pd.date_range(start = self.start, periods = self.ndays + 2, freq = "D"),
                                     parName: np.median(lol, axis = 0),
                                     parName + "_lb": np.quantile(lol, q = 0.0275, axis = 0),
                                     parName + "_ub": np.quantile(lol, q = 0.975, axis = 0)})
        return df