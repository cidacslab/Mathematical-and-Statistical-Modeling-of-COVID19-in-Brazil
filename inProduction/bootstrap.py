#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:18:44 2020

@author: Rafael Veiga
@author: matheustorquato matheusft@gmail.com
"""

import numpy as np
import pandas as pd
import multiprocessing.dummy as mp
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf


class bootstrapTS:
    def __init__(self):
        pass
    
     # #Define a function that mimics the behavior for the model class
    def __runSir(self, model, y, x):
        self.newx = range(0, len(self.y) + self.ndays) 
        model.fit(y = y, x = x)
        model.predict(self.newx)
        if self.model_name == "exponencial":
            return {"a": model.a, "b": model.b, "pred": model.ypred}
        elif  self.model_name == "SIR":
            try:
                return{"pred": model.ypred, "I": model.I, "S": model.S, "R": model.R, "beta1":model.beta1,
                           "beta2":model.beta2, "gamma": model.gamma, "changeDay": model.day_mudar}
            except:
                    return {"pred": model.ypred, "I": model.I, "S": model.S, "R": model.R, "beta":model.beta, "gamma": model.gamma}
    
     #Define a auxiliary functions that generate simulations based on the original series
    def __bootstratpPoisson(self, npArray, replicate, cumSum = True):
        simList = []
        def poissonGen(npArray, replicate = None):
            simSeries = []
            for i in range(0,len(npArray)):
                if i == 0:
                    simSeries.append(npArray[i]) 
                else:
                    simSeries.append(np.random.poisson(lam = npArray[i] - npArray[i-1], size = 1)[0])
            if cumSum:
                return np.cumsum(np.array(simSeries))
            else:
                return np.array(simSeries)
        for i in range(0,replicate):
            simList.append(poissonGen(npArray))
        return np.array(simList)

    #simlating from a duble possion
    def __bootstrapDPoisson(self,npArray, replicate):
        simList = []
        poissonsDist = self.__bootstratpPoisson(npArray, replicate, cumSum = False)
        def poissonGen(npArray, replicate = None):
            simSeries = []
            for i in range(0,len(npArray)):
                if i == 0:
                    simSeries.append(npArray[i])
                else:
                    value = np.random.poisson(lam = np.random.choice(poissonsDist[:,i], size = 1), size = 1)[0]
                    #Sanity check
                    if value > 0:
                        simSeries.append(value)
                    else:
                        simSeries.append(np.random.poisson(lam = npArray[i] - npArray[i-1], size = 1)[0])
            return np.cumsum(np.array(simSeries))
        for i in range(0,replicate):
            simList.append(poissonGen(npArray))
        return np.array(simList)
       
    #simlating from a miximg poisson gamma
    def __bootstrapGammaPoisson(self,npArray, replicate):
        simList = []
        poissonsDist = self.__bootstratpPoisson(npArray, replicate, cumSum = False)
        def poissonGen(npArray, replicate = None):
            simSeries = []
            for i in range(0,len(npArray)):
                if i == 0:
                    simSeries.append(npArray[i])
                else:
                    meanP = np.mean(poissonsDist[:,i])
                    varP = np.var(poissonsDist[:,i])
                    if meanP == 0 or varP == 0:
                        value = np.random.poisson(lam = np.random.choice(poissonsDist[:,i], size = 1), size = 1)[0]
                        #Sanity check
                        if value > 0:
                            simSeries.append(value)
                        else:
                            simSeries.append(np.random.poisson(lam = npArray[i] - npArray[i-1], size = 1)[0])
                    else:
                        scale = varP/meanP
                        shape = meanP**2/varP
                        mean = np.random.gamma(shape = shape, scale = scale, size = 1)[0]
                        simSeries.append(np.random.poisson(lam = mean, size = 1)[0])
            return np.cumsum(np.array(simSeries))
        for i in range(0,replicate):
            simList.append(poissonGen(npArray))
        return np.array(simList) 


    def single_core_CI(self, model, y, x, ndays, bootstrap, simulation = "Poisson", method = "percentile", model_name = "exponencial"):
        """
        y: an array with the series of cases
        x: an range object with the first and last day of cases
        ndays: number of days to be forecasted
        bootstrap: number of times that the model will run
        simulation: Choose among Poisson, Mixing_Poisson or Gamma_Poisson.
        method: accepted methods are percentile, basic or approximation
        """
        self.x = x
        self.y = y
        self.ndays = ndays
        self.model_name = model_name

        #Create a lol with data for run the model
        if simulation == "Poisson":
            lists = self.__bootstratpPoisson(npArray = y, replicate = bootstrap)

        elif simulation == "Mixing_Poisson":
            lists = self.__bootstrapDPoisson(npArray = y, replicate = bootstrap)

        elif simulation == "Gamma_Poisson":
            lists = self.__bootstrapGammaPoisson(npArray = y, replicate = bootstrap)

        elif simulation == "Normal":
            pred = self.__runSir(model, y, x)["pred"]
            sigma = np.std(y - pred[0:len(self.y)])
            error = np.sqrt(sigma**2 + 0.01**2)
            lists = []
            for i in range(0, bootstrap):
                delta = np.random.normal(0., error, len(y))
                randomdataY = y + delta
                lists.append(randomdataY)
      

        #create a empty list that will be fulffil with dictionaries
        self.results = []


        #Iterate over the list of simulated data
        for i in range(0,len(lists)):
            self.results.append(self.__runSir(model, lists[i], x))
        
        # # Get predictions (I + R) and all other fitted values
        self.pred = np.array([self.results[i]["pred"] for i in range(0,len(self.results))])
        self.I = np.array([self.results[i]["I"] for i in range(0,len(self.results))])
        self.S = np.array([self.results[i]["S"] for i in range(0,len(self.results))])
        self.R = np.array([self.results[i]["R"] for i in range(0,len(self.results))])


        #Compute means values for predicitons
        self.meanPred = [np.mean(self.pred[:,i]) for i in range(0,len(self.y) + self.ndays)]
        self.meanI = [np.mean(self.I[:,i]) for i in range(0,len(self.y) + self.ndays)]
        self.meanS = [np.mean(self.S[:,i]) for i in range(0,len(self.y) + self.ndays)]
        self.meanR = [np.mean(self.R[:,i]) for i in range(0,len(self.y) + self.ndays)]
        pred = [self.pred[i][:len(self.y)] for i in range(0,len(self.results))]

        #Compute sgima all models
        sigmAllModels = pred - y
        sigmaMean = np.mean(np.std(sigmAllModels))
        self.sigmaMeanError = np.sqrt(sigmaMean ** 2 + 0.001**2)
        self.std_err = np.sqrt((sigmaMean ** 2) + (self.sigmaMeanError)/np.sqrt(len(y)))

      
        if method == "percentile":
            #self.lim_inf = [np.quantile(self.pred[:,i], q = 0.025) for i in range(0,len(self.meanPred))]
            #self.lim_sup = [np.quantile(self.pred[:,i], q = 0.975) for i in range(0,len(self.meanPred))]
            print("Will be fixed in future")
            pass


        elif method == "basic":

            #deltaStar = self.meanPred - self.pred
            #deltaL = [np.quantile(deltaStar[:,i], q = 0.025) for i in range(0,len(self.meanPred))]
            #deltaU = [np.quantile(deltaStar[:,i], q = 0.975) for i in range(0,len(self.meanPred))]
            #self.lim_inf  = self.meanPred + deltaL
            #self.lim_sup  = self.meanPred + deltaU
            print("Will be fixed in future")
            pass
           

        elif method == "approximation":

            percentiles = np.array([0.025, 1.0 - 0.025])
            t_quantiles = stats.t.ppf(percentiles, df = len(self.y))

            errors  = self.pred - self.meanPred
            self.std_err = np.sqrt(np.diag(errors.T.dot(errors)/len(self.y)))

            meanStd = np.mean(self.std_err[:len(self.y)])

            self.lim_inf = self.meanPred[len(self.y):] + (t_quantiles[0] *  np.sqrt((sigmaMean ** 2) + meanStd))
            self.lim_sup = self.meanPred[len(self.y):] + (t_quantiles[1] *  np.sqrt((sigmaMean ** 2) + meanStd))



        if self.model_name == "exponencial":
                 self.a = [self.results[i]["a"] for i in range(0,len(self.results))]
                 self.b = [self.results[i]["b"] for i in range(0,len(self.results))]
                 return self.a, self.b, self.meanPred, self.lim_inf, self.lim_sup


        elif self.model_name == "SIR":
            try:
                self.beta1 = [self.results[i]["beta1"] for i in range(0,len(self.results))]
                self.beta2 = [self.results[i]["beta2"] for i in range(0,len(self.results))]
                self.gamma = [self.results[i]["gamma"] for i in range(0,len(self.results))]
                self.changeDay = [self.results[i]["changeDay"] for i in range(0,len(self.results))]
                return {"beta1": self.beta1, 
                         "beta2": self.beta2, 
                         "dayChange": self.changeDay, 
                         "gamma": self.gamma, 
                         "meanPred": self.meanPred, 
                         "predLB": self.lim_inf, 
                         "predUB": self.lim_sup,
                         "I": self.meanI, 
                         "R": self.meanR, 
                         "S": self.meanS}
            except:
                self.beta = [self.results[i]["beta"] for i in range(0,len(self.results))]
                self.gamma = [self.results[i]["gamma"] for i in range(0,len(self.results))]
                return [self.beta1, self.beta2, self.changeDay, self.gamma, self.meanPred, self.lim_inf, self.lim_sup]
       
  
    
    #Define a function to plot ACF and PACF
    def plot_lagCor(self, nlags = 1):
        resid = np.array(self.y) - np.array(self.meanPred[0:len(self.y)])
    
        #Compute acf and pacf
        lag_acf = acf(resid, nlags = nlags)
        lag_pacf = pacf(resid, nlags = nlags, method = "ols")
        
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18,8))

        axes[0].plot(lag_acf, marker="o")
        axes[1].plot(lag_pacf, marker = "o")
        axes[0].axhline(y=0,linestyle='--',color='gray')
        axes[0].axhline(y=-1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
        axes[0].axhline(y=1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
        axes[1].axhline(y=0,linestyle='--',color='gray')
        axes[1].axhline(y=-1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
        axes[1].axhline(y=1.96/np.sqrt(len(lag_acf)),linestyle='--',color='gray')
    
       

   
    
