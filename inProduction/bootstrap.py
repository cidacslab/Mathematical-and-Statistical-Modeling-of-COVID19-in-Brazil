#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:18:44 2020

@author: Rafael Veiga
@author: matheustorquato matheusft@gmail.com
"""

import numpy as np
import multiprocessing.dummy as mp
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

logging.disable()

class bootstrapTS:
    def __init__(self):
        pass

    def single_core_CI(self, model, y, x, ndays, bootstrap, simulation = "Poisson", method = "percentile"):
        """
        y: an array with the series of cases
        x: an range object with the first and last day of cases
        ndays: number of days to be forecasted
        bootstrap: number of times that the model will run
        simulation: Choose among Poisson, Mixing_Poisson or Gamma_Poisson.
        method: accepted methods are percentile, approximation or approximation
        """
        self.x = x
        self.y = y
        self.ndays = ndays


        #Define a auxiliary functions that generate simulations based on the original series
        def bootstratpPoisson(npArray, replicate, cumSum = True):
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
        def bootstrapDPoisson(npArray, replicate):
            simList = []
            poissonsDist = bootstratpPoisson(npArray, replicate, cumSum = False)
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
        def bootstrapGammaPoisson(npArray, replicate):
            simList = []
            poissonsDist = bootstratpPoisson(npArray, replicate, cumSum = False)
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


        # #Define a function that mimics the behavior for the model class
        def runSir(model, y, x):
            self.newx = range(0, len(self.y) + self.ndays) 
            model.fit(y = y, x = x)
            model.predict(self.newx)
            try:
                return {"pred": model.ypred, "I": model.I, "S": model.S, "R": model.R, "beta":model.beta, "gamma": model.gamma}
            except:
                return {"pred": model.ypred}

        #Create a lol with data for run the model
        if simulation == "Poisson":
            lists = bootstratpPoisson(npArray = y, replicate = bootstrap)

        elif simulation == "Mixing_Poisson":
            lists = bootstrapDPoisson(npArray = y, replicate = bootstrap)

        elif simulation == "Gamma_Poisson":
            lists = bootstrapGammaPoisson(npArray = y, replicate = bootstrap)
        
        #create a empty list that will be fulffil with dictionaries
        self.results = []

        # #Iterate over the list of simulated data
        for i in range(0,len(lists)):
            self.results.append(runSir(model, lists[i], x))
        
        # Get predictions (I + R)
        self.pred = np.array([self.results[i]["pred"] for i in range(0,len(self.results))])
        self.meanPred = [np.mean(self.pred[:,i]) for i in range(0,len(self.y) + self.ndays)]
      

        if method == "percentile":

            self.lim_inf = [np.quantile(self.pred[:,i], q = 0.025) for i in range(0,len(self.meanPred))]
            self.lim_sup = [np.quantile(self.pred[:,i], q = 0.975) for i in range(0,len(self.meanPred))]

        elif method == "basic":
            deltaL = np.array([np.quantile(self.pred[:,i], q = 0.025) for i in range(0,len(self.meanPred))])
            deltaU = np.array([np.quantile(self.pred[:,i], q = 0.025) for i in range(0,len(self.meanPred))])
            self.lim_inf  = deltaL - self.meanPred
            self.lim_sup  = deltaU - self.meanPred
           

        elif method == "approximation":

            percentiles = np.array([0.025, 1.0 - 0.025])
            norm_quantiles = stats.norm.ppf(percentiles)

            errors  = self.pred - self.meanPred
            self.std_err = np.sqrt(np.diag(errors.T.dot(errors)/bootstrap))

            self.lim_inf = self.meanPred + norm_quantiles[0] * self.std_err
            self.lim_sup = self.meanPred + norm_quantiles[1] * self.std_err

        

            
            


        # #Try to get extra parameters
        try:
            self.beta = [self.results[i]["beta"] for i in range(0,len(self.results))]
            self.gamma = [self.results[i]["gamma"] for i in range(0,len(self.results))]
        except:
            pass

        try:
            return [self.beta, self.gamma, self.meanPred, self.lim_inf, self.lim_sup]
        except:
            return [self.meanPred, self.lim_inf, self.lim_sup]

        


    def plotParam(self, results):
        x = range(0,len(self.meanPred))
        fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (18,12))

        axes[0,0].hist(results[0])
        axes[0,1].hist(results[1])

        axes[1,0].scatter(self.x,self.y, c = "red")
        axes[1,0].plot(self.x,results[3][0:len(self.y)], "--", c = "black")
        axes[1,0].plot(self.x,results[2][0:len(self.y)], c = "blue")
        axes[1,0].plot(self.x,results[4][0:len(self.y)], "--", c = "black")

        axes[1,1].plot(x,results[3][0:len(self.meanPred)], "--", c = "black")
        axes[1,1].plot(x,results[2][0:len(self.meanPred)], c = "blue")
        axes[1,1].plot(x,results[4][0:len(self.meanPred)], "--", c = "black")

   
    