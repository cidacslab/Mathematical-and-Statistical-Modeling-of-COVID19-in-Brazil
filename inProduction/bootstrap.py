#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:18:44 2020

@author: Rafael Veiga
@author: matheustorquato matheusft@gmail.com
"""

import numpy as np

class bootstrapTS:
    def __init__(self):
        pass

    def single_core_CI(self, model, y, x, start, ndays, bootstrap):
        """
        y = an array with the series of cases
        x = an range object with the first and last day of cases
        ndays = number of days to be forecasted
        bootstrap = number of times that the model will run
        """

        #Define a function that mimics the behavior for the model class
        def runSir(model, y, x, ndays):
            newx = range(0, len(x) + ndays) 
            model.fit(y = y, x = x)
            model.predict(newx)
            try:
                return {"pred": model.ypred, "I": model.I, "S": model.S, "R": model.R, "beta":model.beta, "gamma": model.gamma}
            except:
                return {"pred": model.ypred}
        

        #Define a auxiliary function that generate simulations based on the original series
        def bootstratpPoisson(npArray, replicate):
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

        #Create a lol with data for run the model
        lists = bootstratpPoisson(npArray = y, replicate = bootstrap)
        
        #create a empty list that will be fulffil with dictionaries
        self.results = []

        #Iterate over the list of simulated data
        for i in lists:
            self.results.append(runSir(model, i, x, ndays))
        
        #Compute the median and bounds for the predicted value (I + R)
        self.pred = np.median([self.results[i]["pred"] for i in range(0,len(self.results))], axis = 0)
        self.lim_inf = np.quantile([self.results[i]["pred"] for i in range(0,len(self.results))], q = 0.0275,  axis = 0)
        self.lim_sup = np.quantile([self.results[i]["pred"] for i in range(0,len(self.results))], q = 0.975, axis = 0)

        #Try to get extra parameters
        try:
            self.beta = [self.results[i]["beta"] for i in range(0,len(self.results))]
            self.gamma = [self.results[i]["gamma"] for i in range(0,len(self.results))]
        except:
            pass

        try:
            return (self.beta, self.gamma, self.pred, self.lim_inf, self.lim_sup)
        except:
            return (self.pred, self.lim_inf, self.lim_sup)

    