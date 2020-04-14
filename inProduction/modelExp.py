import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.general_optimizer import GeneralOptimizerPSO
from pyswarms.backend.topology import Star
from pyswarms.utils.plotters import plot_cost_history
import multiprocessing as mp
import os

class EXP:
    ''' f(x) = a*exp(b*x) '''
    def __init__(self, N_inicial,numeroProcessadores=None):
        self.N=N_inicial
        self.a = None
        self.b = None
        self.numeroProcessadores = numeroProcessadores
         
        
    def __objectiveFunction(self,coef,x ,y):
        tam = len(y)
        res = []
        for i in range(tam):
            res.append((coef[:, 0]*np.exp(x[i]*coef[:, 1]) - y[i] )**2)
        return sum(res)/tam
    


    def fit(self, x, y , bound = None, name = None):
        self.name=name
        
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound) '''
        df = np.array(y)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        if bound==None:
            optimizer = GlobalBestPSO(n_particles = 50, dimensions = 2, options = options)
            cost, pos = optimizer.optimize(self.__objectiveFunction, 500, x = x, y = df,n_processes = self.numeroProcessadores)
            self.a = pos[0]
            self.b = pos[1]
            self.x = x
            self.y = df
            self.rmse = cost
            self.optimize = optimizer
        else:
            optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=bound)
            cost, pos = optimizer.optimize(self.__objectiveFunction, 500, x = x,y=df,n_processes=self.numeroProcessadores)
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
        return self.ypred
    

    def getCoef(self):
        return {"a": self.a, "b": self.b}


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
        return (self.predict(newx),(self.a, self.b))
        
        
    def fit_predictCI(self, y, x, ndays, bootstrap, n_jobs = None):
        """
        This function fits diffent models to data to get confidence interval for I + R.
        y = an array with the series of cases
        x = an range object with the first and last day of cases
        start =  a date in format "YYYY-mm-dd" indicating the day of the first case reported
        ndays = number of days to be predicted
        bootstrap = number of times that the model will run
        n_jobs = number of core to be used to fit the models
        
        """
        os.remove("report.log")
        #Make some parameters avaliable
        self.x = x
        self.y = np.array(y)
        self.ndays = len(self.x) + ndays
        
        
        #Create a lol with data for run the model
        lists = self.__bootstratpTS(npArray = self.y, replicate = bootstrap)
        
        #Model will be fitted and predicted so R) using ci is not consisent
        #Make cores avalible to the process
        pool =  mp.Pool(processes = n_jobs)
        
        #Run the model
        results = pool.starmap(self.runSir, [(lists[i], self.x, self.ndays) for i in range(0,len(lists))])

        self.preds = [results[i][0] for i in range(0,len(lists))] #get predictions

        a = [results[i][1][0] for i in range(0,len(lists))] #get a
        b = [results[i][1][1] for i in range(0,len(lists))] #get b
        lim_inf, med, lim_sup = self.__computeCI()
       
       
        return {"a": a, "b": b, "lim_inf": lim_inf, "med": med, "lim_sup": lim_sup}
       
    
    def __computeCI(self):
        self.lim_inf = np.quantile(self.preds, q = 0.0275, axis = 0)
        self.med = np.median(self.preds, axis = 0 )
        self.lim_sup = np.quantile(self.preds, q = 0.975, axis = 0)
        return (self.lim_inf, self.med, self.lim_sup)