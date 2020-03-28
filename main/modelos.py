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
import scipy.integrate as spi
from platypus import NSGAII, Problem, Real
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history



logging.disable()



def ler_banco(arq,var):
    banco = pd.read_csv(arq)
    banco =banco[banco[var].notnull()]
    if var=='cod_city':
        banco[var] = pd.to_numeric(banco[var],downcast='integer')
    nome_local =list(banco[var].unique())
    for i in banco.index:
        banco.date[i] = dt.datetime.strptime(banco.date[i], '%Y-%m-%d').date()
    nome_local =list(banco[var].unique())
    local = []
    for est in nome_local:
        
    
        aux = banco[banco[var]==est].sort_values('date')
        data_ini = aux.date.iloc[0]
        data_fim = aux.date.iloc[-1]
        dias = (data_fim-data_ini).days + 1
        d = [(data_ini + dt.timedelta(di)) for di in range(dias)]
        if var=='cod_city':
            cod_city = [est for di in range(dias)]
            estado = [aux.state.iloc[0] for di in range(dias)]
            uf = [aux.UF.iloc[0] for di in range(dias)]
            city = [aux.city.iloc[0] for di in range(dias)]
            df = pd.DataFrame({'date':d,'state':estado,'UF':uf,'city':city,var:cod_city})
            df.cod_city =pd.to_numeric(df.cod_city,downcast='integer')
        else:
            estado = [est for di in range(dias)]
            df = pd.DataFrame({'date':d,var:estado})
        
        casos = []
        caso = 0 
        i_aux = 0
        for i in range(dias):
            if (d[i]-aux.date.iloc[i_aux]).days==0:
                caso = aux.totalcases.iloc[i_aux]
                casos.append(caso)
                i_aux=i_aux+1
            else:
                casos.append(caso)
        df['totalcasos'] = casos
        local.append(df)
    return nome_local, local    


def ler_banco_alternativa():
    
    try:
        url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
        df = pd.read_csv(url)
    except:
        return None,None
    
    df.drop(['country','city'],axis=1,inplace=True)
    df['date'] = pd.to_datetime(df['date'],format='%Y/%m/%d')
    
    filename = '../data/populações.csv'
    populacoes = pd.read_csv(filename,index_col=0)
    populacoes = populacoes.transpose()
    
    timeseries = pd.DataFrame(columns=np.sort(df['state'].unique()),index=df['date'].unique())
    timeseries.fillna(0,inplace=True)
    cols = timeseries.columns.tolist()
    timeseries.columns = cols[-1:] + cols[:-1]
    del cols, url 
    
    
    for i in df.index:
        
        if(timeseries[df.iloc[i]['state']].sum() == 0):
        
            timeseries.loc[df.iloc[i]['date']][df.iloc[i]['state']] = \
            df.iloc[i]['newCases']
    
        else:
            
            timeseries.loc[df.iloc[i]['date']][df.iloc[i]['state']] = \
            df.iloc[i]['newCases'] + timeseries[df.iloc[i]['state']][timeseries[df.iloc[i]['state']]!=0][-1]
    
    return timeseries,populacoes
            
            

class EXP:
    ''' f(x) = a*exp(b*x) '''
    def __init__(self):
        self.a = None
        self.b = None
        
    def __objectiveFunction(self,coef,x ,y):
        tam = len(y)
        res = []
        for i in range(tam):
            res.append((coef[:, 0]*np.exp(x[i]*coef[:, 1]) - y[i] )**2)
        return sum(res)/tam
    


    def fit(self, x,y , bound = None):
        
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound) '''
        df = np.array(y)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        if bound==None:
            optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options)
            cost, pos = optimizer.optimize(self.__objectiveFunction, 1000, x = x,y=df)
            self.a = pos[0]
            self.b = pos[1]
            self.x = x
            self.y = df
            self.rmse = cost
            self.optimize = optimizer
        else:
            optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options,bounds=bound)
            cost, pos = optimizer.optimize(self.__objectiveFunction, 1000, x = x,y=df)
            self.a = pos[0]
            self.b = pos[1]
            self.x = x
            self.y = df
            self.rmse = cost
            self.optimize = optimizer
            
    def predict(self,x):
        ''' x = dias passados do dia inicial 1'''
        res = [self.a*np.exp(self.b*v) for v in x]
         
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
        return ['a','b',['casos']], [self.a,self.b,[self.y]]
   
class SIR_PSO:
    ''' SIR Model'''
    def __init__(self,tamanhoPop):
        self.tamanhoPop = tamanhoPop
        self.beta = None
        self.gamma = None
        
    
    def __cal_EDO(self,x,y,beta,gamma):
        I0 = y[0]
        S0 = self.tamanhoPop - I0
        R0 = 0
        def funct(y,x):
            S=y[0]
            I=y[1]   
            f0 = - beta * S * I        
            f1 = beta * S * I - gamma * I
            f2 = gamma * I
            return [f0, f1, f2]
        initial_cond = [S0,I0,R0]
     #integrate------------------------------------
        self.teste = (funct,initial_cond,x)
        ds = spi.odeint(funct,initial_cond,x)
        return (ds[:,0],ds[:,1],ds[:2])
    
    def __objectiveFunction(self,coef,x ,y):
        tam = len(y)
        soma = []
        for i in range(len(coef[:,0])):
            res = []
            S,I,R = self.__cal_EDO(x,y,coef[i,0],coef[i,1])
            for j in range(tam):
                res.append((I[j] - y[j])**2)
            soma.append(sum(res)/tam)
        return np.array(soma)
    

    def fit(self, x,y , bound = None):
        
        '''
        x = dias passados do dia inicial 1
        y = numero de casos
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound) '''
        df = np.array(y)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        if bound==None:
            optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options)
            cost, pos = optimizer.optimize(self.__objectiveFunction, 1000, x = x,y=df)
            self.beta = pos[0]
            self.gamma = pos[1]
            self.x = x
            self.y = df
            self.rmse = cost
            self.optimize = optimizer
            
        else:
            optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options,bounds=bound)
            cost, pos = optimizer.optimize(self.__objectiveFunction, 1000, x = x,y=df)
            self.beta = pos[0]
            self.gamma = pos[1]
            self.x = x
            self.y = df
            self.rmse = cost
            self.optimize = optimizer
            
            
    def predict(self,x):
        ''' x = dias passados do dia inicial 1'''
        S,I,R = self.__cal_EDO(x,self.y,self.beta,self.gamma)
        self.S = S
        self.I = I
        self.R = R         
        return I
    
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
        return ['beta','gamma',['suscetivel','infectados','recuperados','casos']], [self.beta,self.gamma,self.Y]
    
class SIR_EDO:

    def __init__(self,N):
        """
        Parameters
        ----------
        N : int
            População Inicial
        """
        self.N = N
        self.I0 = 1/N
        self.S0 = 1-self.I0
        self.R0 = 0
        self.beta = None
        self.gamma = None

    
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

        mean_squared_error = ((np.array(y) - result[:, 1]) ** 2).mean()
    
        return [mean_squared_error]

    def fit(self, x,y ,bound = ([0,1/14-0.0001],[1,1/5+0.0001]),name = None):
        
        self.y=np.array(y)
        self.x = x
        
        TS = 1
        ND = len(y) - 1
        
        y = y/self.N
    
        t_start = 0.0
        t_end = ND
        t_inc = TS
        t_range = np.arange(t_start, t_end + t_inc, t_inc)
    
        Model_Input = (self.S0, self.I0, self.R0)
    
        # GA Parameters
        number_of_generations = 1000
        ga_population_size = 100
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
        
            
    def predict(self,x):
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
            
            self.ypred = result_fit[:, 1]*self.N
            self.S=result_fit[:, 0]*self.N
            self.R=result_fit[:, 2]*self.N

            return result_fit[:, 1]*self.N
        
    def plot(self,local):
        plt.plot(self.ypred,c='b',label='Predição Infectados')
        plt.plot(self.y,c='r',marker='o', markersize=3,label='Infectados')
        plt.legend(fontsize=15)
        plt.title('Dinâmica do CoviD19 - {}'.format(local),fontsize=20)
        plt.ylabel('Casos COnfirmados',fontsize=15)
        plt.xlabel('Dias',fontsize=15)
        plt.show()
    def getCoef(self):
        return ['beta','gamma','R0',('S','I','R')], [self.beta,self.gamma,self.gamma/self.beta,(self.S,self.ypred,self.R)]
        

class SEIR_EDO:
    
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
      '''The main set of equations'''
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

    def fit(self, x,y ,bound = None,name = ''):

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
        


class SEQIJR_EDO:

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

   def fit(self, x,y ,bound = None,name = ''):
       
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
 
                           
