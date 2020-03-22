
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history

# Ler dados

 
   

def ler_banco(arq='datastate.csv',var='state'):
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

class SIR:
    ''' f(x) = a*exp(b*x) '''
    def __init__(self,tamanhoPop):
        self.tamanhoPop = tamanhoPop
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
   
        