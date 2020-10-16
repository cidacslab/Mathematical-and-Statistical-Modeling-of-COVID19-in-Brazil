#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:51:06 2020

SEIIHURD Fitting routine for data up to May 5 for Bahia.
"""

import numpy as np
from model_seiihurd import SEIIHURD, gen_bootstrap_full
import time
import pandas as pd
import tqdm as tqdm
import multiprocessing as mp

ncores = mp.cpu_count()
nbootstraps = 500
Ns = 14930424

bounds = {'x0_ALL': [0, 50./Ns],
          'beta_ALL': [0, 2],
          'gammaH': [1/14, 1/5],
          'gammaU': [1/14, 1/5],
          'delta': [0., 0.7],
          'tcut_ALL': [10, 60],
          'tcut_0': [9, 40],
          'h': [0.05, 0.35]
          }
   

df = pd.read_csv("../data/data.csv")
newcases = True

data = {'t': df['ts0'].to_numpy(),
        'H': df['leitos'].to_numpy(),
        'U': df['uti'].to_numpy(),
        'D': df['dthcm'].to_numpy(),
        'Nw': df['infec'].to_numpy()}

if newcases:
    data['D'] = np.r_[data['D'][0], np.diff(data['D'])]
    data['Nw'] = np.r_[data['Nw'][0], np.diff(data['Nw'])]

#Initializing: all parameters to be fitted are filled with dummy values
I0 = 0.0001 * np.ones(1)
param = {'delta': 0.62, #asymptomatic infection correction
    'kappa': 0.25, #exposed decay parameter
    'gammaA': 1./3.5, # asymptomatic decay parameter
    'gammaS': 0.25, # symptomatic decay parameter
    'h': 0.28, #fraction of symptomatic going to hospital
    'xi': 0.53, #fraction of hospitalizations that goes to regular beds
    'gammaH':0.14, # regular hospital bed decay parameter
    'gammaU': 0.14, # ICU decay parameter
    'muH': 0.15, # death rate regular hospital bed
    'muU': 0.4, # death rate ICU
    'wH': 0.14, #  fraction of regular bed that goes to ICU
    'wU': 0.29, # fraction of ICU that goes to regular bed
    'p': 0.2, #fraction of exposed that becomes symptomatic
    'beta': [1.06, 0.63], #infectivity matrix
    'tcut': [30.], #time instants of change of the infectivity matrix
    'x0': np.r_[I0, I0, I0] #initial conditions
  }

#%%

#Initialize fit parameters
pars_to_fit = ['beta_ALL', 'x0_ALL', 'gammaH',
               'gammaU', 'tcut_0', 'delta', 'h']

bound = np.array([bounds[key] for key in pars_to_fit])

bound = [bound[:,0], bound[:,1]]
paramPSO={'n_particles': 300, 'iter': 1000, 'options':{'c1': 1, 'c2': 0.5, 'w': 0.9, 'k': 5, 'p':1}}


#%%
#Create bootstraps and fit data
pars_pso = list()
series_var = list()
for i in tqdm.tqdm(range(nbootstraps)):
    datan = gen_bootstrap_full(data, newcases)
    model = SEIIHURD(Ns, ncores, usenew=newcases)
    start = time.time()
    model.fit(datan, param, pars_to_fit, bound, stand_error=True, paramPSO=paramPSO)
    pars_pso.append(model.pos)
    ts, yp = model.predict(t=np.arange(model.t[0], model.t[-1]+200) )
    series_var.append(yp)


#%%
    
#Export results

parnames = ['beta1', 'beta2', 'e0', 'ia0', 'is0', 'gammaH',	'gammaU', 't1', 'delta', 'h']
df = pd.DataFrame(np.array(pars_pso))
df.columns = parnames
df.to_csv('../results/Bahia/Parameters.csv', index=False)
series_var = np.array(series_var)
names = ['Hosp', 'UTI', 'deaths', 'Infec']

for i in range(4):
    mean = np.mean(series_var[...,i], axis=0)
    low = np.quantile(series_var[...,i], 0.025, axis=0)
    upp = np.quantile(series_var[...,i], 0.975, axis=0)
    cols = ['{}_{}'.format(names[i], ids) for ids in ['mean', 'lb', 'ub']]
    ddf = pd.DataFrame(np.array([mean,low,upp]).T)
    ddf.columns = cols
    ddf.to_csv('../results/Bahia/'+names[i]+'.csv', index=False)

if newcases:
    for i in range(2,4):
        ser = np.cumsum(series_var[...,i], axis=1)
        mean = np.mean(ser, axis=0)
        low = np.quantile(ser, 0.025, axis=0)
        upp = np.quantile(ser, 0.975, axis=0)
        cols = ['{}_{}'.format(names[i], ids) for ids in ['mean', 'lb', 'ub']]
        ddf = pd.DataFrame(np.array([mean,low,upp]).T)
        ddf.columns = cols
        ddf.to_csv('../results/Bahia/'+names[i]+'_cum.csv', index=False)
