#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:51:06 2020

@author: Felipe A. C. Pereira
Teste de funcionamento do modelo SEIIHURD com separação de grupos.
O sistema cria dados artificiais usando a versão estocástica do modelo e 
depois ajusta usando PSO e LS. no final, as curvas de infectados e mortos 
são geradas, junto com a curva dos parametros usados pela simulação.
"""
import matplotlib.pyplot as plt
import numpy as np
from model_seiihurd import SEIIHURD_age
import pickle

bounds = {'x0_ALL': [0, 0.001],
          'beta_ALL': [0, 2],
          'muU_ALL': [0.1 , 0.7],
          'muU_0': [0.1, 0.3],
          'muU_1': [0.4, 0.7],
          'muH_ALL': [0.05 , 0.3],
          'muH_0': [0.05, 0.15],
          'muH_1': [0.15, 0.3],
          'gammaH_ALL': [1/21, 0.25],
          'gammaU_ALL': [1/21, 0.25],
          'gammaH_0': [1/12, 1/4],
          'gammaU_0': [1/14, 1/6],
          'gammaH_1': [1/14, 1/6],
          'gammaU_1': [1/21, 1/10],
          'p_0': [0.2, 0.3],
          'p_1': [0.7, 0.85],
          'delta': [0.4, 0.7],
          'xi_0': [0.88, 1.],
          'xi_1': [0.71, 0.91],
          'tcut_ALL': [95, 120]
          }

bounds_full = {'x0_ALL': [0, 0.01],
          'beta_ALL': [0, 2],
          'muU_ALL': [0.0 , 1],
          'muU_0': [0., 1],
          'muU_1': [0., 1],
          'muH_ALL': [0 , 1],
          'muH_0': [0., 1],
          'muH_1': [0., 1],
          'gammaH_ALL': [1/50, 1],
          'gammaU_ALL': [1/50, 1],
          'gammaH_0': [1/50, 1],
          'gammaU_0': [1/50, 1],
          'gammaH_1': [1/50, 1],
          'gammaU_1': [1/50, 1],
          'p_0': [0.0, 1],
          'p_1': [0.0, 1],
          'delta': [0., 1],
          'xi_0': [0., 1.],
          'xi_1': [0., 1.]
          }


with open('test_export_data_sesab_0806.pik', 'rb') as f:
    data = pickle.load(f)
df, par_age, Ns, age_edges = data


incl_cases = False

data = {'t': df['dayofyear'].to_numpy(),
        'H_ALL': df['H_ALL'].to_numpy(),
        'U_ALL': df['U_ALL'].to_numpy(),
        'D_0': df['D_0'].to_numpy(),
        'D_1': df['D_1'].to_numpy()}

if incl_cases:
    data['Nw_0'] = df['Nw_0'].to_numpy()
    data['Nw_1'] = df['Nw_1'].to_numpy()

I0 = 0.0001 * np.ones(len(age_edges)+1)
#Initializing data
#Mb = np.array([[1., 0.7], [0.7, 1.]])
Mb = np.ones((2,2))
param = {'delta': 0.6, #asymptomatic infection correction
    'kappa': 0.25, #exposed decay parameter
    'gammaA': 1./3.5, # asymptomatic decay parameter
    'gammaS': 0.25, # symptomatic decay parameter
    'h': par_age['h'], #fraction of symptomatic going to hospital
    'xi': par_age['xi'], #fraction of hospitalizations that goes to regular beds
    'gammaH': np.array([0.14, 1./12.]), # regular hospital bed decay parameter
    'gammaU': np.array([0.14, 1./12.]), # ICU decay parameter
    'muH': np.array([0.1, 0.2]), # death rate regular hospital bed
    'muU': np.array([0.2, 0.6]), # death rate ICU
    'wH': 0.14, #  fraction of regular bed that goes to ICU
    'wU': 0.29, # fraction of ICU that goes to regular bed
    'p': par_age['p'], #fraction of exposed that becomes symptomatic
    'beta': [Mb, Mb.copy()], #infectivity matrix
    'tcut': [105], #time instants of change of the infectivity matrix
    'x0': np.r_[I0, I0, I0] #initial conditions
  }
betas = np.array(param['beta']).flatten()
tcuts = np.array(param['tcut']).flatten()

print("Setup Complete")
#%%
#Creating artificial experimental data

#%%
# Fit Data
pars_to_fit = ['beta_ALL', 'x0_ALL', 'xi_0', 'xi_1', 'gammaH_0', 'gammaH_1',
               'gammaU_0', 'gammaU_1', 'muH_0', 'muH_1', 'muU_0', 'muU_1']

pars_to_fit = pars_to_fit + ['tcut_ALL']

if incl_cases:
    pars_to_fit = pars_to_fit + ['delta', 'p_0', 'p_1']

bound = np.array([bounds[key] for key in pars_to_fit])

bound = [bound[:,0], bound[:,1]]

#
#
model = SEIIHURD_age(Ns, 16)
model.fit_lsquares(data, param, pars_to_fit, bound, nrand=32, stand_error=True)
#model.fit(data, param, pars_to_fit, bound, paramPSO={'iter':200,}, stand_error=False)
#print(model.pos)
print(model.pos_ls)


#%%
#Plot Data
#ts, Ypso = model.predict()
Yy = model.Y
ts, Yls = model.predict(coefs='LS')#, t=np.arange(model.t[0], 365))
#ts, Y0 = model.predict(coefs=np.r_[betas, tcuts])
#ts, Y0 = model.predict(coefs=np.r_[betas, param['muU']])
#ts, Y0 = model.predict(coefs=np.r_[1.06, 0.7, param['muU']])
plt.close('all')
for i, Yy in enumerate(model.Y):
    plt.figure()
    plt.plot(model.t, Yy, 'k', label="Data")
#    plt.plot(ts, Ypso[:,i], ':b', label="PSO Fit")
    plt.plot(ts, Yls[:,i], '--r', label="LS Fit")
#    plt.plot(ts, Y0[:,i], ':g', label="EDO Exact")
    plt.legend(loc='best')
    plt.title(list(data.keys())[i+1])
