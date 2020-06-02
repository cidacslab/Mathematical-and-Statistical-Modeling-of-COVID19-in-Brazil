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


ages = np.array([0,1,5,10,20,30,40,50,60,70,80])

#copied from https://github.com/WeitzGroup/covid_shield_immunity/blob/master/paper/si_immune_shielding_033120.pdf
#TODO: implement some weighting
parspad = {'p':np.array([0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.8]),
           'h':np.array([0.001, 0.001, 0.001, 0.003, 0.012, 0.032, 0.049, 0.102, 0.166, 0.243, 0.273]),
           'xi':np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.063, 0.122, 0.274, 0.432, 0.709])       
        }
with open('test_export_data.pik', 'rb') as f:
    data = pickle.load(f)
df, Nt, Ns, age_edges = data

data = {'t': df['dayofyear'].to_numpy() - df['dayofyear'].min(),
        'D_0': df['D_0'].to_numpy(),
        'D_1': df['D_1'].to_numpy(),
        'Nw_0': df['Nw_0'].to_numpy(),
        'Nw_1': df['Nw_1'].to_numpy(),
        'H_ALL': df['H_ALL'].to_numpy(),
        'U_ALL': df['U_ALL'].to_numpy()}

I0 = 0.0001 * np.ones(len(age_edges)+1)
#Initializing data
#Mb = np.array([[1., 0.7], [0.7, 1.]])
Mb = np.ones((2,2))
param = {'delta': 0.62, #asymptomatic infection correction
    'kappa': 0.25, #exposed decay parameter
    'gammaA': 1./3.5, # asymptomatic decay parameter
    'gammaS': 0.25, # symptomatic decay parameter
    'h': np.array([0.04, 0.2]), #fraction of symptomatic going to hospital
    'xi': 0.53, #fraction of hospitalizations that goes to regular beds
    'gammaH': 0.14, # regular hospital bed decay parameter
    'gammaU': 0.14, # ICU decay parameter
    'muH': 0.15, # death rate regular hospital bed
    'muU': 0.35, # death rate ICU
    'wH': 0.14, #  fraction of regular bed that goes to ICU
    'wU': 0.29, # fraction of ICU that goes to regular bed
    'p': .2, #fraction of exposed that becomes symptomatic
    'beta': [Mb], #infectivity matrix
    'tcut': None, #time instants of change of the infectivity matrix
    'x0': np.r_[I0, I0, I0] #initial conditions
  }
betas = np.array(param['beta']).flatten()
tcuts = np.array(param['tcut']).flatten()

print("Setup Complete")
#%%
#Creating artificial experimental data

#%%
# Fit Data
#pars_to_fit = ['beta_ALL', 'tcut_ALL']
#bound = [np.array([0.,1.]), np.array([2.,50.])]
pars_to_fit = ['beta_ALL', 'x0_ALL', 'h_ALL']
bound = [np.array([0.,0., 0.]), np.array([2.,0.01, 10.])]
#
#
model = SEIIHURD_age(Ns, 16)
model.fit(data, param, pars_to_fit, bound, paramPSO={'iter':100,}, stand_error=False)
model.fit_lsquares(data, param, pars_to_fit, bound, nrand=20, stand_error=False)
print(model.pos)
print(model.pos_ls)


#%%
#Plot Data
ts, Ypso = model.predict()
Yy = model.Y
ts, Yls = model.predict(coefs='LS')
#ts, Y0 = model.predict(coefs=np.r_[betas, tcuts])
#ts, Y0 = model.predict(coefs=np.r_[betas, param['muU']])
#ts, Y0 = model.predict(coefs=np.r_[1.06, 0.7, param['muU']])
plt.close('all')
for i, Yy in enumerate(model.Y):
    plt.figure()
    plt.plot(ts, Yy, 'k', label="Data")
    plt.plot(ts, Ypso[:,i], ':b', label="PSO Fit")
    plt.plot(ts, Yls[:,i], '--r', label="LS Fit")
#    plt.plot(ts, Y0[:,i], ':g', label="EDO Exact")
    plt.legend(loc='best')
    plt.title(list(data.keys())[i+1])
