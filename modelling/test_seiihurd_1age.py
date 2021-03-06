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
from stochastic_seiihurd import call_stoch_SEIIHURD
from model_seiihurd import SEIIHURD_age
import copy
from scipy.optimize import dual_annealing
#import time

#Initializing data
Ns = np.array([500000])
I0 = np.array([1000]) / Ns
zer = np.zeros(1)
Mb = np.array([[1.]])
param = {'delta': 0.62, #asymptomatic infection correction
    'kappa': 0.25, #exposed decay parameter
    'gammaA': 1./3.5, # asymptomatic decay parameter
    'gammaS': 0.25, # symptomatic decay parameter
    'h': 0.28, #fraction of symptomatic going to hospital
    'xi': 0.53, #fraction of hospitalizations that goes to regular beds
    'gammaH': 0.14, # regular hospital bed decay parameter
    'gammaU': 0.14, # ICU decay parameter
    'muH': 0.15, # death rate regular hospital bed
    'muU': 0.35, # death rate ICU
    'wH': 0.14, #  fraction of regular bed that goes to ICU
    'wU': 0.29, # fraction of ICU that goes to regular bed
    'p': .2, #fraction of exposed that becomes symptomatic
    'beta': [1.06 * Mb, 0.4 * Mb, 1.06*Mb], #infectivity matrix
    'tcut': [15,45], #time instants of change of the infectivity matrix
    'x0': np.r_[I0, I0, I0] #initial conditions
  }
betas = np.array(param['beta']).flatten()
tcuts = np.array(param['tcut']).flatten()

print("Setup Complete")
#%%
#Creating artificial experimental data
simulate_series = True


if simulate_series:
    paramb = copy.deepcopy(param)
    paramb['x0'] = np.tile(Ns, 3) * paramb['x0']
    tmax = 70
    S = call_stoch_SEIIHURD(Ns, tmax, 1., paramb)
    Sa = np.c_[S[:,0].reshape((-1,1)), S[:,-2:]]
    np.save('teste_stoch', Sa)
else:
    Sa = np.load('teste_stoch1a.npy')

#Sa[10,1] = np.nan

data = {'t': Sa[:,0],
        'D_0': Sa[:,1],
#        'D_1': Sa[:,2],
        'Nw_0': Sa[:,2],
#        'Nw_1': Sa[:,4]
        }

print("Artificial Data Created")


#%%
# Fit Data
#pars_to_fit = ['beta_ALL', 'tcut_ALL']
#bound = [np.array([0.,1.]), np.array([2.,50.])]
pars_to_fit = ['beta_ALL', 'x0_ALL', 'tcut_ALL']
bound = [np.array([0.,0., 5.]), np.array([2., 0.01, 40.])]
#
#
model = SEIIHURD_age(Ns, 16)
model.fit_lsquares(data, param, pars_to_fit, bound, nrand=32, nages=1, stand_error=False)
model.fit(data, param, pars_to_fit, bound, nages=1, paramPSO={'iter':100,}, stand_error=False)
print(model.pos)
print(model.pos_ls)


#%%
#Plot Data
ts, Ypso = model.predict()
Yy = model.Y
ts, Yls = model.predict(coefs='LS')
#ts, Y0 = model.predict(coefs=np.r_[betas, tcuts])
#ts, Y0 = model.predict(coefs=np.r_[betas, param['muU']])
ts, Y0 = model.predict(coefs=np.r_[1.06, 0.4, 1.06, param['x0'], param['tcut']])
#ts, Ys = model.predict(coefs=res.x)
plt.close('all')
for i, Yy in enumerate(model.Y):
    plt.figure()
    plt.plot(ts, Yy, 'k', label="Data")
    plt.plot(ts, Ypso[:,i], ':b', label="PSO Fit")
    plt.plot(ts, Yls[:,i], '--r', label="LS Fit")
#    plt.plot(ts, Ys[:,i], 'c', label="Other")
    plt.plot(ts, Y0[:,i], ':g', label="EDO Exact")
    plt.legend(loc='best')
    plt.title(list(data.keys())[i+1])
