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


#Initializing data
Ns = np.array([100000, 10000])
I0 = np.array([500, 50]) / Ns
zer = np.zeros(2)
Mb = np.array([[1., 0.7], [0.7, 1.]])
param = {'delta': 0.62,
    'kappa': 0.25,
    'gammaA': 1./3.5,
    'gammaS': 0.25,
    'h': 0.28,
    'xi': 0.53,
    'gammaH': 0.14,
    'gammaU': 0.14,
    'muH': 0.15,
    'muU': 0.35,
    'wH': 0.14,
    'wU': 0.29,
    'p': .2,
    'beta': [1.06 * Mb, 0.7 * Mb],
    'tcut': [7],
    'x0': np.r_[1.-3*I0, I0, I0, I0, zer, zer, zer, zer, I0]
  }
betas = np.array(param['beta']).flatten()
tcuts = np.array(param['tcut']).flatten()

print("Setup Complete")
#%%
#Creating artificial experimental data
simulate_series = True


if simulate_series:
    paramb = param.copy()
    paramb['x0'] = Ns * I0
    tmax = 50
    S = call_stoch_SEIIHURD(Ns, tmax, 1., paramb)
    Sa = np.c_[S[:,0].reshape((-1,1)), S[:,-4:]]
    np.save('teste_stoch', Sa)    
else:
    Sa = np.load('teste_stoch.npy')

data = {'t': Sa[:,0],
        'D_0': Sa[:,1],
        'D_1': Sa[:,2],
        'Nw_0': Sa[:,3],
        'Nw_1': Sa[:,4]}

print("Artificial Data Created")


#%%
# Fit Data
#pars_to_fit = ['beta_ALL', 'tcut_ALL']
#bound = [np.array([0.,1.]), np.array([2.,50.])]
pars_to_fit = ['beta_ALL', 'muU']
bound = [np.array([0.,0.]), np.array([2.,1.])]


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
ts, Y0 = model.predict(coefs=np.r_[betas, param['muU']])
plt.close('all')
for i, Yy in enumerate(model.Y):
    plt.figure()
    plt.plot(ts, Yy, 'k', label="Data")
    plt.plot(ts, Ypso[:,i], ':b', label="PSO Fit")
    plt.plot(ts, Yls[:,i], '--r', label="LS Fit")
    plt.plot(ts, Y0[:,i], ':g', label="EDO Exact")
    plt.legend(loc='best')
