#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:00:37 2020

@author: Felipe A C Pereira
"""

import numpy as np
import pandas as pd
from model_SIR import SIR_BETAS
import sys
import matplotlib.pyplot as plt
import copy
plt.close('all')

stand_error = True
nbetas = 3
nproc = 16
pred_days = 7

def create_summary(model, nbetas, estado):
    temp = dict()
    temp['state'] = estado
    temp['I0'] = model.pars_opt_ls['x0']
    temp['gamma'] = model.pars_opt_ls['gamma']
    temp['t0'] = model.t[0]
    model.pars_opt_ls['tcut'].sort()
    for j in range(nbetas):
        temp['beta_{}'.format(j)] = model.pars_opt_ls['beta'][j]
    for j in range(nbetas-1):
        temp['tchange_{}'.format(j)] = model.pars_opt_ls['tcut'][j]
    return temp

def create_output(model, data, pred_days, state):
    tts, Y, mY = model.predict(t=np.arange(model.t[0], model.t[-1]+1+pred_days),
                               coefs='LS', model_output=True)
    mY = Ne * mY
    filler = np.nan*np.ones(pred_days)
    temp = dict()
    time = pd.to_datetime( 2020000 + tts, format='%Y%j')
    temp['date'] = ["{:04}-{:02}-{:02}".format(y,m,d) for y,m,d in zip(time.year, time.month, time.day)]
    temp['state'] = state
    temp['newCases'] = np.r_[data['newCases'].to_numpy(), filler]
    temp['mortes'] = np.r_[data['deaths'].to_numpy(), filler]
    temp['TOTAL'] = np.r_[data['totalCases'].to_numpy(), filler]
    temp['totalCasesPred'] = Y
    temp['residuo_quadratico'] = np.r_[model._residuals(model.pos_ls)**2, filler]
    temp['res_quad_padronizado'] = np.r_[model._residuals(model.pos_ls, stand_error=True)**2,
                                        filler]
    temp['suscetivel'] = mY[:,0]
    temp['infectado'] = mY[:,1]
    temp['recuperado'] = mY[:,2]
    return pd.DataFrame(temp) 
    
    


pops = {'RO':	1777225,
        'AC':	881935,
        'AM':	4144597,
        'RR': 	605761,
        'PA':	8602865,
        'AP':	845731,
        'TO':	1572866,
        'MA':	7075181,
        'PI':	3273227,
        'CE':	9132078,
        'RN':	3506853,
        'PB':   4018127,
        'PE':	9557071,
        'AL':   3337357,
        'SE':   2298696,
        'BA':   14873064,
        'MG':   21168791,
        'ES':	4018650,
        'RJ':   17264943,
        'SP':   45919049,
        'PR':   11433957,
        'SC':   	7164788,
        'RS':	11377239,
        'MS':	2778986,
        'MT':   3484466,
        'GO':   7018354,
        'DF':	3015268,
        'TOTAL':210147125
}

url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
data = pd.read_csv(url)
data['date'] = pd.to_datetime(data['date'], yearfirst=True)
data['DayNum'] = data['date'].dt.dayofyear
estados = data.state.unique()
#estados = ['SP']
columns = ['DayNum', 'totalCases', 'newCases', 'deaths']

outp_par = {'state':[], 'gamma':[], 'I0':[], 't0':[]}
for i in range(nbetas):
    outp_par['beta_{}'.format(i)] = []
for i in range(nbetas-1):
    outp_par['tchange_{}'.format(i)] = []
outp_par = pd.DataFrame(outp_par)

#%%
outp_data = pd.DataFrame({'date':[], 'state':[], 'newCases':[],   'mortes':[],
                         'TOTAL':[], 'totalCasesPred':[], 'residuo_quadratico':[],
                         'res_quad_padronizado':[],	'suscetivel':[],
                         'infectado':[], 'recuperado':[]})
for i, estado in enumerate(estados):
    print(estado)
    Ne = pops[estado]
    data_fil = data[data['state'] == estado][columns]
    data_fil = data_fil.sort_values(by='DayNum')
    model = SIR_BETAS(Ne, nproc)
    model.fit_lsquares(data_fil['totalCases'].to_numpy(), data_fil['DayNum'].to_numpy(),
                       nbetas=nbetas, stand_error=True, nrand=32)
    temp = create_summary(model, nbetas, estado)
    outp_par = outp_par.append(temp, ignore_index=True)
    outp_data = pd.concat([outp_data,create_output(model, data_fil, pred_days,\
                                                   estado)], sort=False)
    tts, Y = model.predict(coefs='LS')
    if i % 12 == 0:
        plt.figure()
    plt.subplot(3,4,(i%12)+1)
    plt.plot(model.t, model.Y, label='Data')
    plt.plot(tts, Y, label='Fit')
    plt.title(estado)
    if i % 12 == 11:
        plt.tight_layout()
outp_par.to_csv('sir_estados_sumario.csv', index=False)
outp_data.to_csv('estados.csv', index=False)