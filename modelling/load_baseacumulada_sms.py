#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:53:37 2020

@author: lhunlindeion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
plt.close('all')

def model(XX, *pars):
    N0 = pars[0]
    la = pars[1]
    P = pars[2:]
    N0s = N0 * np.exp(la*XX[:,0])
    delay = XX[:,1] - XX[:,0]
    pp = np.zeros_like(delay)
    for i in range(len(P)):
        pp[delay==i] = P[i]
        
    pp[delay>=len(P)] = 1. - np.sum(P)
    return N0s*pp

def fit_model(pars, XX, Y, Sig):
    return (Y - model(XX, *pars))/Sig

#def get_pcov(res):
#    _, s, VT = np.linalg.svd(res.jac, full_matrices=False)
#    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
#    s = s[s > threshold]
#    VT = VT[:s.size]
#    pcov = np.dot(VT.T / s**2, VT)
#    cost = 2 * res.cost
#    s_sq = cost / (res.jac.shape[0] - res.x.size)
#    pcov = pcov * s_sq
#    return pcov

file = '~/ownCloud/sms/baseacumulada/dataSerieSMS_0506.csv'

data_raw = pd.read_csv(file, delimiter=',')
data_raw['control.date'] =  pd.to_datetime(data_raw['control.date'], yearfirst=True)

useful_columns = ['inicio_sintomas', 'control.date', 'evolucao']
#data_raw = data_raw.sort_values(by=['inicio_sintomas', 'dt_coleta', 'control.date'])
data_raw['inicio_sintomas'] =  pd.to_datetime(data_raw['inicio_sintomas'], errors='coerce')

data_raw = data_raw[pd.notnull(data_raw['inicio_sintomas'])]
data_raw = data_raw[data_raw['inicio_sintomas'] > pd.Timestamp(year=2020, month=3, day=1)]
data_raw = data_raw.sort_values(by=['control.date'])

data = data_raw.drop_duplicates(subset=data_raw.columns[1:])

data['cdate'] = data['control.date'].dt.dayofyear
data['idate'] = data['inicio_sintomas'].dt.dayofyear
#plt.plot(data['idate'], data['cdate'] - data['idate'], '.')
mindat = min([data['cdate'].min(), data['idate'].min()])
maxdat = max([data['cdate'].max(), data['idate'].max()])

saida = list()

for dat in range(mindat, maxdat+1):
    saida.append([dat, (data['cdate'] <= dat).sum(), (data['idate'] <= dat).sum()])

saida = np.array(saida)
#plt.figure()


vec = list()
for dati in range(data['idate'].min(), data['idate'].max()+1):
    for datc in range(data['cdate'].min(), data['cdate'].max()+1):
        vec.append([dati, datc, ((data['idate']==dati) & (data['cdate']==datc)).sum()])

vec = np.array(vec, dtype=float)

vec = vec[vec[:,2]>0,:]

nd = 60

intervals = np.array([[0.,np.inf],
                      [0., np.inf]])
ps = np.tile([0.,1.], nd).reshape((-1,2))
intervals = np.r_[intervals, ps]
p0 = np.r_[1., 0.05, np.repeat([1/nd],nd)]
res = least_squares(lambda pars: fit_model(pars, vec[:,:2], vec[:,2],np.sqrt(vec[:,2]+1)), p0, bounds=(intervals[:,0], intervals[:,1]))
#popt, pcov = curve_fit(model, vec[:,:2], vec[:,2], sigma=np.sqrt(vec[:,2]+1), p0=p0, bounds=(intervals[:,0], intervals[:,1]))
#print(res.x)
#pcov = get_pcov(res)
#print(np.sqrt(np.diag(pcov)))

plt.figure()
plt.subplot(211)
ttt = np.arange(nd)
plt.plot(ttt, res.x[2:])
plt.xlabel("Atraso (dias)")
plt.ylabel("Probabilidade")
plt.title("Atraso Máximo de {} dias".format(nd))
#print(np.sum(ttt*res.x[2:]), np.sum(res.x[2:]))
tts = np.linspace(data['idate'].min(), data['cdate'].max(), 1000)
#plt.figure()
plt.subplot(212)
plt.plot(tts, res.x[0]*np.exp(res.x[1]*tts)/res.x[1], '.-', label='Fit')
#plt.plot(data.columns.dayofyear, data.sum(axis=0))
plt.plot(saida[:,0], saida[:,1], '.-', label='Notificação')
plt.plot(saida[:,0], saida[:,2], '.-', label='Sintoma')
plt.legend()
plt.tight_layout()