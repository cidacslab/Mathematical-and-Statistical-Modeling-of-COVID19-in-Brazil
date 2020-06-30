#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:56:41 2020

@author: lhunlindeion
"""

import numpy as np
import pandas as pd
from model_SIR import SIR_BETAS
import matplotlib.pyplot as plt

#import PA data from Painel Covida (26/06/20)
data = pd.read_csv('chart.csv', sep=';', decimal=',')
dday = 'DateTime'
data[dday] = pd.to_datetime(data[dday], yearfirst=True)
data['DayNum'] = data[dday].dt.dayofyear
#Remove the predict only days
data = data.dropna()
nbeta_max = 4
#PARA population
N = 8.074e6
models = list()
for nbetas in range(1, nbeta_max+1):
    print(nbetas)
    model = SIR_BETAS(N, 16)
    model.fit_lsquares(data["PA"].to_numpy(), t=data['DayNum'].to_numpy(), nbetas=nbetas, stand_error=True, nrand=16)
    model.fit_ML(data["PA"].to_numpy(), t=data['DayNum'].to_numpy(), nbetas=nbetas, init=model.pos_ls)

#model.fit(data["PA"].to_numpy(), t=data['DayNum'].to_numpy(), nbetas=2)
    models.append(model)
#tt, Yo = model.predict()


#%%
plt.figure()
plt.plot(data['DayNum'], data["PA"], label='Data')
plt.plot(data['DayNum'].to_numpy(), data[u'PREDIÇÃO - PA'].to_numpy(), label='Original Fit')
for model in models:
    tt, Yp = model.predict(coefs='LS')
    plt.plot(tt, Yp, label='Fit LS - n={}, BIC={}'.format(model.nbetas, model.BIC()))
    tt, Yo = model.predict(coefs='ML')
    plt.plot(tt, Yo, label='Fit ML - n={}, BIC={}'.format(model.nbetas, model.BIC('ML')))

#plt.plot(tt, Yo, label='New Fit - PSO')

plt.legend()

for model in models:
    print(model.result_ml.fun, model.result_ls.cost)