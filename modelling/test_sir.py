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
#PARA population
N = 8.074e6

model = SIR_BETAS(N, None)

model.fit_lsquares(data["PA"].to_numpy(), t=data['DayNum'].to_numpy(), nbetas=3, stand_error=True)
#model.fit(data["PA"].to_numpy(), t=data['DayNum'].to_numpy(), nbetas=2)

#tt, Yo = model.predict()
tt, Yp = model.predict(coefs='LS')

#%%
plt.figure()
plt.plot(data['DayNum'], data["PA"], label='Data')
plt.plot(data['DayNum'].to_numpy(), data[u'PREDIÇÃO - PA'].to_numpy(), label='Original Fit')
plt.plot(tt, Yp, label='New Fit - LS')
#plt.plot(tt, Yo, label='New Fit - PSO')

plt.legend()