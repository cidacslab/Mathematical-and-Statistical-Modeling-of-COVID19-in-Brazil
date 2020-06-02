#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:19:41 2020

@author: Felipe A. C. Pereira
"""

import pandas as pd

        
    
def int_br(x):
    return int(x.replace('.',''))

def float_br(x):
    return float(x.replace('.', '').replace(',','.'))

dia = '2805'

file_HU = '~/ownCloud/sesab/exporta_boletim_epidemiologico_csv_{}.csv'.format(dia)
datahu = pd.read_csv(file_HU, sep=';', decimal=',', converters={'CASOS CONFIRMADOS': int_br})
rday = 'DATA DO BOLETIM'
datahu[rday] = pd.to_datetime(datahu[rday], dayfirst=True)
datahu['DayNum'] = datahu[rday].dt.dayofyear

ref = pd.Timestamp(year=2020, month=2, day=27).dayofyear
datahu['ts0'] = datahu['DayNum'] - ref

colsutils = ['DATA DO BOLETIM', 'ts0', 'CASOS CONFIRMADOS', 'CASOS ENFERMARIA',
             'CASOS UTI','TOTAL OBITOS']
dfi = datahu[colsutils]
dff = pd.DataFrame(columns=colsutils)

for i, day in enumerate(dfi['ts0'].unique()):
    line = dfi[dfi['ts0'] == day]
    line = line.sort_values(by=['DATA DO BOLETIM'], ascending=False)
    line.reset_index(drop=True, inplace=True)
    dff.loc[i] = line.loc[0]

cols = ['dates', 'ts0', 'infec', 'leitos', 'uti', 'dthcm']
dff.columns = cols

df0 = pd.read_csv('data_0.csv')
df0['dates'] = pd.to_datetime(df0['dates'], dayfirst=True)

dfnew = pd.concat([df0, dff], sort=False).drop_duplicates(['ts0'], keep='last').sort_values(by=['ts0'])
dfnew.reset_index(drop=True, inplace=True)
ndth = dfnew['dthcm'].diff()
ndth[0] = 0
dfnew['dth'] = ndth
dfnew.to_csv('data_to_seiihurd_2805.csv')

