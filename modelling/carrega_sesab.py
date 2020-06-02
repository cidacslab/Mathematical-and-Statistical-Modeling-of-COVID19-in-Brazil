#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:19:41 2020

@author: Felipe A. C. Pereira
"""

import pandas as pd
import numpy as np
import pickle
#import matplotlib.pyplot as plt

def separate_age_groups(data):
    idades_0 = data['IDADE'].unique()
    idades = list()
    for idade in idades_0:
        if idade[0] == '<':
            idades.append([0, idade])
        elif idade == 'ignorado':
            idades.append([np.nan, idade])
        else:
            idades.append([int(idade.split(' ')[0]), idade])
    DayNums = data['DayNum'].unique()
    DayNums.sort()
    saida = np.zeros((len(DayNums), len(idades)), dtype=int)
    Ns = list()
    for j, idade in enumerate(idades):
        Ns.append(int(data[data['IDADE'] == idade[1]]['N'].mean()))
        for i, dia in enumerate(DayNums):
            saida[i,j] = data[(data['IDADE'] == idade[1]) & \
                 (data['DayNum'] == dia)]['Nw'].max()
    return idades, Ns, saida, DayNums

def consolidate_age_groups(counts, idades, age_edges, Ns, count_ign=True):
    ids = np.array([x[0] for x in idades])
    Ns = np.array(Ns)
    saida = np.empty((counts.shape[0], len(age_edges)+1))
    Na = np.empty(len(age_edges)+1)
    Na[0] = Ns[ids < age_edges[0]].sum()
    saida[:,0] = counts[:,ids < age_edges[0]].sum(axis=1)
    for i in range(1, len(age_edges)):
        saida[:,i] = counts[:,(ids < age_edges[i]) * (ids >= age_edges[i-1])].sum(axis=1)
        Na[i] = Ns[(ids < age_edges[i]) * (ids >= age_edges[i-1])].sum()
    saida[:,-1] = counts[:,ids >= age_edges[-1]].sum(axis=1)
    Na[-1] = Ns[ids >= age_edges[-1]].sum()
    if count_ign:
        ign = counts[:, np.isnan(ids)]
        fsaida = saida / saida.sum(axis=1).reshape((-1,1))
        saida = saida + ign * fsaida
    return saida, Na
        
    
def int_br(x):
    return int(x.replace('.',''))

def float_br(x):
    return float(x.replace('.', '').replace(',','.'))


#edges for the age separated groups
age_edges = [60]
dia = '2805'

#Load the death notifications
file_obitos = '~/ownCloud/sesab/exporta_obitos_individualizados_csv_{}.csv'.format(dia)
data = pd.read_csv(file_obitos, sep=';', decimal=b',')
age = 'IDADE'
dday = 'DATA OBITO'
data[dday] = pd.to_datetime(data[dday], dayfirst=True)
data['DayNum'] = data[dday].dt.dayofyear

#load the Hospitalization Data
file_HU = '~/ownCloud/sesab/exporta_boletim_epidemiologico_csv_{}.csv'.format(dia)
datahu = pd.read_csv(file_HU, sep=';', decimal=b',')
rday = 'DATA DO BOLETIM'
datahu[rday] = pd.to_datetime(datahu[rday], dayfirst=True)
datahu['DayNum'] = datahu[rday].dt.dayofyear


#Load the age separated new cases
file_Nw = '~/ownCloud/sesab/exporta_boletim_faixa_etaria_csv_{}.csv'.format(dia)
dataNw = pd.read_csv(file_Nw, sep=';', decimal=',', names=['DATA', 'IDADE',\
                    'Nw', 'y', 'z', 'N'], converters={'DATA':str, 'IDADE':str,\
                    'Nw':int_br, 'y': float_br, 'z': float_br, \
                    'N': int_br})
nday = 'DATA'
dataNw[nday] = pd.to_datetime(dataNw[nday], dayfirst=True)
dataNw['DayNum'] = dataNw[nday].dt.dayofyear

#Option 1 use data assuming ignored is unbiased
idades, Ns, counts, Days = separate_age_groups(dataNw)
saida, Na = consolidate_age_groups(counts, idades, age_edges, Ns)


fday = max([data['DayNum'].min(), datahu['DayNum'].min()])

deaths = list()
for day in range(fday, data['DayNum'].max()+1):
    temp = [((data['DayNum'] <= day) & (data[age] < age_edges[0])).sum()]
    for i in range(1, len(age_edges)):
        temp = temp + [((data['DayNum'] <= day) & (data[age] < age_edges[i]) &
                        (data[age] >= age_edges[i-1])).sum()]
    temp = temp + [((data['DayNum'] <= day) & (data[age] >= age_edges[-1])).sum()]
    temp = temp + saida[Days==day,:].flatten().tolist()
    temp = temp + [datahu[datahu['DayNum'] == day]['CASOS ENFERMARIA'].max()]
    temp = temp + [datahu[datahu['DayNum'] == day]['CASOS UTI'].max()]

    deaths.append([day] + temp)

deaths = np.array(deaths)
cols = ['dayofyear']
for i in range(len(age_edges)+1):
    cols = cols + ['D_{}'.format(i)]
for i in range(len(age_edges)+1):
    cols = cols + ['Nw_{}'.format(i)]

cols = cols + ['H_ALL', 'U_ALL']

df = pd.DataFrame(deaths)
df.columns = cols



with open('test_export_data.pik', 'wb') as f:
    pickle.dump([df, Ns, Na, age_edges], f, protocol=-1)
#df_deaths.to_csv('obitos_sesab_sep_60_20200521.csv')