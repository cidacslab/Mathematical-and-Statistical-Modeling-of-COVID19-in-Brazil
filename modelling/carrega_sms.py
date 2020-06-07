#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:19:41 2020

@author: Felipe A. C. Pereira
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

def separate_age_groups(data):
    idades_0 = data['ageRange'].unique()
    idades = list()
    for idade in idades_0:
        if idade[0] == '>':
            idades.append([80, idade])
        elif idade == 'SI':
            idades.append([np.nan, idade])
        else:
            idades.append([int(idade.split(' ')[0]), idade])
    DayNums = data['DayNum'].unique()
    DayNums.sort()
#    print(idades)
    saida = np.zeros((len(DayNums), len(idades)), dtype=int)
#    Ns = list()
    for j, idade in enumerate(idades):
#        Ns.append(int(data[data['IDADE'] == idade[1]]['N'].mean()))
        for i, dia in enumerate(DayNums):
            if len(data[(data['ageRange'] == idade[1]) & \
                 (data['DayNum'] == dia)]) > 0:
                saida[i,j] = data[(data['ageRange'] == idade[1]) & \
                     (data['DayNum'] == dia)]['totalCases'].max()
    return idades, saida, DayNums

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
        
def remove_inconsistent_data(counts):
    corr_counts = np.zeros_like(counts, dtype=float)
    N0 = 0
    for i in range(counts.shape[0]):
        Ni = counts[i].sum()
        if Ni >= N0:
            corr_counts[i] = counts[i]
            N0 = Ni
        else:
            corr_counts[i] = np.nan
    return corr_counts
 
def int_br(x):
    return int(x.replace('.',''))

def float_br(x):
    return float(x.replace('.', '').replace(',','.'))


#edges for the age separated groups
age_edges = [60]
#dia = '2805'

#Load the death notifications
file_obitos = '~/ownCloud/sms/obitos/obitos_SSA_01_06_2020_COVID19_edited.xlsx'
data = pd.read_excel(file_obitos)
age = 'IDADE'
dday = 'DTOBITO/HORARIO'

data[dday] = data[dday].fillna(method='ffill')
data[dday] = pd.to_datetime(data[dday])

data['DayNum'] = data[dday].dt.dayofyear

#load the Hospitalization Data
file_HU = '~/ownCloud/sms/leitos/CIDACS_04_06_as_19_52.xlsx'
datahu = pd.read_excel(file_HU)
rday='DATAHORA_INFORMACAO_RECEBIDA'
datahu[rday] = pd.to_datetime(pd.Timestamp(year=1900, month=1, day=1) + pd.to_timedelta(datahu[rday]-2, unit='d'))
#datahu = pd.read_csv(file_HU, sep=';', decimal=b',')
#rday = 'DATA DO BOLETIM'
#datahu[rday] = pd.to_datetime(datahu[rday], dayfirst=True)
datahu['DayNum'] = datahu[rday].dt.dayofyear
datahu['U'] = datahu['QTD_OCUPACAO_UTI_ADULTO'] + datahu['QTD_OCUPACAO_UTI_PEDIATRICO']
datahu['H'] = datahu['QTD_OCUPACAO_CLINICO_ADULTO'] + datahu['QTD_OCUPACAO_CLINICO_PEDIATRICO']

#Load the age separated new cases
file_Nw = '~/ownCloud/sms/dataHistAge_0406.csv'
dataNw = pd.read_csv(file_Nw, sep=',')
nday = 'date'
dataNw[nday] = pd.to_datetime(dataNw[nday], yearfirst=True)
dataNw['DayNum'] = dataNw[nday].dt.dayofyear
dataNw = dataNw.drop(dataNw[dataNw['ageRange'] == 'Total'].index)
#
##Option 1 use data assuming ignored is unbiased
#population SSA
Ns0 = np.array([[	0	,	147488.3	]	,
	[	5	,	153995.77	]	,
	[	10	,	372124.97	]	,
	[	20	,	466648.6	]	,
	[	30	,	514911.6	]	,
	[	40	,	448914.82	]	,
	[	50	,	339514.84	]	,
	[	60	,	227555.3	]	,
	[	70	,	109611.09	]	,
	[	80	,	50792.335	]	]
)

idades, counts, Days = separate_age_groups(dataNw)
Ns = [Ns0[np.flatnonzero(Ns0[:,0] == idade[0]),1] for idade in idades]
Ns = [N[0] if len(N) == 1 else 0 for N in Ns]
#saida, Na = consolidate_age_groups(counts, idades, age_edges, Ns)

ccounts = remove_inconsistent_data(counts)
saida, Na = consolidate_age_groups(ccounts, idades, age_edges, Ns)

#
plt.plot(Days, ccounts, '.-')
plt.plot(Days, ccounts.sum(axis=1), '.-k')
plt.legend([ida[1] for ida in idades] + ['acumulado'])
plt.ylabel('Freq')
plt.xlabel('Dia do Ano')
fday = min([data['DayNum'].min(), dataNw['DayNum'].min()])
#fday = data['DayNum'].min()
deaths = list()
for day in range(fday, data['DayNum'].max()+1):
    temp = [((data['DayNum'] <= day) & (data[age] < age_edges[0])).sum()]
    for i in range(1, len(age_edges)):
        temp = temp + [((data['DayNum'] <= day) & (data[age] < age_edges[i]) &
                        (data[age] >= age_edges[i-1])).sum()]
    temp = temp + [((data['DayNum'] <= day) & (data[age] >= age_edges[-1])).sum()]
    if len(saida[Days==day,:]) > 0:
        temp = temp + saida[Days==day,:].flatten().tolist()
    else:
        temp = temp + saida.shape[1] * [np.nan]
    temp = temp + [datahu[datahu['DayNum'] == day]['H'].max()]
    temp = temp + [datahu[datahu['DayNum'] == day]['U'].max()]
#
    deaths.append([day] + temp)
#
deaths = np.array(deaths)
cols = ['dayofyear']
for i in range(len(age_edges)+1):
    cols = cols + ['D_{}'.format(i)]
for i in range(len(age_edges)+1):
    cols = cols + ['Nw_{}'.format(i)]
#
cols = cols + ['H_ALL', 'U_ALL']
#
df = pd.DataFrame(deaths)
df.columns = cols
#
#
#
ages = np.array([0,5,10,20,30,40,50,60,70,80])
N0 = np.array([N[1] for N in Ns0])
parspad = {'p':np.array([0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.8]),
           'h':np.array([0.001, 0.001, 0.003, 0.012, 0.032, 0.049, 0.102, 0.166, 0.243, 0.273]),
           'xi':np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.063, 0.122, 0.274, 0.432, 0.709])       
        }
par_age = dict()
for ke in parspad.keys():
    idx = ages<age_edges[0]
    temp = [(parspad[ke][idx] * N0[idx]).sum()/Na[0]]
    for i in range(1, len(age_edges)):
        idx = (ages<age_edges[i]) * (ages >= age_edges[i-1])
        temp = temp + [(parspad[ke][idx] * N0[idx]).sum()/Na[i]]
    idx = (ages >= age_edges[-1])   
    temp = temp + [(parspad[ke][idx] * N0[idx]).sum()/Na[-1]]
    par_age[ke] = np.array(temp)
with open('test_export_data_SMS_0506.pik', 'wb') as f:
    pickle.dump([df, par_age, Na, age_edges], f, protocol=2)

#df.to_csv('SMS_data_to_fit.csv')
#df_deaths.to_csv('obitos_sesab_sep_60_20200521.csv')