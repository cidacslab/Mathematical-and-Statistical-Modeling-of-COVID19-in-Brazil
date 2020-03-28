# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:22:11 2020

@author: rafae
"""

import modelos as md
import datetime as dt
import pandas as pd
import sys

# parametros
modelo_usado = 'SIR' # SIR, SIR_EDO ou SEQIJR_EDO
N_inicial = 1000
min_cases = 5
min_dias = 10
arq_saida = '../data/municipios.csv'
arq_sumario = '../data/municipios_sumario.csv'
previsao_ate = dt.date(2020,3,25)

#carregar dados
nome, local = md.ler_banco('../data/datamun.csv','cod_city')

novo_nome = []
novo_local = []

for i in range(len(nome)):
    if (local[i].totalcasos.iloc[-1]>=min_cases) & (len(local[i])>=10):
        novo_local.append(local[i])
        novo_nome.append(nome[i])
previsao_ate = previsao_ate + dt.timedelta(1)
modelos=[]
for i in range(len(novo_nome)):
    print("\n"+str(novo_local[i].city[0])+'\n')
    modelo = None
    if modelo_usado =='SIR':
        modelo = md.SIR(N_inicial)
    elif modelo_usado =='SIR_EDO':
        modelo = md.SIR_EDO(N_inicial)
    elif modelo_usado=='SEQIJR_EDO':
        modelo = md.SEQIJR_EDO(N_inicial)
    else:
        print('Modelo desconhecido '+modelo_usado)
        sys.exit(1)
    # SIR, SIR_EDO ou SEQIJR_EDO
    y = novo_local[i].totalcasos
    x = range(1,len(y)+1)
    modelo.fit(x,y)
    modelos.append(modelo)
    dias = (previsao_ate-novo_local[i].date.iloc[0]).days
    x_pred = range(1,dias+1)
    y_pred =modelo.predict(x_pred)
    novo_local[i]['casos_preditos'] = y_pred[0:len(novo_local[i])]
    ultimo_dia = novo_local[i].date.iloc[-1]
    dias = (previsao_ate-novo_local[i].date.iloc[-1]).days
    for d in range(1,dias):
        di = d+len(x)-1
        novo_local[i]=novo_local[i].append({'casos_preditos':y_pred[di],'date':ultimo_dia+dt.timedelta(d),'state':novo_local[i].state.iloc[0],'UF':novo_local[i].UF.iloc[0],'city':novo_local[i].city.iloc[0],'cod_city':novo_local[i].cod_city.iloc[0]}, ignore_index=True)
    
    novo_local[i].cod_city = pd.to_numeric(novo_local[i].cod_city,downcast='integer')
    
df = novo_local[0]
for i in range(1,len(novo_local)):
    df = df.append(novo_local[i],ignore_index=True)
df.to_csv(arq_saida)

su = pd.DataFrame()
su['state'] = novo_nome
pop = []
coef_list = []
y = []
coef_name = None
for i in range(len(novo_nome)):
    y.append(';'.join(map(str, modelos[i].y)))
    coef_name, coef  = modelos[i].getCoef()
    coef_list.append(coef)
    pop.append(modelos[i].N)
su['populacao']= pop
for c in range(len(coef_name)-1):
    l = []
    for i in range(len(coef_list)):
        l.append(coef_list[i][c])
    su[coef_name[c]]=l
coef_name = coef_name[-1]
for c in range(len(coef_name)):
    l=[]
    for i in range(len(coef_list)):
       l.append(str(list(coef_list[i][-1][c])).replace('  ',' ').replace(' ',';').replace('[','').replace(']','').replace('\n','').replace(',','')) 
    su[coef_name[c]] = l

su['y'] = y
su.to_csv(arq_sumario,index=False)

