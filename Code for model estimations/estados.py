# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:22:11 2020

@author: rafae
"""

import modelos as md
import datetime as dt

# parametros
min_cases = 5
min_dias = 10
arq_saida = 'estados.csv'
previsao_ate = dt.date(2020,3,25)

#carregar dados
nome, local = md.ler_banco()

novo_nome = []
novo_local = []

for i in range(len(nome)):
    if (local[i].totalcasos.iloc[-1]>=min_cases) & (len(local[i])>=10):
        novo_local.append(local[i])
        novo_nome.append(nome[i])
previsao_ate = previsao_ate + dt.timedelta(1)

for i in range(len(novo_nome)):
    print("\n"+str(nome[i])+'\n')
    modelo = md.SIR(10000)
    y = novo_local[i].totalcasos
    x = range(1,len(y)+1)
    modelo.fit(x,y)
    dias = (previsao_ate-novo_local[i].date.iloc[0]).days
    x_pred = range(1,dias+1)
    y_pred =modelo.predict(x_pred)
    novo_local[i]['casos_preditos'] = y_pred[0:len(novo_local[i])]
    ultimo_dia = novo_local[i].date.iloc[-1]
    dias = (previsao_ate-novo_local[i].date.iloc[-1]).days
    for d in range(1,dias):
        di = d+len(x)-1
        novo_local[i]=novo_local[i].append({'casos_preditos':y_pred[di],'date':ultimo_dia+dt.timedelta(d),'state':novo_local[i].state.iloc[0]}, ignore_index=True)
    
    
    
df = novo_local[0]
for i in range(1,len(novo_local)):
    df = df.append(novo_local[i],ignore_index=True)
df.to_csv(arq_saida)