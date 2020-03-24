# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:22:11 2020

@author: Rafael Veiga
"""

import modelos as md
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# parametros
min_cases = 5
min_dias = 10
arq_saida = 'estados.csv'
arq_sumario = 'estado_sumario.csv'
arq_brasil_saida = 'brasil.csv'
previsao_ate = dt.date(2020,3,25)

#carregar dados
nome, local = md.ler_banco()

novo_nome = []
nome.insert(0,'Brasil')
novo_local = []
#brasil
br = local[0]
for i in range(1,len(local)):
    br = br.append(local[i],ignore_index=True)
x = list(br.date.unique())
x = sorted(x)
y = []
for d in x:
    y.append(sum(br[br.date==d].totalcasos))

newy = [y[0]]
for i in range(1,len(y)):
    newy.append(y[i]-y[i-1])
state = ['Brasil' for i in y]
UF = [np.nan for i in y]
brasil = pd.DataFrame({'date':x,'state':state,'UF':UF,'novosCasos':newy,'totalcasos':y})
local.insert(0,brasil)
for i in range(len(nome)):
    if (local[i].totalcasos.iloc[-1]>=min_cases) & (len(local[i])>=10):
        novo_local.append(local[i])
        novo_nome.append(nome[i])
previsao_ate = previsao_ate + dt.timedelta(1)
modelos = []
for i in range(len(novo_nome)):
    print("\n\n"+str(nome[i])+'\n')
    modelo = md.SEQIJR_EDO(10000)   # SIR, SIR_EDO ou SEQIJR_EDO
    y = novo_local[i].totalcasos
    x = range(1,len(y)+1)
    modelo.fit(x,y)
    modelos.append(modelo)
    dias = (previsao_ate-novo_local[i].date.iloc[0]).days
    x_pred = range(1,dias+1)
    y_pred =modelo.predict(x_pred)
    plt.plot(y_pred,c='r',label='Predição Infectados')
    plt.plot(y,c='b',label='Infectados')
    plt.legend(fontsize=15)
    plt.title('Dinâmica do CoviD19 - {}'.format(nome[i]),fontsize=20)
    plt.ylabel('Casos COnfirmados',fontsize=15)
    plt.xlabel('Dias',fontsize=15)
    plt.show()
    
    novo_local[i]['casos_preditos'] = y_pred[0:len(novo_local[i])]
    ultimo_dia = novo_local[i].date.iloc[-1]
    dias = (previsao_ate-novo_local[i].date.iloc[-1]).days
    for d in range(1,dias):
        di = d+len(x)-1
        novo_local[i]=novo_local[i].append({'casos_preditos':y_pred[di],'date':ultimo_dia+dt.timedelta(d),'state':novo_local[i].state.iloc[0]}, ignore_index=True)
    
brasil =   novo_local[0]
del brasil['UF']  
brasil.to_csv(arq_brasil_saida,index=False)    
df = novo_local[1]
for i in range(2,len(novo_local)):
    df = df.append(novo_local[i],ignore_index=True)
df.to_csv(arq_saida,index=False)

su = pd.DataFrame()
su['state'] = novo_nome
a = []
b = []
rmse = []
y = []
for i in range(len(novo_nome)):
    a.append(modelos[i].a)
    b.append(modelos[i].b)
    rmse.append(modelos[i].rmse)
    y.append(';'.join(map(str, modelos[i].y)))
su['coef_a'] = a
su['coef_b'] = b
su['rmse'] = rmse
su['y'] = y
su.to_csv(arq_sumario,index=False)