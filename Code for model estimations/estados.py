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
    modelo1 = md.SIR(14873064)       
    modelo2 = md.SIR_EDO(14873064)  
    modelo3 = md.SEIR_EDO(14873064) 
    modelo4 = md.SEQIJR_EDO(14873064) 
    y = novo_local[i].totalcasos
    x = range(1,len(y)+1)
    modelo1.fit(x,y)
    modelo2.fit(x,y)
    modelo3.fit(x,y)
    modelo4.fit(x,y)
    modelos.append(modelo1)
    dias = (previsao_ate-novo_local[i].date.iloc[0]).days
    x_pred = range(1,dias+1)
    y_pred1 =modelo1.predict(x_pred)
    y_pred2 =modelo2.predict(x_pred)
    y_pred3 =modelo3.predict(x_pred)
    y_pred4 =modelo4.predict(x_pred)
    
    fig = plt.figure(figsize=(7,5))
    plt.plot(y_pred1,c='g',label='SIR_exp')
    plt.text(x_pred[-1]-1,y_pred1[-1],int(y_pred1[-1])) 
    plt.plot(y_pred2,c='r',label='SIR')
    plt.text(x_pred[-1],y_pred2[-1],int(y_pred2[-1])) 
    plt.plot(y_pred3,c='c',label='SEIR')
    plt.text(x_pred[-1],y_pred3[-1],int(y_pred3[-1])) 
    plt.plot(y_pred4,c='y',label='SEQIJR')
    plt.text(x_pred[-1],y_pred4[-1],int(y_pred4[-1])) 
    plt.plot(y,c='b',label='Infectados - {}'.format(nome[i]),linewidth = 3)
    plt.text(len(x)-1,y.tolist()[-1],int(y.tolist()[-1])) 
    plt.legend(fontsize=12)
    plt.title('Dinâmica do CoviD19 - {}'.format(nome[i]),fontsize=18)
    plt.ylabel('Casos Confirmados',fontsize=15)
    plt.xlabel('Dias',fontsize=15)
    plt.grid(alpha = 0.5,which='both')
    fig.savefig('\plots\{}.png'.format(nome[i]), bbox_inches='tight')
    plt.show()
    novo_local[i]['casos_preditos'] = y_pred1[0:len(novo_local[i])]
    ultimo_dia = novo_local[i].date.iloc[-1]
    dias = (previsao_ate-novo_local[i].date.iloc[-1]).days
    for d in range(1,dias):
        di = d+len(x)-1
        novo_local[i]=novo_local[i].append({'casos_preditos':y_pred1[di],'date':ultimo_dia+dt.timedelta(d),'state':novo_local[i].state.iloc[0]}, ignore_index=True)
    
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