# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:22:11 2020

@author: Rafael Veiga rafaelvalenteveiga@gmail.com
"""

import modelos as md
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# parametros
modelo_usado = 'SIR_PSO_beta_variante' #EXP, SIR_PSO, SIR_PSO_padro, SIR_PSO_beta_variante, SIR_GA , SIR_GA_fit_I, SEIR_GA, SEIR_PSO ou SEQIJR_GA
day_beta_change = 6 # dia da mudança do valor do beta se None a busca vai ser automatica (computacionalmente intensivo)
estados = ['TOTAL'] # lista de estados, None para todos
numeroProcessadores = None # numero de prossesadores para executar em paralelo
min_cases = 5
min_dias = 10
arq_saida = '../data/estados.csv'
arq_sumario = '../data/estado_sumario.csv'
arq_brasil_saida = '../data/brasil.csv'
previsao_ate = dt.date(2020,4,24)


#carregar dados
nome, local = md.ler_banco_estados()
df_pop = pd.read_csv('../data/populacoes.csv')
#carregar dados alternativa
#timeseries,populacao = md.ler_banco_alternativa()

novo_nome = []
novo_local = []

for i in range(len(nome)):
    if (local[i].TOTAL.iloc[-1]>=min_cases) & (len(local[i])>=10) & (estados==None):
        novo_local.append(local[i])
        novo_nome.append(nome[i])
    elif estados!= None:
        if nome[i] in estados:
            novo_nome.append(nome[i])
            novo_local.append(local[i])
previsao_ate = previsao_ate + dt.timedelta(1)
modelos = []
N_inicial = 0
for i in range(len(novo_nome)):
    if i==0:
        N_inicial = 217026005
    else:
        N_inicial = int(df_pop['Pop'][df_pop.Sigla==novo_nome[i]])
    print("\n\n"+str(novo_nome[i])+'\n')
    modelo = None
    if modelo_usado =='SIR_PSO':
        modelo = md.SIR_PSO(N_inicial,numeroProcessadores)
    elif modelo_usado =='SIR_PSO_padro':
        modelo = md.SIR_PSO_padro(N_inicial,numeroProcessadores)
    elif modelo_usado =='SIR_PSO_beta_variante':
        modelo = md.SIR_PSO_beta_variante(N_inicial,numeroProcessadores)
    elif modelo_usado =='EXP':
        modelo = md.EXP(N_inicial,numeroProcessadores)
    elif modelo_usado =='SIR_GA':
        modelo = md.SIR_GA(N_inicial)
    elif modelo_usado =='SIR_GA_fit_I':
        modelo = md.SIR_GA_fit_I(N_inicial)
    elif modelo_usado =='SEIR_PSO':
        modelo = md.SEIR_PSO(N_inicial)
    elif modelo_usado =='SEIR_GA':
        modelo = md.SEIR_GA(N_inicial)
    elif modelo_usado=='SEQIJR_GA':
        modelo = md.SEQIJR_GA(N_inicial)
    else:
        print('Modelo desconhecido '+modelo_usado)
        sys.exit(1)
    
    y = novo_local[i].TOTAL
    x = range(1,len(y)+1)
    
    if modelo_usado == 'SIR_PSO_beta_variante':
        if day_beta_change==None:
            modelo.fit_busca_dia(x,y,name=novo_nome[i])
        else:
            modelo.fit(x,y,name=novo_nome[i],day_mudar=day_beta_change)
    else:
        modelo.fit(x,y,name=novo_nome[i])

    modelos.append(modelo)
    dias = (previsao_ate-novo_local[i].date.iloc[0]).days
    x_pred = range(1,dias+1)
    y_pred =modelo.predict(x_pred)
#    plt.plot(y_pred,c='r',label='Predição Infectados')
#    plt.plot(y,c='b',label='Infectados')
#    plt.legend(fontsize=15)
#    plt.title('Dinâmica do CoviD19 - {}'.format(nome[i]),fontsize=20)
#    plt.ylabel('Casos COnfirmados',fontsize=15)
#    plt.xlabel('Dias',fontsize=15)
#    plt.show()    
    novo_local[i]['totalCasesPred'] = y_pred[0:len(novo_local[i])]
    novo_local[i]['residuo_quadratico'] = modelo.getResiduosQuadatico()
    novo_local[i]['res_quad_padronizado'] = modelo.getReQuadPadronizado()
    ultimo_dia = novo_local[i].date.iloc[-1]
    dias = (previsao_ate-novo_local[i].date.iloc[-1]).days
    for d in range(1,dias):
        di = d+len(x)-1
        novo_local[i]=novo_local[i].append({'totalCasesPred':y_pred[di],'date':ultimo_dia+dt.timedelta(d),'state':novo_local[i].state.iloc[0]}, ignore_index=True)
    
brasil =   novo_local[0]
if modelo_usado=='SIR_PSO' or modelo_usado=='SIR_GA' or modelo_usado=='SIR_GA_fit_I' or modelo_usado=='SIR_PSO_beta_variante':
    brasil['sucetivel'] = pd.to_numeric(pd.Series(modelos[0].S[0:len(brasil.TOTAL)]),downcast='integer')
    brasil['infectado'] =  pd.to_numeric(pd.Series(modelos[0].I[0:len(brasil.TOTAL)]),downcast='integer') 
    brasil['recuperado'] =  pd.to_numeric(pd.Series(modelos[0].R[0:len(brasil.TOTAL)]),downcast='integer') 
if modelo_usado=='SEIR_PSO' or modelo_usado=='SEIR_GA':
    brasil['sucetivel'] = pd.to_numeric(pd.Series(modelos[0].S[0:len(brasil.TOTAL)]),downcast='integer')
    brasil['exposto'] =  pd.to_numeric(pd.Series(modelos[0].E[0:len(brasil.TOTAL)]),downcast='integer') 
    brasil['infectado'] =  pd.to_numeric(pd.Series(modelos[0].I[0:len(brasil.TOTAL)]),downcast='integer') 
    brasil['recuperado'] =  pd.to_numeric(pd.Series(modelos[0].R[0:len(brasil.TOTAL)]),downcast='integer') 

brasil.to_csv(arq_brasil_saida,index=False)    
df = novo_local[0]
if modelo_usado=='SIR_PSO' or modelo_usado=='SIR_GA' or modelo_usado=='SIR_GA_fit_I'or modelo_usado=='SIR_PSO_beta_variante':
    for i in range(len(modelos)):
        novo_local[i]['sucetivel'] = pd.to_numeric(pd.Series(modelos[i].S[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['infectado'] = pd.to_numeric(pd.Series(modelos[i].I[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['recuperado'] = pd.to_numeric(pd.Series(modelos[i].R[0:len(novo_local[i].TOTAL)]),downcast='integer')
    for i in range(1,len(novo_local)):
        df = df.append(novo_local[i],ignore_index=True)
if modelo_usado=='SEIR_PSO' or modelo_usado=='SEIR_GA':
    for i in range(len(modelos)):
        novo_local[i]['sucetivel'] = pd.to_numeric(pd.Series(modelos[i].S[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['exposto'] = pd.to_numeric(pd.Series(modelos[i].E[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['infectado'] = pd.to_numeric(pd.Series(modelos[i].I[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['recuperado'] = pd.to_numeric(pd.Series(modelos[i].R[0:len(novo_local[i].TOTAL)]),downcast='integer')
    for i in range(1,len(novo_local)):
        df = df.append(novo_local[i],ignore_index=True)

df.to_csv(arq_saida,index=False)

su = pd.DataFrame()
su['state'] = novo_nome
pop = []
rmse = []
coef_list = []
y = []
coef_name = None
for i in range(len(novo_nome)):
    y.append(';'.join(map(str, modelos[i].y)))
    coef_name, coef  = modelos[i].getCoef()
    rmse.append(modelos[i].rmse)
    coef_list.append(coef)
    pop.append(modelos[i].N)
su['populacao']= pop
su['rmse'] = rmse
for c in range(len(coef_name)):
    l = []
    for i in range(len(coef_list)):
        l.append(coef_list[i][c])
    su[coef_name[c]]=l

su.to_csv(arq_sumario,index=False)

