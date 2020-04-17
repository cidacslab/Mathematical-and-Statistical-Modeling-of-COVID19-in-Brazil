# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:22:11 2020

@author: rafae rafaelvalenteveiga@gmail.com
"""

import modelos as md
import datetime as dt
import pandas as pd
import sys

# parametros
modelo_usado = 'EXP' #EXP, SIR_PSO, SIR_PSO_padro, SIR_GA, SIR_GA_fit_I ou SEQIJR_GA
stand_error = True # se true usa erro ponderado, se false usa erro simples
beta_variavel = False # funciona no SIR, caso True ocorre mudança do beta no dia definido no parametro abaixo
day_beta_change = 6 # funciona no SIR,dia da mudança do valor do beta se None a busca vai ser automatica (computacionalmente intensivo)
numeroProcessadores = None # numero de prossesadores para executar em paralelo
N_inicial = 1000
min_cases = 5
min_dias = 10
arq_saida = '../data/municipios.csv'
arq_sumario = '../data/municipios_sumario.csv'
previsao_ate = dt.date(2020,4,24)

#carregar dados
nome, local = md.ler_banco_municipios()
df = pd.read_csv('../data/populacao_municipio.csv')
pop_local =df.Armenor.unique()
pop = {}
for o in pop_local:
    aux = df[df.Armenor==o]
    pop[o] = sum(aux.Total)
    
novo_nome = []
novo_local = []

for i in range(len(nome)):
    if (local[i].TOTAL.iloc[-1]>=min_cases) & (len(local[i])>=10):
        novo_local.append(local[i])
        novo_nome.append(nome[i])
previsao_ate = previsao_ate + dt.timedelta(1)
modelos=[]
for i in range(len(novo_nome)):
    print("\n"+str(novo_local[i].city[0])+'\n')
    modelo = None
    N = 0
    if novo_nome[i] in pop:
        N = pop[novo_nome[i]]
    else:
        print('não achou pop da cidade '+str(novo_nome[i]))
        N=10000
    if modelo_usado =='SIR_PSO':
        modelo = md.SIR_PSO(N,numeroProcessadores)
    elif modelo_usado =='SIR_PSO_padro':
        modelo = md.SIR_PSO_padro(N,numeroProcessadores)
    elif modelo_usado =='SIR_PSO_beta_variante':
        modelo = md.SIR_PSO_beta_variante(N,numeroProcessadores)
    elif modelo_usado =='SIR_GA_fit_I':
        modelo = md.SIR_GA_fit_I(N)
    elif modelo_usado =='SIR_GA':
        modelo = md.SIR_GA(N)
    elif modelo_usado =='EXP':
        modelo = md.EXP(N,numeroProcessadores)
    elif modelo_usado =='SEIR_PSO':
        modelo = md.SEIR_PSO(N)
    elif modelo_usado=='SEQIJR_GA':
        modelo = md.SEQIJR_GA(N)
    else:
        print('Modelo desconhecido '+modelo_usado)
        sys.exit(1)
    
    y = novo_local[i].TOTAL
    x = range(1,len(y)+1)
    if modelo_usado == 'SIR':
        modelo.fit(x,y,stand_error=stand_error,beta2=beta_variavel,day_mudar =day_beta_change)
    else:
        modelo.fit(x,y,stand_error=stand_error)

    modelos.append(modelo)
    dias = (previsao_ate-novo_local[i].date.iloc[0]).days
    x_pred = range(1,dias+1)
    y_pred =modelo.predict(x_pred)
    novo_local[i]['totalCasesPred'] = y_pred[0:len(novo_local[i])]
    novo_local[i]['residuo_quadratico'] = modelo.getResiduosQuadatico()
    novo_local[i]['res_quad_padronizado'] = modelo.getReQuadPadronizado()
    ultimo_dia = novo_local[i].date.iloc[-1]
    dias = (previsao_ate-novo_local[i].date.iloc[-1]).days
    for d in range(1,dias):
        di = d+len(x)-1
        novo_local[i]=novo_local[i].append({'totalCasesPred':y_pred[di],'date':ultimo_dia+dt.timedelta(d),'UF':novo_local[i].UF.iloc[0],'city':novo_local[i].city.iloc[0],'ibgeID':novo_local[i].ibgeID.iloc[0]}, ignore_index=True)
    
    novo_local[i].ibgeID = pd.to_numeric(novo_local[i].ibgeID,downcast='integer')
    
df = novo_local[0]
if modelo_usado=='SIR_PSO' or modelo_usado=='SIR_GA' or modelo_usado=='SIR_GA_fit_I'or modelo_usado=='SIR_PSO_beta_variante':
    for i in range(0,len(modelos)):
        novo_local[i]['sucetivel'] = pd.to_numeric(pd.Series(modelos[i].S[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['infectado'] = pd.to_numeric(pd.Series(modelos[i].I[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['Recuperado'] = pd.to_numeric(pd.Series(modelos[i].R[0:len(novo_local[i].TOTAL)]),downcast='integer')
    
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
su['ibgeID'] = novo_nome
pop = []
rmse = []
coef_list = []
y = []
coef_name = None
for i in range(len(novo_nome)):
    y.append(';'.join(map(str, modelos[i].y)))
    coef_name, coef  = modelos[i].getCoef()
    coef_list.append(coef)
    pop.append(modelos[i].N)
    rmse.append(modelos[i].rmse)
su['populacao']= pop
su['rmse'] = rmse
for c in range(len(coef_name)):
    l = []
    for i in range(len(coef_list)):
        l.append(coef_list[i][c])
    su[coef_name[c]]=l

su.to_csv(arq_sumario,index=False)

