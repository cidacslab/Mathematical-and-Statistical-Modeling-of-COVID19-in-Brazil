# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:22:11 2020

@author: Rafael Veiga rafaelvalenteveiga@gmail.com
"""

import modelos as md
import datetime as dt
import pandas as pd
import sys

# parametros
modelo_usado = 'SIR' #EXP, SIR, SEIR_GA, SEIR_PSO ou SEQIJR_GA
stand_error = False # se true usa erro ponderado, se false usa erro simples
beta_variavel = True # funciona no SIR, caso True ocorre mudança do beta no dia definido no parametro abaixo
day_beta_change = None # funciona no SIR,dia da mudança do valor do beta se None a busca vai ser automatica (computacionalmente intensivo)
estados = ['RJ','BA'] # lista de estados, None para todos
numeroProcessadores = None # numero de prossesadores para executar em paralelo
min_cases = 5
min_dias = 10
arq_saida = '../data/estados.csv'
arq_sumario = '../data/estado_sumario.csv'
previsao_ate = dt.date(2020,12,24)


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
    if novo_nome[i]=='TOTAL':
        N_inicial = 217026005
    else:
        N_inicial = int(df_pop['Pop'][df_pop.Sigla==novo_nome[i]])
    print("\n\n"+str(novo_nome[i])+'\n')
    modelo = None
    if modelo_usado =='SIR':
        modelo = md.SIR(N_inicial,numeroProcessadores)
        bound = ([0,1/21],[1,1/5])
    elif modelo_usado =='EXP':
        modelo = md.EXP(N_inicial,numeroProcessadores)
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
    
    if modelo_usado == 'SIR':
        modelo.fit(x,y,bound = bound,stand_error=stand_error,beta2=beta_variavel,day_mudar =day_beta_change)
    else:
        modelo.fit(x,y,stand_error=stand_error)

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
    
df = pd.DataFrame()

if modelo_usado=='SIR':
    for i in range(len(modelos)):
        novo_local[i]['sucetivel'] = pd.to_numeric(pd.Series(modelos[i].S[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['infectado'] = pd.to_numeric(pd.Series(modelos[i].I[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['recuperado'] = pd.to_numeric(pd.Series(modelos[i].R[0:len(novo_local[i].TOTAL)]),downcast='integer')
    
if modelo_usado=='SEIR_PSO' or modelo_usado=='SEIR_GA':
    for i in range(len(modelos)):
        novo_local[i]['sucetivel'] = pd.to_numeric(pd.Series(modelos[i].S[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['exposto'] = pd.to_numeric(pd.Series(modelos[i].E[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['infectado'] = pd.to_numeric(pd.Series(modelos[i].I[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['recuperado'] = pd.to_numeric(pd.Series(modelos[i].R[0:len(novo_local[i].TOTAL)]),downcast='integer')
for i in range(0,len(novo_local)):
    df = df.append(novo_local[i],ignore_index=True)

df.to_csv(arq_saida,index=False)

su = pd.DataFrame()
su['state'] = novo_nome
pop = []
rmse = []
coef_list = []

coef_name = None
for i in range(len(novo_nome)):
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

