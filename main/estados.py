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
modelo_usado = 'SEIRHUD' #EXP, SIR, SEIR, SEIRHUD
stand_error = True # se true usa erro ponderado, se false usa erro simples
beta_variavel = True # funciona no SIR, caso True ocorre mudança do beta no dia definido no parametro abaixo
day_beta_change = None # funciona no SIR,dia da mudança do valor do beta se None a busca vai ser automatica
intervalo_data = (dt.date(2020,3,15),dt.date(2020,4,15))
estados = ['TOTAL','BA'] # lista de estados, None para todos
numeroProcessadores = None # numero de prossesadores para executar em paralelo
min_cases = 5
min_dias = 10
arq_saida = '../data/estados.csv'
arq_sumario = '../data/estado_sumario.csv'
previsao_ate = dt.date.today() +  dt.timedelta(200)

def ler_banco_estados():
    try:
        url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv'
        banco = pd.read_csv(url)
    except:
        return None,None
    
    

    nome_local =list(banco['state'].unique())
    nome_local.remove('TOTAL')
    nome_local.insert(0,'TOTAL')
    for i in banco.index:
        banco.date[i] = dt.datetime.strptime(banco.date[i], '%Y-%m-%d').date()
    local = []
    for est in nome_local:
        
    
        aux = banco[banco['state']==est].sort_values('date')
        data_ini = aux.date.iloc[0]
        data_fim = aux.date.iloc[-1]
        dias = (data_fim-data_ini).days + 1
        d = [(data_ini + dt.timedelta(di)) for di in range(dias)]
        
        estado = [est for di in range(dias)]
        df = pd.DataFrame({'date':d,'state':estado})
        
        casos = []
        mortes = []
        caso = 0
        morte=0
        i_aux = 0
        for i in range(dias):
            if (d[i]-aux.date.iloc[i_aux]).days==0:
                caso = aux['totalCases'].iloc[i_aux]
                morte = aux.deaths.iloc[i_aux]
                casos.append(caso)
                mortes.append(morte)
                i_aux=i_aux+1
            else:
                casos.append(caso)
                mortes.append(morte)
        new = [casos[0]]        
        for i in range(1,dias):
            new.append(casos[i]-casos[i-1])
        df['newCases'] = new
        df['mortes'] = mortes
        df['TOTAL'] = casos
        local.append(df)
        nome_local[0]='TOTAL'
        local[0].state='TOTAL'
    return nome_local, local   


#carregar dados
nome, local = ler_banco_estados()
df_pop = pd.read_csv('../data/Populacoes.csv')
#carregar dados alternativa
#timeseries,populacao = md.ler_banco_alternativa()

novo_nome = []
novo_local = []
for i in range(len(nome)):
    if (local[i].TOTAL.iloc[-1]>=min_cases) & (len(local[i])>=min_dias) & (estados==None):
        novo_local.append(local[i])
        novo_nome.append(nome[i])
    elif estados!= None:
        if nome[i] in estados:
            novo_nome.append(nome[i])
            novo_local.append(local[i])
previsao_ate = previsao_ate + dt.timedelta(1)
#extrair data bound

diaBound = []
for i in range(len(novo_nome)):
    dia_ini = novo_local[i].date.iloc[0]
    dia_fim = novo_local[i].date.iloc[-1]
    #print('\n'+str(dia_ini) + " " + str(dia_fim) + '\n')
    ini = (intervalo_data[0]-dia_ini).days if dia_ini < intervalo_data[0] else 5
    fim = (intervalo_data[1]-dia_ini).days -5
    ini = 5 if ini < 5 else ini
   # print(str(ini) + '\n')
    #print(str(fim) + '\n')
   # print(str((dia_fim - dia_ini).days))
    diaBound.append((ini,fim,dia_ini))
    if ini > fim:
        beta_variavel = False
    
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
    elif modelo_usado =='EXP':
        modelo = md.EXP(N_inicial,numeroProcessadores)
    elif modelo_usado =='SEIR':
        modelo = md.SEIR(N_inicial,numeroProcessadores)
    elif modelo_usado =='SEIRHUD':
        modelo = md.SEIRHUD(N_inicial,numeroProcessadores)
    
    else:
        print('Modelo desconhecido '+modelo_usado)
        sys.exit(1)
    
    y = novo_local[i].TOTAL
    d = novo_local[i].mortes
    x = range(1,len(y)+1)
    
    if modelo_usado == 'SIR':
        if beta_variavel:
            if day_beta_change==None:
                bound = [[0,1/21,0,diaBound[i][0]],[1,1/5,1,diaBound[i][1]]]
            else:
                bound = [[0,1/21,0],[1,1/5,1]]
            modelo.fit(x,y,stand_error=stand_error,beta2=beta_variavel,day_mudar =day_beta_change,bound = bound)
        else:
            modelo.fit(x,y,stand_error=stand_error,beta2=beta_variavel,day_mudar =day_beta_change)
    elif modelo_usado == 'SEIRHUD':
        if beta_variavel:
            if day_beta_change==None:
                #[[0,0,diaBound[i][0],1/8,1/12,0,0.05],[2,2,diaBound[i][1],1/4,1/3,0.7,0.25]]
                bound = [[0,0,diaBound[i][0],1/8,1/12,0,0.05,0,0,0],[1,1,diaBound[i][1],1/4,1/3,0.7,0.25,10/N_inicial,10/N_inicial,10/N_inicial]]
            else:
                bound = [[0,0,1/8,1/12,0,0.05,0,0,0],[1,1,1/4,1/3,0.7,0.25,10/N_inicial,10/N_inicial,10/N_inicial]]
            modelo.fit(x,y,d,stand_error=stand_error,beta2=beta_variavel,day_mudar =day_beta_change,bound = bound)
        else:
            modelo.fit(x,y,d,stand_error=stand_error,beta2=beta_variavel,day_mudar =day_beta_change)
    
    else:
        modelo.fit(x,y,stand_error=stand_error)

    modelos.append(modelo)
    dias = (previsao_ate-novo_local[i].date.iloc[0]).days
    x_pred = range(1,dias+1)
    y_pred =modelo.predict(x_pred)
    d_pred = None
    if modelo_usado == 'SEIRHUD':
        d_pred = modelo.dpred
    novo_local[i]['totalCasesPred'] = y_pred[0:len(novo_local[i])]
    if modelo_usado == 'SEIRHUD':
        novo_local[i]['totalMortesPred'] = d_pred[0:len(novo_local[i])]
    novo_local[i]['residuo_quadratico'] = modelo.getResiduosQuadatico()
    novo_local[i]['res_quad_padronizado'] = modelo.getReQuadPadronizado()
    ultimo_dia = novo_local[i].date.iloc[-1]
    dias = (previsao_ate-novo_local[i].date.iloc[-1]).days
    for d in range(1,dias):
        di = d+len(x)-1
        if modelo_usado == 'SEIRHUD':
            novo_local[i]=novo_local[i].append({'totalCasesPred':y_pred[di],'totalMortesPred':d_pred[di],'date':ultimo_dia+dt.timedelta(d),'state':novo_local[i].state.iloc[0]}, ignore_index=True)
        else:
            novo_local[i]=novo_local[i].append({'totalCasesPred':y_pred[di],'date':ultimo_dia+dt.timedelta(d),'state':novo_local[i].state.iloc[0]}, ignore_index=True)

df = pd.DataFrame()
if modelo_usado=='SIR':
    for i in range(len(modelos)):
        novo_local[i]['sucetivel'] = pd.to_numeric(pd.Series(modelos[i].S[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['infectado'] = pd.to_numeric(pd.Series(modelos[i].I[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['recuperado'] = pd.to_numeric(pd.Series(modelos[i].R[0:len(novo_local[i].TOTAL)]),downcast='integer')
   
if modelo_usado=='SEIR' or modelo_usado=='SEIR_GA':
    for i in range(len(modelos)):
        novo_local[i]['sucetivel'] = pd.to_numeric(pd.Series(modelos[i].S[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['exposto'] = pd.to_numeric(pd.Series(modelos[i].E[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['infectado'] = pd.to_numeric(pd.Series(modelos[i].I[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['recuperado'] = pd.to_numeric(pd.Series(modelos[i].R[0:len(novo_local[i].TOTAL)]),downcast='integer')
if modelo_usado=='SEIRHUD':
    for i in range(len(modelos)):
        novo_local[i]['sucetivel'] = pd.to_numeric(pd.Series(modelos[i].S[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['exposto'] = pd.to_numeric(pd.Series(modelos[i].E[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['Infectado_assintomatico'] = pd.to_numeric(pd.Series(modelos[i].IA[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['Infectado_sintomatico'] = pd.to_numeric(pd.Series(modelos[i].IS[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['Hospitalizado'] = pd.to_numeric(pd.Series(modelos[i].H[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['UTI'] = pd.to_numeric(pd.Series(modelos[i].U[0:len(novo_local[i].TOTAL)]),downcast='integer')
        novo_local[i]['recuperado'] = pd.to_numeric(pd.Series(modelos[i].R[0:len(novo_local[i].TOTAL)]),downcast='integer')
for i in range(len(novo_local)):
    df = df.append(novo_local[i],ignore_index=True)

df.to_csv(arq_saida,index=False)

su = pd.DataFrame()
su['state'] = novo_nome
pop = []
rmse = []
coef_list = []
coef_name = None
data_mudan = []
for i in range(len(novo_nome)):
    if beta_variavel & ((modelo_usado == 'SIR') or (modelo_usado == 'SEIRHUD') or (modelo_usado == 'SEIR')):
        data_mudan.append(diaBound[i][2] + dt.timedelta(int(modelos[i].day_mudar)))
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
if beta_variavel & ((modelo_usado == 'SIR') or (modelo_usado=='SEIRHUD') or (modelo_usado == 'SEIR')):
    su['data_muda'] = data_mudan
su.to_csv(arq_sumario,index=False)

