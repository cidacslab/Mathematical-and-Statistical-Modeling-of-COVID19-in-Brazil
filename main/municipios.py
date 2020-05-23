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
modelo_usado = 'SEIR' #EXP, SIR, SEIR, SEIRHUD
stand_error = True # se true usa erro ponderado, se false usa erro simples
beta_variavel = True # funciona no SIR, caso True ocorre mudança do beta no dia definido no parametro abaixo
day_beta_change = None  # funciona no SIR,dia da mudança do valor do beta se None a busca vai ser automatica
municipios = [3300605]
numeroProcessadores = None # numero de prossesadores para executar em paralelo
N_inicial = 1000
min_cases = 5
min_dias = 10
arq_saida = '../data/municipios.csv'
arq_sumario = '../data/municipios_sumario.csv'
previsao_ate = dt.date.today() +  dt.timedelta(10)


def ler_banco_municipios():
    try:
        url = 'https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv'
        banco = pd.read_csv(url)
    except:
        return None,None
    
    
    banco =banco[banco['ibgeID'].notnull()]
    banco = banco[banco.city!='TOTAL']
    nome_local =list(banco['ibgeID'].unique())
    data = []
    for i in banco.index:
        print(str(i)+'\n')
        data.append( dt.datetime.strptime(banco.date[i], '%Y-%m-%d').date())
    banco.date = data
    local = []
    for est in nome_local:
        
    
        aux = banco[banco['ibgeID']==est].sort_values('date')
        data_ini = aux.date.iloc[0]
        data_fim = aux.date.iloc[-1]
        dias = (data_fim-data_ini).days + 1
        d = [(data_ini + dt.timedelta(di)) for di in range(dias)]
        
        estado = [aux.state.iloc[0] for di in range(dias)]
        city = [aux.city.iloc[0] for di in range(dias)]
        ibgeID = [est for di in range(dias)]
        df = pd.DataFrame({'date':d,'UF':estado,'city':city,'ibgeID':ibgeID})
        
        casos = []
        mortes = []
        morte = 0
        caso = 0
        i_aux = 0
        for i in range(dias):
            if (d[i]-aux.date.iloc[i_aux]).days==0:
                caso = aux['totalCases'].iloc[i_aux]
                morte = aux['deaths'].iloc[i_aux]
                casos.append(caso)
                mortes.append(morte)
                i_aux=i_aux+1
            else:
                casos.append(caso)
                mortes.append(morte)
        new = [casos[0]]        
        for i in range(1,dias):
            new.append(casos[i]-casos[i-1])
        df['mortes'] = mortes
        df['newCases'] = new
        df['TOTAL'] = casos
        local.append(df)
    return nome_local, local      

#carregar dados'
nome, local = ler_banco_municipios()
df_pop = pd.read_csv('../data/populacao_municipio.csv')
pop_local =df_pop.Armenor.unique()
pop = {}
for o in pop_local:
    aux = df_pop[df_pop.Armenor==o]
    pop[o] = sum(aux.Total)
    
novo_nome = []
novo_local = []
for i in range(len(nome)):
    if (local[i].TOTAL.iloc[-1]>=min_cases) & (len(local[i])>=min_dias) & (municipios==None):
        novo_local.append(local[i])
        novo_nome.append(nome[i])
    elif municipios!= None:
        if nome[i] in municipios:
            novo_nome.append(nome[i])
            novo_local.append(local[i])
previsao_ate = previsao_ate + dt.timedelta(1)
#for i in range(len(nome)):
#    if (local[i].TOTAL.iloc[-1]>=min_cases) & (len(local[i])>=10):
#        novo_local.append(local[i])
#        novo_nome.append(nome[i])
#previsao_ate = previsao_ate + dt.timedelta(1)
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
    if modelo_usado =='SIR':
        modelo = md.SIR(N,numeroProcessadores)
    elif modelo_usado =='EXP':
        modelo = md.EXP(N,numeroProcessadores)
    elif modelo_usado =='SEIR':
        modelo = md.SEIR(N,numeroProcessadores)
    elif modelo_usado =='SEIRHUD':
        modelo = md.SEIRHUD(N,numeroProcessadores)
    else:
        print('Modelo desconhecido '+modelo_usado)
        sys.exit(1)
    
    y = novo_local[i].TOTAL
    x = range(1,len(y)+1)
    d = novo_local[i].mortes

    if modelo_usado == 'SIR':
        modelo.fit(x,y,stand_error=stand_error,beta2=beta_variavel,day_mudar =day_beta_change)                                                                      
    elif modelo_usado == 'SEIRHUD':
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
    for a in range(1,dias):
        di = a+len(x)-1
        if modelo_usado == 'SEIRHUD':
            novo_local[i]=novo_local[i].append({'totalCasesPred':y_pred[di],'totalMortesPred':d_pred[di],'date':ultimo_dia+dt.timedelta(a),'UF':novo_local[i].UF.iloc[0],'city':novo_local[i].city.iloc[0],'ibgeID':novo_local[i].ibgeID.iloc[0]}, ignore_index=True)
        else:
            novo_local[i]=novo_local[i].append({'totalCasesPred':y_pred[di],'date':ultimo_dia+dt.timedelta(a),'UF':novo_local[i].UF.iloc[0],'city':novo_local[i].city.iloc[0],'ibgeID':novo_local[i].ibgeID.iloc[0]}, ignore_index=True)
    
    novo_local[i].ibgeID = pd.to_numeric(novo_local[i].ibgeID,downcast='integer')
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
su['ibgeID'] = novo_nome
popu = []
rmse = []
coef_list = []
coef_name = None
data_mudan = []
for i in range(len(novo_nome)):
    coef_name, coef  = modelos[i].getCoef()
    coef_list.append(coef)
    popu.append(modelos[i].N)
    rmse.append(modelos[i].rmse)
su['populacao']= popu
su['rmse'] = rmse
for c in range(len(coef_name)):
    l = []
    for i in range(len(coef_list)):
        l.append(coef_list[i][c])
    su[coef_name[c]]=l
if beta_variavel:
    for i in range(len(novo_nome)):
        data = novo_local[i].date[0]+dt.timedelta(int(su.dia_mudanca[i]))
        data_mudan.append(data)
    su['data_muda'] = data_mudan
su.to_csv(arq_sumario,index=False)

