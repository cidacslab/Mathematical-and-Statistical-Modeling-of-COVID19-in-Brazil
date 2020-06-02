#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import csv
import requests
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def unique_list(l):
    x = []
    for a in l:
        if a not in x:
            x.append(a)
    return x

 

#Opening Raw_data States

#Aqui vc só cria um txt pra receber os dados do site do wesley

req = requests.get('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')
url_content = req.content
csv_file = open('data/CoronaData1.txt', 'wb')

csv_file.write(url_content)
csv_file.close()
raw_data=[]
with open('data/CoronaData1.txt', encoding="utf8") as data:                                                                                          
    data_reader = csv.reader(data, delimiter='\t')
    for data in data_reader:
        raw_data.append(data)
    for i in range(len(raw_data)):
        raw_data[i]=raw_data[i][0].split (",")
for i in range(len(raw_data)-1):
    raw_data[i+1][5]=int(raw_data[i+1][5])
    raw_data[i+1][6]=int(raw_data[i+1][6])
    raw_data[i+1][7]=int(raw_data[i+1][7])
    
    if raw_data[i+1][2]=='TOTAL':
        raw_data[i+1][2]= 'Brazil'

#Separating States in the Raw Data
t=[]
dates=[]
data_state=[]
states=[]
for i in range(len(raw_data)-1):
    t.append(raw_data[i+1][0]) 
    data_state.append(raw_data[i+1][2])

states=unique_list(data_state)
dates=unique_list(t)



            
#Separating the data for each state
epidemic={}
for s in states:
    state_dic=[[],[],[],[],[]]
    for i in range(len(raw_data)-1):
        if raw_data[i+1][2] == s :
            state_dic[0].append(raw_data[i+1][0])
            state_dic[1].append(raw_data[i+1][6])
            state_dic[2].append(raw_data[i+1][7])
            state_dic[3].append(raw_data[i+1][5])
 


    epidemic[s]= state_dic

  
def unique_list(l):
    x = []
    for a in l:
        if a not in x:
            x.append(a)
    return x


req = requests.get('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv')
url_content = req.content
csv_file = open('data/CoronaData1.txt', 'wb')

csv_file.write(url_content)
csv_file.close()
raw_data=[]
with open('data/CoronaData1.txt', encoding="utf8") as data:                                                                                          
    data_reader = csv.reader(data, delimiter='\t')
    for data in data_reader:
        raw_data.append(data)
    for i in range(len(raw_data)):
        raw_data[i]=raw_data[i][0].split (",")
for i in range(len(raw_data)-1):
    raw_data[i+1][6]=int(raw_data[i+1][6])
    raw_data[i+1][7]=int(raw_data[i+1][7])
    raw_data[i+1][8]=int(raw_data[i+1][8])
    
    if raw_data[i+1][2]=='TOTAL':
        raw_data[i+1][2]= 'Brazil'

#Separating States in the Raw Data
t=[]
dates=[]
data_city=[]
cities=[]
for i in range(len(raw_data)-1):
    t.append(raw_data[i+1][0]) 
    data_city.append(raw_data[i+1][3])

cities=unique_list(data_city)
dates=unique_list(t)



            
#Separating the data for each state

for s in cities:
    city_dic=[[],[],[],[],[]]
    for i in range(len(raw_data)-1):
        if raw_data[i+1][3] == s :
            city_dic[0].append(raw_data[i+1][0])
            city_dic[1].append(raw_data[i+1][7])
            city_dic[2].append(raw_data[i+1][8])
            city_dic[3].append(raw_data[i+1][6])
 


    epidemic[s]= city_dic


# importing packages

#The state of the data that will be fitted
s='BA'

N=14873064


#Fixed parameters

kappa = 1/4
p = 0.15
gammaA = 1/5
gammaS = 1/5
gammaH = 1/10
gammaU = 1/10
muH = 0.2 
muU = 0.55 
omega_U = 0.29 
omega_H = 0.14
ia0 = 3.415991211337811e-06 
is0 = 3.415991211337811e-06
e0 = 3.415991211337811e-06
beta0 = 0.9999999131310117 
beta1 = 0.5750325381623237
xi = 0.53
delta = 0.7079084695853096
h = 0.24999999999999997
t1 = 14.62199106944937

#Find when the death starts

for i in range(len(epidemic[s][3])):
    if epidemic[s][3][i] != 0:
        ti=i
        break
#ti = date of the start of the death curve


#Defining the Steap Function 
k=50
def H(t,k):
    h = 1.0/(1.0+ np.exp(-2.0*k*t))
    return h


#Defining the Beta(t)
def beta(t,t1,b,b1,k):
    beta = b*H(t1-t,k) + b1*H(t-t1,k) 
    return beta



#Defining the Model
def SEIRHU(f,t,parametros):
    
    #parameters
    b = parametros[0]
    b1 = parametros[1]
    gammaH = parametros[2]
    gammaU = parametros[3]
    delta = parametros[4]
    h = parametros[5]
    t1 = parametros[6]
    k = parametros[7]
    
    
    #variables
    S = f[0]
    E = f[1]
    IA = f[2]
    IS = f[3]
    H = f[4]
    U = f[5]
    
    
    #equations
    dS_dt = - beta(t,t1,b,b1,k)*S*(IS + delta*IA) 
    dE_dt = beta(t,t1,b,b1,k)*S*(IS + delta*IA) - kappa * E
    dIA_dt = (1-p)*kappa*E - gammaA*IA
    dIS_dt = p*kappa*E - gammaS*IS
    dH_dt = h*xi*gammaS*IS + (1- muU+ omega_U*muU)*gammaU*U - gammaH*H
    dU_dt = h*(1-xi)*gammaS*IS + omega_H*gammaH*H -gammaU*U
    dR_dt = gammaA*IA + (1-(muH))*(1-omega_H)*gammaH*H + (1-h)*gammaS*IS
    dD_dt = (1-omega_H)*muH*gammaH*H + (1-omega_U)*muU*gammaU*U
    
    #epidemic curve
    dNw_dt = p*kappa*E
    
    #Returning the derivatives
    return [dS_dt,dE_dt,dIA_dt,dIS_dt,dH_dt,dU_dt,dR_dt,dD_dt,dNw_dt]


problem = {
    'num_vars': 8,
    'names': ['beta0', 'beta1', 'gammaH','gammaU', 'delta', 'h', 't1','k'],
    'bounds':np.column_stack((np.array([0., 0., 1/12, 1/12, 0., 0.05, 0., 0]),\
                              np.array([2., 2., 1/4, 1/3,   0.7, 0.25,30.,100])))
}

# Generate samples
nsamples = 12000

sampleset = saltelli.sample(problem, nsamples, calc_second_order=False)

#define your data
infec = np.array(epidemic[s][2])

ts0 = np.array(list(range(len(infec))))+1

print(sampleset.shape, len(ts0))

# Arrays to store the index estimates
S1_estimates_Ia = np.zeros([problem['num_vars'],len(ts0)])
ST_estimates_Ia = np.zeros([problem['num_vars'],len(ts0)])

S1_estimates_Is = np.zeros([problem['num_vars'],len(ts0)])
ST_estimates_Is = np.zeros([problem['num_vars'],len(ts0)])

S1_estimates_U = np.zeros([problem['num_vars'],len(ts0)])
ST_estimates_U = np.zeros([problem['num_vars'],len(ts0)])

S1_estimates_H = np.zeros([problem['num_vars'],len(ts0)])
ST_estimates_H = np.zeros([problem['num_vars'],len(ts0)])

S1_estimates_D = np.zeros([problem['num_vars'],len(ts0)])
ST_estimates_D = np.zeros([problem['num_vars'],len(ts0)])

S1_estimates_Nw = np.zeros([problem['num_vars'],len(ts0)])
ST_estimates_Nw = np.zeros([problem['num_vars'],len(ts0)])


# initializing matrix to store output
Y = [odeint(SEIRHU,[1-ia0 - is0 -e0,e0,ia0,is0,0,0,0,0,0],ts0,\
                     args=([sampleset[j][0],sampleset[j][1], sampleset[j][2],\
                           sampleset[j][3],sampleset[j][4],sampleset[j][5], sampleset[j][6], sampleset[j][7]],),mxstep=1000000) for j in range(len(sampleset))]
sol_ = np.asarray(Y)

for i in range(len(ts0)):
    results = sobol.analyze(problem, sol_[:,i,2], calc_second_order=False, \
                              num_resamples=100, conf_level=0.95, print_to_console=False, \
                              parallel=True, n_processors=2, seed=None)
    ST_estimates_Ia[:,i]=results['ST']
    S1_estimates_Ia[:,i]=results['S1']

    results = sobol.analyze(problem, sol_[:,i,3], calc_second_order=False, \
                              num_resamples=100, conf_level=0.95, print_to_console=False, \
                              parallel=True, n_processors=2, seed=None)
    ST_estimates_Is[:,i]=results['ST']
    S1_estimates_Is[:,i]=results['S1']

    results = sobol.analyze(problem, sol_[:,i,4], calc_second_order=False, \
                              num_resamples=100, conf_level=0.95, print_to_console=False, \
                              parallel=True, n_processors=2, seed=None)
    ST_estimates_U[:,i]=results['ST']
    S1_estimates_U[:,i]=results['S1']


    results = sobol.analyze(problem, sol_[:,i,5], calc_second_order=False, \
                              num_resamples=100, conf_level=0.95, print_to_console=False, \
                              parallel=True, n_processors=2, seed=None)
    ST_estimates_H[:,i]=results['ST']
    S1_estimates_H[:,i]=results['S1']


    results = sobol.analyze(problem, sol_[:,i,7], calc_second_order=False, \
                              num_resamples=100, conf_level=0.95, print_to_console=False, \
                              parallel=True, n_processors=2, seed=None)
    ST_estimates_D[:,i]=results['ST']
    S1_estimates_D[:,i]=results['S1']

    results = sobol.analyze(problem, sol_[:,i,8], calc_second_order=False, \
                              num_resamples=100, conf_level=0.95, print_to_console=False, \
                              parallel=True, n_processors=2, seed=None)
    ST_estimates_Nw[:,i]=results['ST']
    S1_estimates_Nw[:,i]=results['S1']


object = ["Ia","Is", "U", "H", "D", "Nw"]

for idx, var in enumerate(object):
    print(object[idx])
    exec("np.savetxt('ST_estimates_" + object[idx] + ".txt', ST_estimates_" + object[idx] + ")")
    exec("np.savetxt('S1_estimates_" + object[idx] + ".txt', S1_estimates_" + object[idx] + ")")
 


object = ["Ia","Is", "U", "H", "D", "Nw"]

object_plot = [r"$I_a$","$I_s$", "$U$", "$H$", "$D$", "$Nw$"]

names = [r"$\beta_0$", r"$\beta_1$", r"$\gamma_H$", r"$\gamma_U$", r"$\delta$", r"$h$", r"$t_1$", r"$k$"]

for idx, var in enumerate(object):

    S1_estimates = np.loadtxt('S1_estimates_' + object[idx] + '.txt')
    ST_estimates = np.loadtxt('ST_estimates_' + object[idx] + '.txt')


    # Create figure and plot space
    plt.rc('font', family='serif')

    # S1 Plot
    fig, ax = plt.subplots(figsize=(16,8))

    # Add x-axis and y-axis
    handles = []
    for j in range(problem['num_vars']):
        handles += ax.plot(ts0, S1_estimates[j], linewidth=5)

    ax.set_title(r'Evolution of $S_{i}$ index estimates - '+ object_plot[idx] , fontsize=28)
    ax.set_ylabel(r'$S_{i}$', fontsize=28)
    ax.set_xlabel('Time (days)', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    plt.legend(handles, names, loc='best', fontsize=28)
    plt.savefig('S1_indexevolution_' + object[idx] + '.pdf')

    # ST Plot
    fig, ax = plt.subplots(figsize=(16,8))
    plt.rc('font', family='serif')

    # Add x-axis and y-axis
    handles = []
    for j in range(problem['num_vars']):
        handles += ax.plot(ts0, ST_estimates[j], linewidth=5)

    ax.set_title(r'Evolution of $S_{T_i}$ index estimates - '+ object_plot[idx] , fontsize=28)
    ax.set_ylabel(r'$S_{T_i}$', fontsize=28)
    ax.set_xlabel('Time (days)', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    plt.legend(handles, names, loc='best', fontsize=28)
    plt.savefig('ST_indexevolution_' + object[idx] + '.pdf')
