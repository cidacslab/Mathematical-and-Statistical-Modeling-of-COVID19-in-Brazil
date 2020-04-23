import numpy as np
import pandas as pd
import functools
import matplotlib.pyplot as plt
import scipy.integrate as spi
from platypus import NSGAII, Problem, Real  # Genetic Algorithm Library


def SIR_diff_eqs(INP, t, beta, gamma):
    '''The main set of equations'''
    Y = np.zeros((3))
    V = INP
    Y[0] = - beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    Y[2] = gamma * V[1]
    return Y


def SEQIJR_diff_eqs(INP, t, beta, epsilon_E, epsilon_Q, epsilon_J, Pi,
                    mu, v, gamma1, gamma2, kappa1, kappa2, d1, d2, sigma1, sigma2, DS, DE, DI, DJ, DQ):
    '''The main set of equations'''
    Y = np.zeros((8))
    V = INP
    L = beta * (V[3] + epsilon_E * V[1] + epsilon_Q * V[2] + epsilon_J * V[4])
    Y[0] = Pi - L * V[0] / N - DS * V[0]  # (1)
    Y[1] = L * V[0] / N - DE * V[1]  # (2)
    Y[2] = gamma1 * V[1] - DQ * V[2]  # (3)
    Y[3] = kappa1 * V[1] - DI * V[3]  # (4)
    Y[4] = gamma2 * V[3] + kappa2 * V[2] - DJ * V[4]  # (5)
    Y[5] = v * V[0] + sigma1 * V[3] + sigma2 * V[4] - mu * V[5]  # (6)
    Y[6] = Pi - d1 * V[3] - d2 * V[4] - mu * V[6]  # (7)
    Y[7] = d1 * V[3] + d2 * V[4]
    return Y


# Optimisation Function which minimises the MSE between the SIR model and the data
def fitness_function(x, model, Model_Input, t_range):

    global real_data
    mean_squared_error = 0

    if model == 'SIR':

        beta = x[0]
        gamma = x[1]

        result = spi.odeint(SIR_diff_eqs, Model_Input, t_range, args=(beta, gamma))

        mean_squared_error = ((np.array(real_data) - result[:, 1]) ** 2).mean()

    elif model == 'SEQIJR':

        beta = x[0]  # Infectiousness and contact rate between a susceptible and an infectious individual
        epsilon_E = x[1]  # Modification parameter associated with infection from an exposed asymptomatic individual
        epsilon_Q = x[2]  # Modification parameter associated with infection from a quarantined individual
        epsilon_J = x[3]  # Modification parameter associated with infection from an isolated individual
        Pi = x[4]  # Rate of inflow of susceptible individuals into a region or community through birth or migration.
        mu = x[5]  # The natural death rate for disease-free individuals
        v = x[6]  # Rate of immunization of susceptible individuals
        gamma1 = x[7]  # Rate of quarantine of exposed asymptomatic individuals
        gamma2 = x[8]  # Rate of isolation of infectious symptomatic individuals
        kappa1 = x[9]  # Rate of development of symptoms in asymptomatic individuals
        kappa2 = x[10]  # Rate of development of symptoms in quarantined individuals
        d1 = x[11]  # Rate of disease-induced death for symptomatic individuals
        d2 = x[12]  # Rate of disease-induced death for isolated individuals
        sigma1 = x[13]  # Rate of recovery of symptomatic individuals
        sigma2 = x[14]  # Rate of recovery of isolated individuals

        DS = mu + v
        DE = gamma1 + kappa1 + mu
        DI = gamma2 + d1 + sigma1 + mu
        DJ = sigma2 + d2 + mu
        DQ = mu + kappa2

        result = spi.odeint(SEQIJR_diff_eqs, Model_Input, t_range, args=(beta, epsilon_E, epsilon_Q, epsilon_J,
                                                                         Pi, mu, v, gamma1, gamma2, kappa1, kappa2, d1,
                                                                         d2, sigma1, sigma2, DS, DE, DI, DJ, DQ))

        mean_squared_error = ((np.array(real_data) - result[:, 3]/N) ** 2).mean()

    return [mean_squared_error]


df = pd.read_csv('datastate.csv')
popBA = 14873064
N = popBA
df_BA = np.array([1, 2, 2, 2, 2, 3, 3, 7, 7, 9, 10, 16, 28]) / popBA
data_label = 'Bahia'

real_data = df_BA

model = 'SIR' # SIR ou SEQIJR

if model == 'SIR':

    I0 = 1 / N
    S0 = 1 - I0
    R0 = 0

    # The model needs to output in the same shape as the data to be fitted
    TS = 1
    ND = len(real_data) - 1

    t_start = 0.0
    t_end = ND
    t_inc = TS
    t_range = np.arange(t_start, t_end + t_inc, t_inc)

    # INPUT = [S(0), I(0), R(0)]
    Model_Input = (S0, I0, 0.0)

    # GA Parameters
    number_of_generations = 10000
    ga_population_size = 100
    number_of_objective_targets = 1  # The MSE
    number_of_constraints = 0
    number_of_input_variables = 2  # beta and gamma
    problem = Problem(number_of_input_variables, number_of_objective_targets, number_of_constraints)
    problem.function = functools.partial(fitness_function, model=model, Model_Input=Model_Input, t_range=t_range)

    algorithm = NSGAII(problem, population_size=ga_population_size)

    problem.types[0] = Real(0, 1)  # beta initial Range
    problem.types[1] = Real(1 / 5, 1 / 14)  # gamma initial Range

    # Running the GA
    algorithm.run(number_of_generations)

    # Getting the feasible solutions only just to be safe
    # The GA may generate bad solutions if number_of_generations is not enough or there are too many contraints
    feasible_solutions = [s for s in algorithm.result if s.feasible]

    beta = feasible_solutions[0].variables[0]
    print('Beta Optimsed = {}'.format(beta))
    gamma = feasible_solutions[0].variables[1]
    print('Gamma Optimsed = {}'.format(gamma))

    # Running the model and plotting with the optimised values
    ND = 2 * (len(real_data) - 1)
    t_start = 0.0
    t_end = ND
    t_inc = TS
    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    RES = spi.odeint(SIR_diff_eqs, Model_Input, t_range, args=(beta, gamma))

    fig = plt.figure(figsize=(20, 18 / 3), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(real_data*100, c='red', label='Porcentagem Infectada - {}'.format(data_label), linewidth=5)
    plt.plot(RES[:, 1]*100, label='Porcentagem Infectada - Modelo SIR')
    plt.xlabel('Dias', fontsize=20)
    plt.ylabel('População Infectada (%)', fontsize=20)
    plt.xticks(range(1, t_end + 1, round(t_end * 0.1)), fontsize=10)
    plt.grid(alpha=0.5, which='both')
    plt.legend(fontsize=15)
    plt.show()

elif model == 'SEQIJR':


    # The model needs to output in the same shape as the data to be fitted
    TS = 1
    ND = len(real_data) - 1

    t_start = 0.0
    t_end = ND
    t_inc = TS
    t_range = np.arange(t_start, t_end + t_inc, t_inc)

    I0 = 1  # Infectious
    # An infectious person is symptomatic

    E0 = 0  # Exposed
    # An exposed person is someone who has come into contact
    # with an infectious person but is asymptomatic


    S0 = N-I0-E0  # Susceptible
    # A susceptible person is an uninfected person who can
    # be infected through contact with an infectious or exposed
    # person

    Q0 = 0  # Quarantined
    # A quarantined person is an exposed person who is removed
    # from contact with the general population

    J0 = 0  # Isolated
    # an isolated person is an infectious person who
    # is removed from contact with the general population,
    # usually by being admitted to a hospital.

    R0 = 0  # Recovered
    # A recovered person is someone who has recovered
    # from the disease

    N0 = 0  # Population size ???

    D0 = 0  # Death Rate ???

    Model_Input = (S0,E0,Q0,I0,J0,R0,N0,D0)

    input_variables = ['beta', 'epsilon_E', 'epsilon_Q', 'epsilon_J', 'Pi', 'mu', 'v', 'gamma1', 'gamma2', 'kappa1',
                       'kappa2', 'd1', 'd2', 'sigma1', 'sigma2']

    number_of_generations = 10000
    ga_population_size = 100
    number_of_objective_targets = 1
    number_of_constraints = 0
    number_of_input_variables = len(input_variables)

    problem = Problem(number_of_input_variables, number_of_objective_targets, number_of_constraints)

    problem.types[0] = Real(0,0.4)  # beta      - Infectiousness and contact rate between a susceptible and an infectious individual
    problem.types[1] = Real(0,0.5)  # epsilon_E - Modification parameter associated with infection from an exposed asymptomatic individual
    problem.types[2] = Real(0,0.5)  # epsilon_Q - Modification parameter associated with infection from a quarantined individual
    problem.types[3] = Real(0,1)  # epsilon_J - Modification parameter associated with infection from an isolated individual
    problem.types[4] = Real(0,500)  # Pi        - Rate of inflow of susceptible individuals into a region or community through birth or migration.
    problem.types[5] = Real(0, 0.00005)  # mu        - The natural death rate for disease-free individuals
    problem.types[6] = Real(0, 0.1)  # v         - Rate of immunization of susceptible individuals
    problem.types[7] = Real(0, 0.3)  # gamma1    - Rate of quarantine of exposed asymptomatic individuals
    problem.types[8] = Real(0, 0.7)  # gamma2    - Rate of isolation of infectious symptomatic individuals
    problem.types[9] = Real(0, 0.3)  # kappa1    - Rate of development of symptoms in asymptomatic individuals
    problem.types[10] = Real(0, 0.3)  # kappa2    - Rate of development of symptoms in quarantined individuals
    problem.types[11] = Real(0, 0.1)  # d1        - Rate of disease-induced death for symptomatic individuals
    problem.types[12] = Real(0, 0.1)  # d2        - Rate of disease-induced death for isolated individuals
    problem.types[13] = Real(0, 0.1)  # sigma1    - Rate of recovery of symptomatic individuals
    problem.types[14] = Real(0, 0.1)  # sigma2    - Rate of recovery of isolated individuals

    problem.function = functools.partial(fitness_function, model=model, Model_Input=Model_Input, t_range=t_range)
    algorithm = NSGAII(problem, population_size=ga_population_size)
    algorithm.run(number_of_generations)

    feasible_solutions = [s for s in algorithm.result if s.feasible]

    beta = feasible_solutions[0].variables[0]  # Infectiousness and contact rate between a susceptible and an infectious individual
    epsilon_E = feasible_solutions[0].variables[1]  # Modification parameter associated with infection from an exposed asymptomatic individual
    epsilon_Q = feasible_solutions[0].variables[2]  # Modification parameter associated with infection from a quarantined individual
    epsilon_J = feasible_solutions[0].variables[3]  # Modification parameter associated with infection from an isolated individual
    Pi = feasible_solutions[0].variables[4]  # Rate of inflow of susceptible individuals into a region or community through birth or migration.
    mu = feasible_solutions[0].variables[5]  # The natural death rate for disease-free individuals
    v = feasible_solutions[0].variables[6]  # Rate of immunization of susceptible individuals
    gamma1 = feasible_solutions[0].variables[7]  # Rate of quarantine of exposed asymptomatic individuals
    gamma2 = feasible_solutions[0].variables[8]  # Rate of isolation of infectious symptomatic individuals
    kappa1 = feasible_solutions[0].variables[9]  # Rate of development of symptoms in asymptomatic individuals
    kappa2 = feasible_solutions[0].variables[10]  # Rate of development of symptoms in quarantined individuals
    d1 = feasible_solutions[0].variables[11]  # Rate of disease-induced death for symptomatic individuals
    d2 = feasible_solutions[0].variables[12]  # Rate of disease-induced death for isolated individuals
    sigma1 = feasible_solutions[0].variables[13]  # Rate of recovery of symptomatic individuals
    sigma2 = feasible_solutions[0].variables[14]  # Rate of recovery of isolated individuals

    DS = mu + v
    DE = gamma1 + kappa1 + mu
    DI = gamma2 + d1 + sigma1 + mu
    DJ = sigma2 + d2 + mu
    DQ = mu + kappa2

    alphaE = DI / kappa1
    alphaS = DE * alphaE
    alphaQ = (gamma1 / DQ) * alphaE
    alphaJ = (gamma2 + kappa2 * alphaQ) / DJ
    alphaN = d1 + d2 * alphaJ
    alphaR = (1 / mu) * ((v * alphaS / DS) - sigma1 - sigma2 * alphaJ)
    alphaL = beta * (1 + epsilon_E * alphaE + epsilon_Q * alphaQ + epsilon_J * alphaJ)

    I2 = (Pi * (mu * alphaL - DS * alphaS)) / (alphaS * (mu * alphaL - DS * alphaN))

    S2 = (1 / DS) * (Pi - alphaS * I2)

    E2 = alphaE * I2

    J2 = alphaJ * I2

    N2 = (1 / mu) * (Pi - alphaN * I2)

    Q2 = alphaQ * I2

    R2 = ((v * Pi) / (mu * DS)) - alphaR * I2

    Rdf = (mu * alphaL) / (DS * alphaS)

    R0 = alphaL / alphaS

    print('Rdf = {:.4f}, R0 = {:.4f}'.format(Rdf, R0))

    t_start = 0.0
    t_end = 2 * (len(real_data) - 1)
    t_inc = TS
    t_range = np.arange(t_start, t_end + t_inc, t_inc)

    result_fit = spi.odeint(SEQIJR_diff_eqs, Model_Input, t_range, args=(
    beta, epsilon_E, epsilon_Q, epsilon_J, Pi, mu, v, gamma1, gamma2, kappa1, kappa2, d1, d2, sigma1, sigma2, DS, DE,
    DI, DJ, DQ))

    for i in range(len(input_variables)):
        print('{} = {:.6f}'.format(input_variables[i], feasible_solutions[0].variables[i]))

    fig = plt.figure(figsize=(20, 18 / 3), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(100*(real_data), c='red', label='Porcentagem Infectada - {}'.format(data_label), linewidth=5)
    plt.plot(100 * (result_fit[:, 3] / N), label='Porcentagem Infectada - Modelo SEQIJR')
    plt.xlabel('Dias', fontsize=20)
    plt.ylabel('População Infectada (%)', fontsize=20)
    plt.xticks(range(1, t_end + 1, round(t_end * 0.1)), fontsize=10)
    plt.grid(alpha=0.5, which='both')
    plt.legend(fontsize=15)
    plt.show()
