
# Mathematical and Statistical Modeling of COVID19 in Brazil                               

## Table of contents
* [Objective](#Objective)
* [Overview](#Overview)
* [Mathematical modeling: compartimental models](#Mathematical)
  * [The Susceptible-Infected-Recovered model](#SIR)
* [References](#references)

## Objective

To make a library of models that aim to understand the spread of COVID19 in adequate scenarios of the Brazilian population, and also to compare our results in the literature for possible validations in real time.

## Overview

  The propagation of an infectious disease in a population depends on many factors. These factors range from the conditions the pathogen behaves in  the individual to levels of the dissemination in mass that may depend on demographic-social-economic factors. There are many approaches to understand how the disease spread among the individuals and how  to control it. Here we want to present models that are used in the literature to try to contribute to the understanding of the spread of COVID19. In particular, we want to present a comparison of a variety of results that our team is producing with those found on the literature. 

## Mathematical modeling: compartimental models

  In this section, we gradatively show the application of compartimental models for disease transmission. We start illustrating the use of the so well known Susceptible, infected and Recovered model, increasing to more complex systems involving other divisions of the population as quarantined individuals, Isolated individuals, and so on. We apply all our analysis to data of coronavirus in Brazil and its subregions.  

### The Susceptible-Infected-Recovered model

#### Conceptual model

Depending on the disease and the research scope, after the contact of the susceptible population with infected individuals, the most basic model of disease transmission assumes that the population can be divided into three compartments: 

  * the susceptible, S, individuals who can catch the disease;
  * the infectives, I, who have the disease and can transmit it; 
  * and the removed, R, which comprises the ones that no longer is infectious due to isolation or immunity.

The amount of new individuals that get infected in time varies at a rate $\beta  S  I$. Analogously, the number of individuals that get recovered, changes in time with a recovery rate $\gamma$ (implying that the mean infectious period is $1/\gamma$ days). Therefore, an infected individual is expected to infect $R_{0} = \beta /  \gamma$ individuals in the period of $1 / \gamma$ days.

Summarising, we have two parameters in this simple SIR model: 

   * the transmission rate (per capita) $\beta$;
   * the recovery rate $\gamma$. 

#### Mathematical model

The system of equations for the SIR model is given by:

\begin{array}{lcl} 
\frac{dS}{dt} & = & -\beta SI  &  (1.1) \\ 
\frac{dI}{dt} & = & \beta SI -\gamma I  &  (1.2) \\
\frac{dR}{dt} & = & \gamma I  &  (1.3) 
\end{array}

The system cannot be solved in terms of known analitic functions. 

It is useful to estimate parameters of the model with simplified model. We assume inicially that the amount of infected people $I$ is much smaller than amount of susceptible $S$ people, i.e. the simplification is valid until $\epsilon \sim \frac{I}{S} << 1$. Hence, we can approximate the system of equations (1.1)-(1.3) in the following way 

\begin{array}{lcl} 
S(t) & =  & S_{0} + \epsilon S_{1}(t) + \mathcal{O}(\epsilon^2) & (1.1.A), \\ 
I(t) & = & I_{0} -\epsilon I_{1}(t) + \mathcal{O}(\epsilon^2)  &  (1.2.A), \\ 
\end{array}
where $\epsilon$ is a small parameter, which represents that $\frac{I}{S} << 1$, $S_0 \approx 1$ and $I_0 \approx 0$ for normalized model (we normalized $S$ for unity). Substituting (1.1.A) and (1.2.A) into (1.1) and (1.2) one can obtain the system

\begin{array}{lcl} 
\frac{dS_1}{dt} & = & - \beta I_{1} + \mathcal{O}(\epsilon) & (1.1.A), \\ 
\frac{dI_1}{dt} & = & I_{1} \left( \beta - \gamma \right) + \mathcal{O}(\epsilon) &  (1.2.A), \\ 
\end{array}
which has analytic solution in the form 

\begin{array}{lcl} 
S_1 & = & C_2 + C_1 \frac{e^{(\beta - \gamma) t}}{a-b} &  (1.1.B) , \\ 
I_1 & = & C_1 e^{(\beta - \gamma) t} &  (1.2.B), \\ 
\end{array}
where $C_1$ and $C_2$ are constants of integration. Chosing $C_1$ and $C_2$ to satisfy initial conditions for $S_1(0)$ and $I_1(0)$ we obtain

\begin{array}{lcl} 
S_1 & = & - \left( \gamma S_1(0) + \beta \left( I_1(0) \left( e^{(\beta - \alpha) t} - S_1(0) \right) \right) \right)  / (\beta-\gamma) &  (1.1.C), \\ 
I_1 & = & I_1(0) e^{(\beta - \gamma) t} &  (1.2.C). \\ 
\end{array}

The equations (1.1.C)-(1.2.C) provide a simple connection with parameters $\alpha$ and $\beta$ for fitting the begining of outbreak. When the amount of infected people tends to amount of susceptible people, $I \sim S$ the full model should be used for correct values of $\beta$.

In this directory we provide estimation of the parâmeters taking into consideration the whole system of differential equations. We provide the code in the folder "Code for model estimations", with the respective instruction on how to run the codes. 


##### Interpretation of the model
