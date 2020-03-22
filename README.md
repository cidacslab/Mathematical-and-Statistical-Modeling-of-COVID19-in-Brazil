
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

The amount of new individuals that get infected in time varies at a rate $\beta * S * I$. Analogously, the number of individuals that get recovered, changes in time with a recovery rate $\gamma$ (implying that the mean infectious period is $1/\gamma$ days). Therefore, an infected individual is expected to infect $R_{0} = \beta /  \gamma$ individuals in the period of $1 / \gamma$ days.

Summarising, we have two parameters in this simple SIR model: 

   * the transmission rate (per capita) $\beta$;
   * the recovery rate $\gamma$. 

#### Mathematical model


##### Fitted parameters for Brazil and Brazilian States

| Date of Fit | UF Code | UF Name        | Coefficients (with 95% confidence bounds):                                          | Goodness of fit:                                                                                         |                                                                                                                                                SIR model parameters | Reprodution number <br>  $R_0 = \frac{\beta}{\gamma}$ |
|-------------|---------|----------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|-------------------------------------------------------|
| 2020-03-21  | BR      | Brazil         | a =      0.6918  (0.07204, 1.312) <br> <br> b =      0.2905  (0.2452, 0.3357) <br>  |       SSE: 5517 <br> <br> R-square: 0.964 <br> <br> Adjusted R-square: 0.9621 <br> <br> RMSE: 17.04 <br> | Population (N) = <br> <br> $I_{0} = 1/N$ <br> <br> $S{0} = 1 - I_{0}$ <br> <br> $\gamma = 1/14$ <br> <br> $\beta = \frac{b + \gamma}{S_{0}} =$ 0.362 (0.317, 0.407) | 5.067 (4.433, 5.7)                                    |
| 2020-03-21  | 29      | Bahia          | a =      0.2789  (0.03912, 0.5186) <br> <br> b =      0.3482  (0.2771, 0.4193) <br> | SSE: 29.82 <br> <br> R-square: 0.9576 <br> <br>   Adjusted R-square: 0.9537 <br> <br>   RMSE: 1.646 <br> | Population (N) = <br> <br> $I_{0} = 1/N$ <br> <br> $S{0} = 1 - I_{0}$ <br> <br> $\gamma = 1/14$ <br> <br> $\beta = \frac{b + \gamma}{S_{0}}$                        |                                                       |
| 2020-03-21  | 33      | Rio de Janeiro | a =       4.709  (2.883, 6.536) <br> <br> b =       0.163  (0.1328, 0.1933) <br>    |     SSE: 204.1 <br> <br> R-square: 0.9477 <br> <br> Adjusted R-square: 0.9437 <br> <br> RMSE: 3.963 <br> | Population (N) = <br> <br> $I_{0} = 1/N$ <br> <br> $S{0} = 1 - I_{0}$ <br> <br> $\gamma = 1/14$ <br> <br> $\beta = \frac{b + \gamma}{S_{0}}$                        |                                                       |
