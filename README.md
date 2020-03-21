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


##### Fitted values for Brazil and Brazilian States

Tabel 1:  General model Exp1:   f(x) = a*exp(b*x)                                                               
| Date of Fit | UF Code | UF Name | Coefficients (with 95% confidence bounds):                                | Goodness of fit:                                                                  |
|-------------|---------|---------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| 2020-03-21  | BR      | Brazil  | a =      0.6918  (0.07204, 1.312) <br>  b =      0.2905  (0.2452, 0.3357) | SSE: 5517 <br>  R-square: 0.964 <br>  Adjusted R-square: 0.9621 <br>  RMSE: 17.04 |
