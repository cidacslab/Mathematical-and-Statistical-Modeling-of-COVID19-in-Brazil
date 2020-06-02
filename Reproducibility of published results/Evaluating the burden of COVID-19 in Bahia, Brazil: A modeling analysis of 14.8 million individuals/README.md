
![](images/cidacs.png)


# Evaluating the burden of COVID-19 in Bahia, Brazil: A modeling analysis of 14.8 million individuals                               

## Table of contents
* [General info](#general-info)
* [Input data](#data)
* [Compilation](#compilation)
* [References](#references)

## General info
In this folder the reader can find the code used to produce the results in the manuscript (preprint [1]) entitled "Evaluating the burden of COVID-19 in Bahia, Brazil: A modeling analysis of 14.8 million individuals".

## Input data

The models produced in this study were informed by data from multiple sources: the daily series of the cumulative confirmed COVID-19 cases and the daily mortality series for the state of Bahia, its capital Salvador and the remaining cities were obtained from [publicly available data](http://www.saude.ba.gov.br/temasdesaude/coronavirus/notas-tecnicas-e-boletins-epidemiologicos-covid-19/) provided by the Secretary of Health of the State of Bahia (SESAB). Throughout our analyses we consider separately the state capital (which concentrates the number of cases in the region) and the remaining 416 state municipalities. Data was available throughout the period of March 6 to May 4, 2020.

Our data is available in the folder named "Data". They are described as follow:

* pop_muni.csv contains the number of population for each Brazilian municipality. It was obtained by the Brazilian Institute of Geography and Statistics (IBGE). By aggregating the data, the population of the state of Bahia, estimated for 2020, is 14,930,424. The population of Salvador is 2,831,557 and the population of the inland cities is 12,098,867.

* data.csv contains variables of counts for the state of Bahia, they are:

   * dates: identifying the period;
   * infec: containing the daily cumulative number of confirmed cases; 
   * leitos: daily hospital beds occupancy; 
   * uti: daily ICU beds occupancy;
   * dth: daily number of deaths caused by COVID-19;
   * dthcm: daily cumulative number of deaths.

* dataInterior.csv contains the aggregated number of cases (variable name “cases”) and deaths (variable name “deaths”) for the inland cities of the state of Bahia;

* salvador.csv contains the number of cases (variable name “cases”) and deaths (variable name “deaths”) for the capital city of the state of Bahia.

In particular, the raw data, depicted as dots, plotted in the figures 2, 3, 5 and 6, in the manuscript [1] are provided by the datasets described above. 


## Compilation
We performed our analysis using Python version 3. The codes are presented in the folder named “scripts”. The "modelo.py" file contains the class with the SEIRUHD model. Complementary to that, the reader can find three Jupyter Notebook codes:

* SEIRUHD.ipynb compiles the mathematical model for the whole state;
* SEIRHUDInterior.ipynb compiles the mathematical model for the inland cities;
* SEIRHUDSalvador.ipynb compiles the mathematical model for the capital city.

The Particle Swarm Optimization (PSO) method estimates the parameters beta_0, beta_1, delta, h, gamma_H and gamma_U, and percentile confidence intervals were constructed using the weighted non-parametric bootstrap method, as described in [1]. 

The code to obtain the effective reproduction number is presented in the folder “Effective Reproduction number”.

Additionally, the sensitivity analysis of the parameters is performed by the file sensitivityAnalysis.py present in the folder “scripts”.

## References 
[1] Evaluating the burden of COVID-19 on hospital resources in Bahia, Brazil: A modelling-based analysis of 14.8 million individuals. https://doi.org/10.1101/2020.05.25.20105213



 
 
