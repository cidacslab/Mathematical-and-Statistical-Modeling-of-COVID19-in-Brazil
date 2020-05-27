
![](images/cidacs.png)


# Evaluating the burden of COVID-19 in Bahia, Brazil: A modeling analysis of 14.8 million individuals                               

## Table of contents
* [General info](#general-info)
* [Compilation](#compilation)
* [Input data](#data)
* [References](#references)

## General info
In this folder the reader can find the code used to produce the results in the manuscript (preprint [1]) entitled "Evaluating the burden of COVID-19 in Bahia, Brazil: A modeling analysis of 14.8 million individuals".

## Compilation
We performed our analysis using Python version 3. 

## Input data

The models produced in this study were informed by data from multiple sources: the daily series of the cumulative confirmed COVID-19 cases and the daily mortality series for the state of Bahia, its capital Salvador and the remaining cities were obtained from [publicly available data](http://www.saude.ba.gov.br/temasdesaude/coronavirus/notas-tecnicas-e-boletins-epidemiologicos-covid-19/) provided by the Secretary of Health of the State of Bahia (SESAB). Throughout our analyses we consider separately the state capital (which concentrates the number of cases in the region) and the remaining 416 state municipalities. Data was available throughout the period of March 6 to May 4, 2020.

Our data is available in the folder named "Data". They are described as follow:

* data.csv contains variables of counts for the state of Bahia, they are:

   * dates: identifying the period;
   * infec: containing the daily cumulative number of confirmed cases; 
   * leitos: daily hospital beds occupancy; 
   * uti: daily ICU beds occupancy;
   * dth: daily number of deaths caused by COVID-19;
   * dthcm: daily cumulative number of deaths.

* dataInterior.csv contains the aggregated number of cases (variable name “cases”) and deaths (variable name “deaths”) for the inland cities of the state of Bahia;

* salvador.csv contains the number of cases (variable name “cases”) and deaths (variable name “deaths”) for the capital city of the state of Bahia.

In particular, the raw data, and depicted as dots, plotted in the figures 2, 3, 5 and 6, are provided by the datasets described above. 


## References 
[1] Evaluating the burden of COVID-19 on hospital resources in Bahia, Brazil: A modelling-based analysis of 14.8 million individuals. https://doi.org/10.1101/2020.05.25.20105213



 
 