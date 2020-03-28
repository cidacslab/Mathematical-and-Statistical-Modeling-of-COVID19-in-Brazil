
![](images/cidacs.png)


# Mathematical study of parameters 

In this folder we adapt codes from the main directory folder in order to deeply study the parameter range of the implemented models. Every week we will update the process of which we peformed the analises for the modeling team report. It is important to note that runing a static projection is different of the dinamic modeling update presented in the COvida dashboard, once the dataset is refreshed every 15 minute. 
    
## SIR parameters 

    * beta  = tranmission rate (per capita)
    
              ** range(0,1) 

    * gama  = recovery rate (inverse of mean infectious period)
            
            ** range_{1} = (1/5 - epsilon, 1/21 + epsilon)
            ** range_{2} = (1/14 - epsilon, 1/14+epsilon)
            ** range_{3} = (1/21-epsilon, 1/21+epsilon)

## SEIR parameters

