#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:53:37 2020

@author: lhunlindeion
"""

import pandas as pd
import numpy as np

file = '~/ownCloud/sms/baseacumulada/dataSerieSMS_0506.csv'

data_raw = pd.read_csv(file, delimiter=',')
data_raw['control.date'] =  pd.to_datetime(data_raw['control.date'], yearfirst=True)

useful_columns = ['inicio_sintomas', 'control.date', 'evolucao']
data_raw = data_raw.sort_values(by=['inicio_sintomas', 'dt_coleta', 'control.date'])
