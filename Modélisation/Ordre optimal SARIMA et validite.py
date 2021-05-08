#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:03:57 2021

@author: victorhuynh
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

# !pip install pmdarima 
from pmdarima import auto_arima



def ordre_SARIMA(histoMod, dateDebMod, dateFinMod):
    
    histoMod = histoMod.reset_index(drop = True)
    coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
    train = histoMod[histoMod.index <= coupe]
    
    stepwise_fit = auto_arima(train['PAX'], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)       

    return stepwise_fit.order, stepwise_fit.seasonal_order



def previsions_SARIMA (histoMod, dateDebMod, dateFinMod, hPrev, ic) :
    
    histoMod = histoMod.reset_index(drop = True)
    coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
    train = histoMod[histoMod.index <= coupe]
    
    ordres, ordres_sais = ordre_SARIMA(histoMod, dateDebMod, dateFinMod)
    model = SARIMAX(train['PAX'], order = ordres, seasonal_order = ordres_sais)
    model_fit = model.fit(disp =-1)
    
    prediction = model_fit.get_prediction(start = len(train) - hPrev + 1, end = len(train), freq = 'A')
    PrevisionsSARIMA = pd.DataFrame(histoMod[["ArrDep" , "Faisceau"]]).head(hPrev)
    new_dates =pd.date_range(start = dateFinMod  + timedelta(1), end = dateFinMod  + timedelta(hPrev))
    PrevisionsSARIMA['Date'] = new_dates
    PrevisionsSARIMA['PAX_SARIMA'] = prediction.predicted_mean.values
    
    if ic == 0:
        PrevisionsARIMA['IC'+str(int(ic*100))+'_low_ARIMA'] = 0
        PrevisionsARIMA['IC'+str(int(ic*100))+'_up_ARIMA'] = 0
    else: 
        PrevisionsSARIMA['IC'+str(int(ic*100))+'_low_SARIMA'] = prediction.conf_int(alpha = 1-ic)['lower PAX'].values
        PrevisionsSARIMA['IC'+str(int(ic*100))+'_up_SARIMA'] = prediction.conf_int(alpha = 1-ic)['upper PAX'].values
    
    return(PrevisionsSARIMA)
    

### Validité des modèles

import statsmodels.api as sm

database = pd.read_csv("/Users/victorhuynh/Documents/ENSAE/ENSAE 2A/2A S2/Stat App/StatApp_2020/Data/database_sieges.csv",low_memory=False,decimal=',')
database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

dateDebMod = pd.to_datetime("2007-01-01")
dateFinMod = pd.to_datetime("2016-01-15")

histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
histoMod = histoMod[(histoMod['Faisceau']=='International') & (histoMod['ArrDep']=='Arrivée')]
#A ré-adapter selon le faisceau et le type de mouvement qu'on étudie

histoMod = histoMod.reset_index(drop = True)
coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
train = histoMod[histoMod.index <= coupe]
    
ordres, ordres_sais = ordre_SARIMA(histoMod, dateDebMod, dateFinMod)
res = SARIMAX(train['PAX'], order = ordres, seasonal_order = ordres_sais).fit(disp = 0)

sm.stats.acorr_ljungbox(res.resid, lags=12, return_df=True) #Test de Ljung-Box pour l'autocorrélation des résidus
res.summary() #Pour voir la significativité des coefficients







