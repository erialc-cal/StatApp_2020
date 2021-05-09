#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:53:39 2021

@author: victorhuynh
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

# !pip install pmdarima 
from pmdarima import auto_arima



def ordre_ARIMA(histoMod, dateDebMod, dateFinMod):
    
    histoMod = histoMod.reset_index(drop = True)
    coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
    train = histoMod[histoMod.index <= coupe]
    
    stepwise_fit = auto_arima(train['PAX'], 
                              error_action='ignore', 
                              suppress_warnings=True, 
                              stepwise=True)           

    return(stepwise_fit.order)



def previsions_ARIMA(histoMod, dateDebMod, dateFinMod, hPrev, ic = 0.95) :
    
    histoMod = histoMod.reset_index(drop = True)
    coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
    train = histoMod[histoMod.index <= coupe]
    
    ordres = ordre_ARIMA(histoMod, dateDebMod, dateFinMod)
    model = SARIMAX(train['PAX'], order = ordres)
    model_fit = model.fit(disp =-1)
    
    prediction = model_fit.get_prediction(start = len(train) - hPrev + 1, end = len(train), freq = 'A')
    PrevisionsARIMA = pd.DataFrame(histoMod[["ArrDep" , "Faisceau"]]).head(hPrev)
    new_dates =pd.date_range(start = dateFinMod  + timedelta(1), end = dateFinMod  + timedelta(hPrev))
    PrevisionsARIMA['Date'] = new_dates
    PrevisionsARIMA['PAX_ARIMA'] = prediction.predicted_mean.values
    
    if ic == 0 : 
        PrevisionsARIMA['IC'+str(int(ic*100))+'_low_ARIMA'] = [0 for k in range(hPrev)]
        PrevisionsARIMA['IC'+str(int(ic*100))+'_up_ARIMA'] = [0 for k in range(hPrev)]
    else :    
        PrevisionsARIMA['IC'+str(int(ic*100))+'_low_ARIMA'] = prediction.conf_int(alpha = 1-ic)['lower PAX'].values
        PrevisionsARIMA['IC'+str(int(ic*100))+'_up_ARIMA'] = prediction.conf_int(alpha = 1-ic)['upper PAX'].values
    
    return(PrevisionsARIMA)

    

#Exemple complet d'exécution de ces fonctions : 

# dateDebMod = pd.to_datetime("2007-01-01")
# dateFinMod = pd.to_datetime("2016-01-15")

# hPrev = 365
    
# database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
# database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
# database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

# histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
# histoMod_2 = histoMod[(histoMod['Faisceau']=='Schengen') & (histoMod['ArrDep']=='Arrivée')]
           
# test = previsions_ARIMA (histoMod_2, dateDebMod, dateFinMod, hPrev,0)
# print(test)



# #Visualisation des résultats : 

# fig, ax = plt.subplots()
# plt.plot(histoMod_2['Date'],histoMod_2['PAX'], label='Réalisé')
# plt.plot(test['Date'],test['PAX_ARIMA'], label='Prédiction')
# #test['PAX_ARIMA'].plot(ax=ax, label='Prédiction', alpha=.7, figsize=(14, 4))
# plt.fill_between(test['Date'],
#                 test['IC95_up_ARIMA'],
#                 test['IC95_low_ARIMA'], color='k', alpha=.2, label = 'Intervalle de confiance')
# ax.set_xlabel('Date')
# ax.set_ylabel('PAX')
# plt.legend()
# plt.show()
    