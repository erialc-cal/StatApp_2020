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



def previsions_SARIMA (histoMod, dateDebMod, dateFinMod, hPrev, ic = 0.95) :
    
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
    
    if ic == 0 :
        PrevisionsSARIMA['IC'+str(int(ic*100))+'_low_SARIMA'] = [0 for k in range(hPrev)]
        PrevisionsSARIMA['IC'+str(int(ic*100))+'_up_SARIMA'] = [0 for k in range(hPrev)]        
    else :     
        PrevisionsSARIMA['IC'+str(int(ic*100))+'_low_SARIMA'] = prediction.conf_int(alpha = 1-ic)['lower PAX'].values
        PrevisionsSARIMA['IC'+str(int(ic*100))+'_up_SARIMA'] = prediction.conf_int(alpha = 1-ic)['upper PAX'].values
    
    return(PrevisionsSARIMA)
    

    
#Exemple complet d'exécution de ces fonctions : 

# dateDebMod = pd.to_datetime("2007-01-01")
# dateFinMod = pd.to_datetime("2016-01-15")

# hPrev = 365
    
# database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
# database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
# database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

# histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
# histoMod_2 = histoMod[(histoMod['Faisceau']=='Schengen') & (histoMod['ArrDep']=='Arrivée')]
           
# test = previsions_SARIMA(histoMod_2, dateDebMod, dateFinMod, hPrev)
# print(test)



# #Visualisation des résultats :


# fig, ax = plt.subplots()
# plt.plot(histoMod_2['Date'],histoMod_2['PAX'], label='Réalisé')
# plt.plot(test['Date'],test['PAX_SARIMA'], label='Prédiction')
# #test['PAX_ARIMA'].plot(ax=ax, label='Prédiction', alpha=.7, figsize=(14, 4))
# plt.fill_between(test['Date'],
#                 test['IC95_up_SARIMA'],
#                 test['IC95_low_SARIMA'], color='k', alpha=.2, label = 'Intervalle de confiance')
# ax.set_xlabel('Date')
# ax.set_ylabel('PAX')
# plt.legend()
# plt.show()





#### TEST - Réalisation des prévisions Sarima à part ####

# dateDebMod = pd.to_datetime("2008-01-01")
# dateFinMod = pd.to_datetime("2015-12-31")


# hPrev = 7   # faire avec : 7, 31+29+31, 365
# ic = 0.95   # Seuil de l'intervalle de confiance souhaité

    
# database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
# database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
# database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

# Calendrier = pd.read_csv("Calendrier.csv", dayfirst = True , sep = ';' , parse_dates = ['Date'])

# histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
# #     # histoMod.to_csv("HistoMod.csv")
    
# histoPrev = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]
        


# Prev_Sarima_1 = pd.DataFrame()

# for faisceau in ['National', 'Schengen', 'Autre UE', 'Dom Tom', 'International']: 
#     for mvt in ['Départ']: #, 'Arrivée']: 
            
#         histoMod_2 = histoMod[(histoMod['Faisceau']==faisceau) & (histoMod['ArrDep']==mvt)]

# #                 # Modèle Sarima :
#         prev_Sarima = previsions_SARIMA(histoMod_2, dateDebMod, dateFinMod, hPrev, ic)
#         Prev_Sarima_1 = pd.concat([Prev_Sarima_1, prev_Sarima],ignore_index=True) 

# # Prev_Sarima_1.to_csv("Prev_Sarima_1.csv")  

               
# # Ajout des prévisions aux autres :          
   

# autres_pred = pd.read_csv("Previsions_"+str(hPrev)+"j.csv",low_memory=False,decimal=',')
# autres_pred = autres_pred.astype({'Date': 'datetime64[ns]'})
# autres_pred.drop(columns=["PAX_SARIMA","IC95_low_SARIMA","IC95_up_SARIMA"],inplace=True)


# # Prev_Sarima_1 = pd.read_csv('Prev_Sarima_1.csv',low_memory=False,decimal=',')
# # Prev_Sarima_1 = Prev_Sarima_1.astype({'Date': 'datetime64[ns]'})
# # Prev_Sarima = pd.concat([Prev_Sarima_1,Prev_Sarima_2])
# # Prev_Sarima = Prev_Sarima.reset_index()
       
# resultat = pd.concat([autres_pred.set_index(['Date','Faisceau','ArrDep']),
#                                  Prev_Sarima_1.set_index(['Date','Faisceau','ArrDep'])],axis=1)

# resultat.to_csv("Previsions_"+str(hPrev)+"j.csv")


