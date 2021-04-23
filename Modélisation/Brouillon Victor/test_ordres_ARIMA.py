#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:00:05 2021

@author: victorhuynh
"""

import pandas as pd
from datetime import timedelta


# KPSS test pour déterminer si la série est stationnaire ou non

from statsmodels.tsa.stattools import kpss
def kpss_test(series, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    resultat = ""
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    if p_value < 0.05 : 
        resultat = 'Result: The series is not stationary'
    else :
        resultat = 'Result: The series is stationary'
    return(resultat)

#Détermination de l'ordre optimal : 

#Détermination de d : on différentie la série jusqu'à ce qu'on obtienne une série stationnaire
    
histoMod_2 = histoMod[(histoMod['Faisceau']=='Schengen') & (histoMod['ArrDep']=='Arrivée')]
    
resultat = ""
serie_diff = histoMod_2['PAX']
d = 0
while resultat != 'Result: The series is stationary' :
    serie_diff = serie_diff - serie_diff.shift() # Différenciation 
    serie_diff = serie_diff.dropna() #Retrait de la première ligne qui est un NaN 
    d = d+1 #Ordre de différenciation
    resultat = kpss_test(serie_diff)
print(d)

#Détermination de pmax et qmax : on plot l'ACF et le PACF de la série différentiée et stationnaire

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8)) 
plot_acf(serie_diff, lags=30, zero=False, ax=ax1) # plot de l'ACF 
plot_pacf(serie_diff, lags=30, zero=False, ax=ax2) # plot du PACF 

#On trouve pmax = 7, qmax = 24

!pip install pmdarima 
from pmdarima import auto_arima

def ordre_ARIMA(histoMod, pmax, qmax, dateDebMod, dateFinMod):
    
    histoMod = histoMod.reset_index(drop = True)
    coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
    train = histoMod[histoMod.index <= coupe]
    
    stepwise_fit = auto_arima(train['PAX'],max_p= pmax, max_q= qmax, suppress_warnings=True)           

    print(stepwise_fit.summary())

dateDebMod = pd.to_datetime("2008-01-01")
dateFinMod = pd.to_datetime("2015-12-31")
ordre_ARIMA(histoMod, 7, 24, dateDebMod, dateFinMod)
#Sortie : ARIMA(3, 1, 2)

##On reprend la fonction ARIMA : 

import numpy as np
from matplotlib import pyplot as plt

from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

!pip install pmdarima 
from pmdarima import auto_arima

def previsions_ARIMA(histoMod, dateDebMod, dateFinMod, hPrev, p, d, q) :
    
    histoMod = histoMod.reset_index(drop = True)
    coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
    train = histoMod[histoMod.index <= coupe]
    
    model = SARIMAX(train['PAX'], order = (p,d,q))
    model_fit = model.fit(disp =-1)
    
    prediction = model_fit.get_prediction(start = len(train) - hPrev + 1, end = len(train), freq = 'A')
    PrevisionsARIMA = pd.DataFrame(histoMod_2[["ArrDep" , "Faisceau"]]).head(hPrev)
    new_dates =pd.date_range(start = dateFinMod  + timedelta(1), end = dateFinMod  + timedelta(hPrev))
    PrevisionsARIMA['Date'] = new_dates
    PrevisionsARIMA['PAX_ARIMA'] = prediction.predicted_mean.values
    PrevisionsARIMA['IC95.0_low_ARIMA'] = prediction.conf_int(alpha = 0.05)['lower PAX'].values
    PrevisionsARIMA['IC95.0_up_ARIMA'] = prediction.conf_int(alpha = 0.05)['upper PAX'].values
    
    return(PrevisionsARIMA)

#Comparaison

dateDebMod = pd.to_datetime("2008-01-01")
dateFinMod = pd.to_datetime("2015-12-31")


horizonsPrev = [365] # 7, 31+29+31, 365]  # (en jours)
ic = 0.95   # Seuil de l'intervalle de confiance souhaité
            


if __name__ == '__main__':
    
    database = pd.read_csv("/Users/victorhuynh/Documents/ENSAE/ENSAE 2A/2A S2/Stat App/StatApp_2020/Data/database_sieges.csv",low_memory=False,decimal=',')
    database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
    database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

    Calendrier = pd.read_csv("/Users/victorhuynh/Documents/ENSAE/ENSAE 2A/2A S2/Stat App/StatApp_2020/Data/Calendrier/Calendrier.csv", dayfirst = True , sep = ';' , parse_dates = ['Date'])

    histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
    # histoMod.to_csv("HistoMod.csv")
    
    
    for hPrev in horizonsPrev :
        
        # on va ajouter les prévisions à l'historique précédent + sur la période de prévisions
        histoPrev = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]
            # ( sans historique précédent : histoPrev = database[(database['Date']>dateFinMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]   )
    
        Prev_Arima_pratique = pd.DataFrame()
        Prev_Arima_theorie = pd.DataFrame()

    
        for faisceau in ['National', 'Schengen', 'Autre UE', 'International', 'Dom Tom'] :
            for mvt in ['Départ']: #, 'Arrivée']: 
            
                histoMod_2 = histoMod[(histoMod['Faisceau']==faisceau) & (histoMod['ArrDep']==mvt)]

                # Modèle Arima avec les meilleurs paramètres en pratique :
                prev_Arima_pratique = previsions_ARIMA(histoMod_2, dateDebMod, dateFinMod, hPrev, 5,1,5)
                Prev_Arima_pratique = pd.concat([Prev_Arima_pratique, prev_Arima_pratique],ignore_index=True) 

                # Modèle Arima avec les meilleurs paramètres en théorie :
                prev_Arima_theorie = previsions_ARIMA(histoMod_2, dateDebMod, dateFinMod, hPrev, 3,1,2)
                Prev_Arima_theorie = pd.concat([Prev_Arima_theorie, prev_Arima_theorie],ignore_index=True) 
                    
                    
        # Ajout des prévisions des différents modèles à histoPrev           
        histoPrev = pd.concat([histoPrev.set_index(['Date','Faisceau','ArrDep']),
                               Prev_Arima_pratique.set_index(['Date','Faisceau','ArrDep']),
                               Prev_Arima_theorie.set_index(['Date','Faisceau','ArrDep']),
                               ],
                               axis=1)
        histoPrev = histoPrev.reset_index()

        histoPrev.to_csv("Previsions_"+str(hPrev)+"j.csv")

test = pd.read_csv("Previsions_"+str(hPrev)+"j.csv",low_memory=False,decimal=',')

comparaison = test[['Date','Faisceau','PAX','PAX_ARIMA','PAX_ARIMA.1']]
comparaison = comparaison.dropna()
comparaison = comparaison.astype({'Date': 'datetime64[ns]', 'PAX_ARIMA': 'float64', 'PAX_ARIMA.1': 'float64', 'PAX': 'float64'})

plt.plot(comparaison['Date'],comparaison['PAX_ARIMA'])
plt.plot(comparaison['Date'],comparaison['PAX_ARIMA.1'])


import math

def rmse2(predictions, targets):
    MSE = np.square(np.subtract(predictions,targets)).mean() 
    return math.sqrt(MSE)

rmse2(comparaison['PAX'],comparaison['PAX_ARIMA'])
rmse2(comparaison['PAX'],comparaison['PAX_ARIMA.1'])

#Comparaison des RMSE :

for faisceau in ['National', 'Schengen', 'Autre UE', 'International', 'Dom Tom'] :
    comparaison_2 = comparaison[comparaison['Faisceau'] == faisceau].dropna()
    rmse_pratique = rmse2(comparaison_2['PAX'],comparaison_2['PAX_ARIMA'])
    rmse_theorique = rmse2(comparaison_2['PAX'],comparaison_2['PAX_ARIMA.1'])
    print("Pour le faisceau " + faisceau, "le RMSE vaut", rmse_pratique,"pour le meilleur modèle en pratique et ",rmse_theorique, "pour le meilleur modèle en choisissant les ordres théoriquement")
    

def mape(a, b): 
    mask = a != 0
    return (np.fabs(a - b)/a)[mask].mean()

mape(comparaison['PAX'],comparaison['PAX_ARIMA']) 
mape(comparaison['PAX'],comparaison['PAX_ARIMA.1']) 

for faisceau in ['National', 'Schengen', 'Autre UE', 'International', 'Dom Tom'] :
    comparaison_2 = comparaison[comparaison['Faisceau'] == faisceau].dropna()
    rmse_pratique = mape(comparaison_2['PAX'],comparaison_2['PAX_ARIMA'])
    rmse_theorique = mape(comparaison_2['PAX'],comparaison_2['PAX_ARIMA.1'])
    print("Pour le faisceau " + faisceau, "le MAPE vaut", rmse_pratique,"pour le meilleur modèle en pratique et ",rmse_theorique, "pour le meilleur modèle en choisissant les ordres théoriquement")
    
