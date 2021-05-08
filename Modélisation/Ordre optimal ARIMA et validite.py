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
    
    stepwise_fit = auto_arima(train['PAX'], suppress_warnings=True)           

    return(stepwise_fit.order)
    
### Validité des modèles

import statsmodels.api as sm

database = pd.read_csv("/Users/victorhuynh/Documents/ENSAE/ENSAE 2A/2A S2/Stat App/StatApp_2020/Data/database_sieges.csv",low_memory=False,decimal=',')
database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

dateDebMod = pd.to_datetime("2007-01-01")
dateFinMod = pd.to_datetime("2016-01-15")

histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
histoMod = histoMod[(histoMod['Faisceau']=='Autre UE') & (histoMod['ArrDep']=='Départ')] 
#A ré-adapter selon le faisceau et le type de mouvement qu'on étudie

histoMod = histoMod.reset_index(drop = True)
coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
train = histoMod[histoMod.index <= coupe]
    
ordres = ordre_ARIMA(histoMod, dateDebMod, dateFinMod)
res = sm.tsa.ARMA(histoMod["PAX"], ordres_ARMA).fit(disp=0)

sm.stats.acorr_ljungbox(res.resid, lags=12, return_df=True) #Test de Ljung-Box pour l'autocorrélation des résidus
res.summary() #Pour voir la significativité des coefficients