#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:53:39 2021

@author: victorhuynh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import plotly.graph_objs as go
from  plotly.offline import plot


def validite_residus_ARIMA(histoMod, dateDebMod, dateFinMod, p, d, q) :
    
    bool = True
    histoMod = histoMod.reset_index(drop = True)
    coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
    train = histoMod[histoMod.index <= coupe]
    model = sm.tsa.ARIMA(histoMod["PAX"], (p,d,q)).fit(disp=0)     
    
    lj = sm.stats.acorr_ljungbox(model.resid, lags = 24)
    pval_corrigee = stats.chi2.sf(lj[-1], 24 - p - q)
    
    for i in pval_corrigee :
        if i < 0.05 :
            bool = False
    
    if bool == True :
        print('Les résidus ne sont pas corrélés')
    else :
        print('Les résidus sont corrélés')

# Si toutes les p-valeurs sont supérieures à 0.05, alors on ne rejette pas l'hypothèse de non corrélation des résidus. 
    

def validite_param_ARIMA(histoMod, dateDebMod, dateFinMod, p, d, q) :
    
    bool = True
    histoMod = histoMod.reset_index(drop = True)
    coupe = np.where(histoMod['Date'] == dateFinMod)[0][0]
    train = histoMod[histoMod.index <= coupe]
    model = sm.tsa.ARIMA(histoMod["PAX"], (p,d,q)).fit(disp=0)     
    
    if p == 0 :
        if model.pvalues[q] > 0.05 : 
            bool = False
    elif q == 0 :
        if model.pvalues[p] > 0.05 : 
            bool = False
    else : 
        if model.pvalues[p] > 0.05 : 
            bool = False
        if model.pvalues[p+q] > 0.05 : 
            bool = False

    if bool == True :
        print('Les paramètres sont significatifs')
    else :
        print('Les paramètres ne sont pas significatifs')
    
def normalite_residus_ARIMA(histoMod, dateDebMod, dateFinMod, p, d, q) :
    trace = go.Histogram(x=model.resid)
    plot([motrace])

#Exemple complet d'exécution de ces fonctions : 
        
# database = pd.read_csv("/Users/victorhuynh/Documents/ENSAE/ENSAE 2A/2A S2/Stat App/StatApp_2020/Data/database_sieges.csv",low_memory=False,decimal=',')
# database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
# database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

# dateDebMod = pd.to_datetime("2007-01-01")
# dateFinMod = pd.to_datetime("2016-01-15")

# histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
# histoMod = histoMod[(histoMod['Faisceau']=='National') & (histoMod['ArrDep']=='Départ')] 

# p, d, q = 3, 0, 2
        
# validite_residus_ARIMA(histoMod, dateDebMod, dateFinMod, p, d, q)
# validite_param_ARIMA(histoMod, dateDebMod, dateFinMod, p, d, q)

