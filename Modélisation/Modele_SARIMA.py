#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:33:58 2021

@author: victorhuynh
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/victorhuynh/Downloads/database_sieges.csv", parse_dates = ['Date'], index_col = ['Date'])
df1 = df[['PAX']] #On ne garde que la variable PAX
df1 = df1.groupby('Date').agg({'PAX':'mean'})
df1 = df1.drop([pd.to_datetime('2010-04-18'),pd.to_datetime('2010-04-19')])
#On retire ces deux dates foireuses

pourcent_train = 0.70 #Pourcentage des données qu'on veut utiliser en entraînement du modèle
point_sep = round(len(df1) * pourcent_train)
train, test = df1.iloc[:point_sep], df1.iloc[point_sep:]
train = np.log(train)
train_diff = train.diff(periods = 1).dropna()

#On applique les fonctions ACF et PACF
lag_acf = acf(train_diff, nlags = 40)
lag_pacf = pacf(train_diff, nlags = 40)

#Plot de l'ACF
plt.figure(figsize = (15,5))
plt.subplot(121)
plt.stem(lag_acf)
plt.axhline(y = 0, linestyle = '-', color = 'black')
plt.axhline(y = -1.96/np.sqrt(len(train)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(train)), linestyle = '--', color = 'gray')
plt.xlabel('Décalage')
plt.ylabel('Auto-corrélation')

#Plot du PACF
plt.figure(figsize = (15,5))
plt.subplot(121)
plt.stem(lag_pacf)
plt.axhline(y = 0, linestyle = '-', color = 'black')
plt.axhline(y = -1.96/np.sqrt(len(train)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(train)), linestyle = '--', color = 'gray')
plt.xlabel('Décalage')
plt.ylabel('Auto-corrélation partielle')

model = SARIMAX(train, order = (4,1,4), seasonal_order = (1,0,0,7)) #Bien choisir les ordres SARIMA
model_fit = model.fit(disp = False)

K = len(test)
prediction = model_fit.forecast(K) #On va prédire K valeurs
prediction = np.exp(prediction) #Exponentialisation pour remettre à l'échelle convenable nos données
index_dates = pd.date_range(start='2014-04-22', end='2016-12-31')
prediction.index = index_dates

#Comparaisons réel/prédiction sur nos données de test
plt.figure(figsize = (10, 5))
plt.plot(prediction,'r')
plt.plot(df1,'b')
#plt.title('RMSE : %.2f'% np.sqrt(sum(prediction-test)**2)/len(test)) #Cette ligne est foireuse, à corriger
plt.xlabel('Date')
plt.ylabel('PAX')
plt.autoscale(enable = True, axis = 'x', tight = True)