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
!pip install pmdarima 
from pmdarima import auto_arima

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

model = SARIMAX(train, order = (2,0,3), seasonal_order = (2,1,1,12)) #Bien choisir les ordres SARIMA
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


# Pour trouver les meilleurs ordres SARIMA :

meilleur_modele = auto_arima(train['PAX'], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

meilleur_modele.summary()



# Ce code m'a l'air de donner des meilleures prédictions et un bel intervalle de confiance :

model = SARIMAX(df1, order = (2,0,3), seasonal_order = (2,1,1,12))
model_fit = model.fit(disp =-1)

prediction = model_fit.get_prediction(start=pd.to_datetime('2014-04-22'), dynamic=False)
inter_conf = prediction.conf_int()
sup_conf = pd.Series(inter_conf['upper PAX'])
inf_conf = pd.Series(inter_conf['lower PAX'])

ax = df1['PAX'].plot(label='Réalisé')
prediction.predicted_mean.plot(ax=ax, label='Prédiction', alpha=.7, figsize=(14, 4))
ax.fill_between(inter_conf.index,
                sup_conf,
                inf_conf, color='k', alpha=.2, label = 'Intervalle de confiance')
ax.set_xlabel('Date')
ax.set_ylabel('PAX')
plt.legend()
plt.show()

