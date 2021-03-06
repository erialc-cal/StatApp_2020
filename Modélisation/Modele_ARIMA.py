#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 22:06:55 2021

@author: victorhuynh
"""

#Importation des modules nécessaires 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


def test_statio(serie, variable):
    
    # Statistiques mobiles
    rolling_mean = serie.rolling(window=12).mean()
    rolling_std = serie.rolling(window=12).std()
    
    #Tracé statistiques mobiles
    original = plt.plot(serie, color='blue', label='Valeurs initiales')
    mean = plt.plot(rolling_mean, color='red', label='Moyenne Mobile')
    std = plt.plot(rolling_std, color='black', label='Ecart-type Mobile')
    plt.legend(loc='best')
    plt.title('Moyenne et écart-type Mobiles')
    plt.show(block=False)
    
    #Test de Dickey–Fuller :
    result = adfuller(serie[variable])
    print('Statistiques ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
        

def simulation_ARIMA(variable,p,d,q): 
#Il faut saisir la variable qu'on souhaite prédire, le faisceau concerné, le type de mouvement (A/D) et les paramètres ARIMA

    df = pd.read_csv("/Users/victorhuynh/Downloads/database_sieges.csv", parse_dates = ['Date'], index_col = ['Date'])
    df = df[~(df.index.isin([pd.to_datetime('2010-04-18'),pd.to_datetime('2010-04-19')]))] 
    #Retrait de ces deux dates où les données sont nulles (car incident volcanique)
    
    df1 = df[[variable]]
    df1 = df1.groupby('Date').agg({variable:'mean'})
    #On retire ces deux dates où le trafic est nul
        
    df_log = np.log(df1)
    #Passer au log pour réduire les variations

    rolling_mean = df_log.rolling(window=12).mean()
    df_log_minus_mean = df_log - rolling_mean
    #Pour stationnariser
    test_statio(df_log_minus_mean.dropna(), variable)

    df_log_shift = df_log - df_log.shift()
    df_log_shift.dropna(inplace=True)
    test_statio(df_log_shift, variable)

    #ARIMA en action
    decomposed = seasonal_decompose(df_log, freq=1)
    model = ARIMA(df_log, order=(p,d,q))
    results = model.fit(disp=-1)
    plot1 = plt.figure(1)
    plt.plot(df_log_minus_mean)
    plt.plot(results.fittedvalues, color='red')
    #On compare prédiction et réalité

    #Pour comparer à la série de base : il faut effectuer diverses opérations, dont le renversement de la différentiation et le passage à exp
    predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(df_log['PAX'].iloc[0], index=df_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plot2 = plt.figure(2)
    plt.plot(df1)
    plt.plot(predictions_ARIMA)
    #On compare prédiction et réalité, mais cette fois avec les valeurs de base
    
    return(predictions_ARIMA)
    #La fonction retourne un dataframe avec les données prédites par ARIMA
    
#Exemple d'exécution : simulation_ARIMA('PAX',5,1,1)
    
#Pour obtenir les ordres p,d et q : 
def ordre_ARIMA(variable):
    
    df = pd.read_csv("/Users/victorhuynh/Downloads/database_sieges.csv", parse_dates = ['Date'], index_col = ['Date'])
    df = df[~(df.index.isin([pd.to_datetime('2010-04-18'),pd.to_datetime('2010-04-19')]))] 
    #Retrait de ces deux dates où les données sont nulles (car incident volcanique)
    
    df1 = df[[variable]]
    df1 = df1.groupby('Date').agg({variable:'mean'})
    #On retire ces deux dates où le trafic est nul
        
    df_log = np.log(df1)
    
    stepwise_fit = auto_arima(df_log[variable], suppress_warnings=True)           

    print(stepwise_fit.summary())
    
#Exemple d'exécution : ordre_ARIMA('PAX')
    
    
    
    
    
    
#Version sans passage au log et différenciation uniquement (pour stationnariser)
    
def simulation_ARIMA(variable,p,d,q): 
#Il faut saisir la variable qu'on souhaite prédire, le faisceau concerné, le type de mouvement (A/D) et les paramètres ARIMA

    df = pd.read_csv("/Users/victorhuynh/Downloads/database_sieges.csv", parse_dates = ['Date'], index_col = ['Date'])
    df = df[~(df.index.isin([pd.to_datetime('2010-04-18'),pd.to_datetime('2010-04-19')]))] 
    #Retrait de ces deux dates où les données sont nulles (car incident volcanique)
    
    df1 = df[[variable]]
    df1 = df1.groupby('Date').agg({variable:'mean'})
    #On retire ces deux dates où le trafic est nul

    df1_shift = df1 - df1.shift()
    df1_shift.dropna(inplace=True)
    test_statio(df1_shift, variable)

    #ARIMA en action
    model = ARIMA(df1, order=(p,d,q))
    results = model.fit(disp=-1)
    plot1 = plt.figure(1)
    plt.plot(df1_shift)
    plt.plot(results.fittedvalues, color='red')
    #On compare prédiction et réalité

    #Pour comparer à la série de base : il faut effectuer diverses opérations, dont le renversement de la différentiation et le passage à exp
    predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA = pd.Series(df1['PAX'].iloc[0], index=df1.index)
    predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    plot2 = plt.figure(2)
    plt.plot(df1)
    plt.plot(predictions_ARIMA)
    #On compare prédiction et réalité, mais cette fois avec les valeurs de base
    
    return(predictions_ARIMA)
    #La fonction retourne un dataframe avec les données prédites par ARIMA
    
#Exemple d'exécution : simulation_ARIMA('PAX',5,1,1)
    
    
    
    

# Version avec cross-validation et intervalle de confiance :
    
    
df = pd.read_csv("/Users/victorhuynh/Downloads/database_sieges.csv", parse_dates = ['Date'], index_col = ['Date'])
df1 = df[['PAX']] #On ne garde que la variable PAX
df1 = df1.groupby('Date').agg({'PAX':'mean'})
df1 = df1.drop([pd.to_datetime('2010-04-18'),pd.to_datetime('2010-04-19')])

pourcent_train = 0.70
point_sep = round(len(df1) * pourcent_train)
train, test = df1.iloc[:point_sep], df1.iloc[point_sep:]
train_diff = train.diff(periods = 1).dropna()

model = ARIMA(train, order=(5,1,4)) #Paramètres choisis en utilisant la fonction précédente
model_fit = model.fit(disp= -1)

prediction, se, conf = model_fit.forecast(len(test), alpha=0.05)
# On construit un intervalle de confiance à 95%
prediction_serie = pd.Series(prediction, index=test.index)
sup_conf = pd.Series(conf[:, 0], index=test.index)
inf_conf = pd.Series(conf[:, 1], index=test.index)

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='Entraînement')
plt.plot(test, label='Test')
plt.plot(prediction_serie, label='Prédiction')
plt.fill_between(inf_conf.index, inf_conf, sup_conf, 
                 color='k', alpha=.15)
plt.title('Prédiction vs Test')
plt.legend(loc='upper left', fontsize=8)
plt.show()