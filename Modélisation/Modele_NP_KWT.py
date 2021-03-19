#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implémentation avec décomposition en ondelettes

"""

import os

# En supposant qu'on parte du dossier où se trouve le fichier Modele_NP.py

# ajout pour le path : on pointe sur le dossier Data pour charger les fichiers, plus besoin de recharger les fichiers à chaque fois

print(os.getcwd()) #doit renvoyer le chemin dans le dossier StatApp_2020

DATA_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.path.abspath('/Data'))


import math
from datetime import timedelta

import pandas as pd
import numpy as np

#%% 

database = pd.read_csv(os.path.join(DATA_DIR, "database_sieges.csv"),low_memory=False,decimal=',')
database = database.astype({'Date': 'datetime64[ns]'})

Calendrier = pd.read_csv(os.path.join(DATA_DIR,'Calendrier/Calendrier.csv'), dayfirst = True , sep= ';' , parse_dates = ['Date'])
# Il manque les corrections calendaires 

tailleBlocs = 365
dateDebMod = pd.to_datetime("2007-01-01")
dateFinMod = pd.to_datetime("2016-10-23")

hPrev = 50

histoMod = database[(database['Faisceau']=='National') & (database['ArrDep']=='Départ') & (database['Aerog'] == 'ORYO')]
histoMod = histoMod[(histoMod['Date']>=dateDebMod) & (histoMod['Date']<=dateFinMod)]

histoMod = histoMod.groupby(['ArrDep','Faisceau','Date']).agg({'PAX':'sum'}).reset_index()

#%%%


def previsions_NP_KWT (histoMod, dateFinMod , hPrev) :
    """
    Fonction qui réalise les prédictions selon le modèle non-paramétrique avec ajout de la décomposition en ondelettes
    
    Parameters
    ----------
    histoMod, dateFinMod, hPrev :

    Returns
    -------
    PrevisionsNP : DataFrame
        Prévisions journalières du modèle (contient 'PAX_NP', 'Date', 'Faisceau', 'ArrDep')  avec ondelettes 
    """

    
        
        
    histoMod = histoMod.sort_values(by='Date')
    
    
    # Choix de la largeur de la fenêtre :
    h = 20  
    
    # taille des blocs :
    tailleBlocs = 365
    
    # Création des blocs, de leurs stat et de blocs centrés réduits :
    as_strided = np.lib.stride_tricks.as_strided
    
    Blocs = pd.DataFrame(as_strided(histoMod["PAX"], (len(histoMod["PAX"])-(tailleBlocs-1) , tailleBlocs) , (histoMod["PAX"].values.strides * 2)))   
    #print(Blocs)
    
    
    
    ### Décomposition en ondelettes
    
    ## Première étape 
    
    import pywt 
    
    # Préparation des données
    
    Stats = pd.DataFrame()
    Stats['Mean'] = Blocs.mean(axis=1)
    Stats['Stds'] = Blocs.std(axis=1) #on centre tout et on réduit
    
    Blocs_CR = (Blocs - np.array(Stats['Mean']).reshape(-1,1)) / np.array(Stats['Stds']).reshape(-1,1)
                #décomposition en ondelette 1D alternative
                #data_pad = np.lib.pad(np.ravel(Blocs), (0, 366), 'constant', constant_values=0)
                #(c_1, c_2) = pywt.swt(data_pad, 'db1', level=1)[0]
                #print((c_1, c_2))
    #LastBloc_CR = pd.DataFrame(np.array(Blocs_CR)[-1].reshape(1,-1))
    
    (cA, cD) = pywt.dwt(Blocs_CR, 'db1') #coeff d'approx, coeff détail (voir équation 0)
    #print(len(cA), len(cD))
    
    # ce qui nous intéresse c'est le coeff détail qui donne les structures locales
    
    # Matrice de dissymétrie
    #Last_cD = pd.DataFrame(np.array(cD)[-1].reshape(1,-1))
    
    #D = np.sum((np.array(Last_cD)-np.array(cD[:-1]))**2,axis=1)
    
    
    ## Deuxième étape
    
    # on passe dans un noyau gaussien les coefficients récupérés dans cD afin d'obtenir les poids
    weights =  1/math.sqrt(2*math.pi)* np.exp(- cD/ (2*h))
    
    # transformée inverse :
    Blocs_WKT = pywt.idwt(cA, weights, 'db1')
    
    #print(Blocs_WKT)
    
    LastBloc_WKT = pd.DataFrame(np.array(Blocs_WKT)[-1].reshape(1,-1))
    
    distances = np.sum((np.array(LastBloc_WKT)-np.array(Blocs_WKT[:-1]))**2,axis=1)
    
    
    # calcul des nouveaux poids de prévision 
    coeff =  pd.DataFrame( (1/math.sqrt(2*math.pi)**tailleBlocs) * np.exp(- distances / (2*h)))
    

    
    ### Implémentation des prévisions 
    
    # Le prédicteur est le barycentre des futurs des segments du passé (équation 5)
    PrevisionsNP = pd.DataFrame()
    datePrev = dateFinMod
    
    for horizonPrev in range(1,hPrev+1) :         
        datePrev += timedelta(days=1)
    
        # on normalise les poids 
        coeff_prev = np.array(coeff)[:1+len(coeff)-horizonPrev]
        s = np.sum(coeff_prev , axis = 0)
        Sim = coeff_prev / s
               
        
        
        histoMod_CR = (np.array(histoMod['PAX'][tailleBlocs-1+horizonPrev:]).reshape(-1,1) - np.array(Stats['Mean'][ : - horizonPrev]).reshape(-1,1)) / np.array(Stats['Stds'][ : - horizonPrev]).reshape(-1,1)
        histoMod_RS = histoMod_CR * np.array(Stats['Stds'])[-1].reshape(-1,1) + np.array(Stats['Mean'])[-1].reshape(-1,1)
        #print(np.shape(Sim), np.shape(histoMod_CR))
        
        UnePrev =  ((np.array(histoMod_RS)) * np.array(Sim)).sum().sum()
        
            
            
    # Ajout de la prévision à la table finale :
        UnePrev = pd.DataFrame(data=[UnePrev],columns = ["PAX_NP"])
        PrevisionsNP = pd.concat([PrevisionsNP, pd.concat([UnePrev , pd.DataFrame([datePrev]) , pd.DataFrame(histoMod[["ArrDep" , "Faisceau"]]).head(1).reset_index().drop(columns = ['index'])] , axis = 1)])


    
    return PrevisionsNP.rename(columns = {0 : "Date"})

#%%         
df = previsions_NP_KWT(histoMod, dateFinMod, hPrev)



    
