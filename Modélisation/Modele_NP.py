"""
Implémentation du modèle Non-paramétrique
"""

import math
from datetime import timedelta

import pandas as pd
import numpy as np
   

def previsions_NP (histoMod, Calendrier, dateDebMod, dateFinMod, hPrev, tailleBlocs = 365) :
    """
    Fonction qui réalise les prédictions d'un faisceau selon le modèle non paramétrique, à l'horizon donnée

    Parameters
    ----------
    histoMod : DataFrame
        Historique du trafic journalier, sur un seul faisceau et un seul type de mouvement, contenant : 'ArrDep', 'Faisceau', 'Date', 'PAX'
    Calendrier : DataFrame
        Calendrier à utiliser pour déterminer les groupes de blocs
    dateDebMod : datetime64[ns]
        Date de début de l'historique
    dateFinMod : datetime64[ns]
        Date de fin de l'historique
    hPrev : int
        Nombre de jours pour lesquels faire une prédiction du trafic
    tailleBlocs : int, optional
        Taille des blocs du modèle. The default is 365 days.

    Returns
    -------
    None.

    """
    
    # Sélection colonnes intérêt Calendrier + dans les colonnes Pont_LunF, .., Pont_VenF, on remplace par 0 si c'est les vacances en même temps
    colonnesCalendrier = ["Date" , "Mois" , "JourSem" , "Semaine" , "Semaine_Perso" , "Pont_LunF" , "Pont_MarF" , 
                              "Pont_Mer1F" , "Pont_Mer2F" , "Pont_JeuF" , "Pont_VenF" ,"Pont_SamF" , "Pont_DimF" , "Vac_Toussaint" , "Vac_Noel" ,
                              "Vac_Hiver_A" , "Vac_Hiver_B" , "Vac_Hiver_C" ,"Vac_Printemps_A" , "Vac_Printemps_B" , "Vac_Printemps_C" ,"Vac_Ete"]
    Calendrier = np.array(Calendrier[colonnesCalendrier])
    vacances = Calendrier[:,13:].sum(axis=1) > 0
    for i in range(5,11) :
        Calendrier[:,i] *= vacances
    Calendrier = pd.DataFrame(Calendrier, columns=colonnesCalendrier)
    
    
    # Augmentation de l'historique avec le calendrier : 
    histoMod = pd.merge(histoMod, Calendrier, left_on = ['Date'], right_on = ['Date'], how = 'left')
    histoMod = histoMod.sort_values(by='Date')
    
    
    # Choix de la largeur de la fenêtre :
    h = 20  #Faire une fonction annexe à appeler ici, qui choisit le meilleur h 
    
    
    # Création des blocs, de leurs stat et de blocs centrés réduits :
    as_strided = np.lib.stride_tricks.as_strided
    
    Blocs = pd.DataFrame(as_strided(histoMod["PAX"], (len(histoMod["PAX"])-(tailleBlocs-1) , tailleBlocs) , (histoMod["PAX"].values.strides * 2)))   
    
    Stats = pd.DataFrame()
    Stats['Mean'] = Blocs.mean(axis=1)
    Stats['Stds'] = Blocs.std(axis=1)
    
    Blocs_CR = (Blocs - np.array(Stats['Mean']).reshape(-1,1)) / np.array(Stats['Stds']).reshape(-1,1)
    LastBloc_CR = pd.DataFrame(np.array(Blocs_CR)[-1].reshape(1,-1))
    
    
    # Calcul des poids en comparant la similarité des motifs des blocs :    
    distances = np.sum((np.array(LastBloc_CR)-np.array(Blocs_CR[:-1]))**2,axis=1)
    weights = pd.DataFrame( (1/math.sqrt(2*math.pi)**tailleBlocs) * np.exp(- distances / (2*h)))
    
    
    # Calcul des prévisions une par une : 
    PrevisionsNP = pd.DataFrame()
    datePrev = dateFinMod
    
    for horizonPrev in range(1,hPrev+1) :         
        datePrev += timedelta(days=1)
        
        # Correction des poids en utilisant le calendrier :    
        CalPrev = np.array(Calendrier[Calendrier['Date']==datePrev])
        histoPrev = histoMod[tailleBlocs - 1 + horizonPrev : ]
        
        indJourPrev = np.array(histoPrev["JourSem"]) == CalPrev[:,2]
        colHistoPrev = ["Pont_LunF","Pont_MarF","Pont_Mer1F","Pont_Mer2F","Pont_JeuF","Pont_VenF","Vac_Toussaint","Vac_Noel","Vac_Hiver_A","Vac_Hiver_B","Vac_Hiver_C","Vac_Printemps_A","Vac_Printemps_B","Vac_Printemps_C","Vac_Ete"]
        indCalPrev = [5,6,7,8,9,10,13,14,15,16,17,18,19,20,21]
        for k in range(len(indCalPrev)) :
            indJourPrev = indJourPrev & (np.array(histoPrev[colHistoPrev[k]]) == CalPrev[:,indCalPrev[k]])
        print(indJourPrev.sum())
        
        # weightsPrev = indJourPrev * np.array(weights)[:-horizonPrev]
        
        # # Normalisation des poids :
        # s = np.sum(weightsPrev , axis = 0)
        # Sim = weightsPrev / s
        
        
        
    return 

    
    
    
    
# Test sur le faisceau National pour les départs : 

database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
database = database.astype({'Date': 'datetime64[ns]'})

Calendrier = pd.read_csv('Calendrier.csv', dayfirst = True , sep
= ';' , parse_dates = ['Date'])



dateDebMod = pd.to_datetime("2007-01-01")
dateFinMod = pd.to_datetime("2016-10-23")

hPrev = 50

histoMod = database[(database['Faisceau']=='National') & (database['ArrDep']=='Départ') & (database['Aerog'] == 'ORYO')]
histoMod = histoMod[(histoMod['Date']>=dateDebMod) & (histoMod['Date']<=dateFinMod)]

histoMod = histoMod.groupby(['ArrDep','Faisceau','Date']).agg({'PAX':'sum'}).reset_index()

test = previsions_NP(histoMod, Calendrier, dateDebMod, dateFinMod, hPrev)




