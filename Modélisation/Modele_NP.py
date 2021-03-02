"""
Implémentation du modèle Non-paramétrique
"""


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
    
    
    # Calcul des poids en comparant la similarité des motifs des blocs :
    
    
    
    
    
    
    
    return 
    
    
    
    
# Test sur le faisceau National pour les départs : 

database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
database = database.astype({'Date': 'datetime64[ns]'})

Calendrier = pd.read_csv('Calendrier.csv', dayfirst = True , sep
= ';' , parse_dates = ['Date'])
Calendrier = Calendrier[["Date" , "Mois" , "JourSem" ,
"Semaine" , "Semaine_Perso" , "Pont_LunF" , "Pont_MarF" ,
"Pont_Mer1F" , "Pont_Mer2F" , "Pont_JeuF" , "Pont_VenF" ,
"Pont_SamF" , "Pont_DimF" , "Vac_Toussaint" , "Vac_Noel" ,
"Vac_Hiver_A" , "Vac_Hiver_B" , "Vac_Hiver_C" ,
"Vac_Printemps_A" , "Vac_Printemps_B" , "Vac_Printemps_C" ,
"Vac_Ete"]]


dateDebMod = pd.to_datetime("2007-01-01")
dateFinMod = pd.to_datetime("2016-10-23")

hPrev = 2

histoMod = database[(database['Faisceau']=='National') & (database['ArrDep']=='Départ')]
histoMod = histoMod[(histoMod['Date']>=dateDebMod) & (histoMod['Date']<=dateFinMod)]

histoMod = histoMod.groupby(['ArrDep','Faisceau','Date']).agg({'PAX':'sum'}).reset_index()

test = previsions_NP(histoMod, Calendrier, dateDebMod, dateFinMod, hPrev)
