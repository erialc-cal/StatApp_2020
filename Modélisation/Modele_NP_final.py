"""
Implémentation du modèle Non-paramétrique 

Plusieurs fonctions implémentées :
    - infos_blocs : crée les blocs à partir de l'historique, et calcule moyenne, ... (évite de refaire le calcul à chaque fois)
    - previsions_NP_h_fixe : réalise des prédictions pour un hyper paramètre h donné
    - rmse : calcule l'erreur rmse (utilisé pour le choix de l'hyper-paramètre h)
    - meilleur_h : choisit le meilleur paramètre h, comme meilleur paramètre qui aurait permis de prédire la dernière année de l'historique (comparaison erreur rmse, pour différentes valeurs de h, entre prédictions et réalisé sur cette dernière année d'historique)
    - previsions_NP : fonction finale, qui réalise les prédictions souhaitées après avoir choisir le meilleur hyper paramètre h 
"""



import math
from datetime import timedelta

import pandas as pd
import numpy as np



def infos_blocs (histoMod,tailleBlocs) :
    """
    Fonction qui crée les blocs à partir de l'historique, 
    et renvoie toutes les informations utilisées par la suite sur ces blocs

    Parameters
    ----------
    histoMod : DataFrame
        Historique du trafic sur lequel faire les blocs
    tailleBlocs : int
        taille des blocs à réaliser

    Returns
    -------
    Blocs, Blocs_CR, LastBlocs_CR : DataFrame
        Blocs, blocs centrés réduits et dernier bloc centré réduit
    Stats : DataFrame
        Moyennes et Ecarts-types de chacun des blocs
    distances : array
        distances entre chacun des blocs centrés réduits avec le dernier bloc centré réduit

    """
    as_strided = np.lib.stride_tricks.as_strided
    
    Blocs = pd.DataFrame(as_strided(histoMod["PAX"], (len(histoMod["PAX"])-(tailleBlocs-1) , tailleBlocs) , (histoMod["PAX"].values.strides * 2)))   
    
    Stats = pd.DataFrame()
    Stats['Mean'] = Blocs.mean(axis=1)
    Stats['Stds'] = Blocs.std(axis=1)
    
    Blocs_CR = (Blocs - np.array(Stats['Mean']).reshape(-1,1)) / np.array(Stats['Stds']).reshape(-1,1)
    LastBloc_CR = pd.DataFrame(np.array(Blocs_CR)[-1].reshape(1,-1))
    
    
    # Calcul des distances entre chaque bloc (utilisées ensuite pour le calcul des poids) :
    distances = np.sum((np.array(LastBloc_CR)-np.array(Blocs_CR[:-1]))**2,axis=1)
    
    return (Blocs, Blocs_CR, LastBloc_CR, Stats, distances)





def previsions_NP_h_fixe (histoMod, Calendrier, dateFinMod, infosBlocs , hPrev, h, ic = 0, tailleBlocs = 365) :
    """
    Fonction qui réalise les prédictions selon le modèle non-paramétrique
    à partir de l'historique donné (déjà filtré pour un faisceau et un type de mouvement) 
    et des caractéristiques des blocs déjà calculées (pour éviter de le refaire à chaque fois dans l'appel de la fonction qui fait une cross-validation),
    pour un hyper paramètre h donné.

    Parameters
    ----------
    histoMod, Calendrier, dateFinMod, hPrev, tailleBlocs, ic : 
        idem que dans la fonction previsions_NP 
    infosBlocs : tuple
        Contient toutes les infos des blocs déjà calculées : Blocs, Blocs_CR, LastBloc_CR, Stats, distances
    h : int
        hyper paramètre h / largeur de la fenêtre

    Returns
    -------
    PrevisionsNP : DataFrame
        Prévisions journalières du modèle (contient 'PAX_NP', 'Date', 'Faisceau', 'ArrDep') 
        + intervalle de confiance 'ICic_low_NP' et 'ICic_up_NP' si ic != 0

    """
    
    Blocs, Blocs_CR, LastBloc_CR, Stats, distances = infosBlocs
    
    
    # Calcul des poids en comparant la similarité des motifs des blocs (noyau gaussien):      
    weights = pd.DataFrame( (1/math.sqrt(2*math.pi)**tailleBlocs) * np.exp(- distances / (2*h)))
    
    
    # Calcul des prévisions une par une : 
    PrevisionsNP = pd.DataFrame()
    datePrev = dateFinMod
    
    
    for horizonPrev in range(1,hPrev+1) :         
        datePrev += timedelta(days=1)
        
        
        # Correction des poids en utilisant le calendrier :    
        CalPrev = np.array(Calendrier[Calendrier['Date']==datePrev])
        histoPrev = histoMod[tailleBlocs - 1 + horizonPrev : ]
        
        indJourPrev = np.array(histoPrev["Pont_LunF"]) == CalPrev[:,5]
        colHistoPrev = ["Pont_MarF","Pont_Mer1F","Pont_Mer2F","Pont_JeuF","Pont_VenF","Vac_Toussaint","Vac_Noel","Vac_Hiver_A","Vac_Hiver_B","Vac_Hiver_C","Vac_Printemps_A","Vac_Printemps_B","Vac_Printemps_C","Vac_Ete"]
        indCalPrev = [6,7,8,9,10,13,14,15,16,17,18,19,20,21]
        for k in range(len(indCalPrev)) :
            indJourPrev = indJourPrev & (np.array(histoPrev[colHistoPrev[k]]) == CalPrev[:,indCalPrev[k]])
        
        # On ajoute une correction avec le jour de la semaine uniquement si cela n'annule pas tous les poids :
        indJourPrev2 = indJourPrev & (np.array(histoPrev["JourSem"]) == CalPrev[:,2])
        if sum(indJourPrev2) != 0 :
            indJourPrev = indJourPrev2
            
        indJourPrev = indJourPrev.reshape(-1,1)

        weightsPrev = indJourPrev * np.array(weights)[:1+len(weights)-horizonPrev]
        
        weightsPrev = np.nan_to_num(weightsPrev) #permet de remplacer les éventuels Nan par 0 (nécéssaire pour l'échantillonnage)
        weightsPrev = pd.DataFrame(weightsPrev)
        
        
        # Normalisation des poids :
        s = np.sum(weightsPrev , axis = 0)
        Sim = weightsPrev / s


        # Calcul de la prévision : 
        histoMod_CR = (np.array(histoMod['PAX'][tailleBlocs-1+horizonPrev:]).reshape(-1,1) - np.array(Stats['Mean'][ : - horizonPrev]).reshape(-1,1)) / np.array(Stats['Stds'][ : - horizonPrev]).reshape(-1,1)
        histoMod_RS = histoMod_CR * np.array(Stats['Stds'])[-1].reshape(-1,1) + np.array(Stats['Mean'])[-1].reshape(-1,1)
        UnePrev =  np.sum(histoMod_RS*np.array(Sim))
        
        
        # si ic != 0 : calcul d'un intervalle de confiance par méthode bootstrap :
        if ic != 0 :
            # Tirage aléatoire avec remise en prenant pour probas : weightsPrev
            B = 1000 #Taille de l'échantillon Bootstrap
            
            echantillon = np.random.choice(histoMod_CR.reshape(1,-1)[0], size = B, replace = True, p=np.array(Sim).reshape(1,-1)[0]) 
            echantillon = echantillon * np.array(Stats['Stds'])[-1].reshape(1,-1)[0] + np.array(Stats['Mean'])[-1].reshape(1,-1)[0]
 
            residus = echantillon - UnePrev
            
            # On prend ensuite simplement les quantiles de l'échantillon des résidus simulé par bootstrap
            p_l = ((1 - ic)/2) * 100
            lower = np.percentile(residus, p_l)
            p_u = ((1 + ic)/2) * 100
            upper = np.percentile(residus, p_u)
            
            # Ajout de la prévision et de son intervalle de confiance à la table finale :
            UnePrev = pd.DataFrame(data={"PAX_NP" : [UnePrev], 'IC'+str(ic*100)+'_low_NP' : [UnePrev+lower] ,'IC'+str(ic*100)+'_up_NP' : [UnePrev+upper]})
            PrevisionsNP = pd.concat([PrevisionsNP , pd.concat([UnePrev , pd.DataFrame([datePrev]) , pd.DataFrame(histoMod[["ArrDep" , "Faisceau"]]).head(1).reset_index().drop(columns = ['index'])] , axis = 1)])
        
        
        else : 
            # Ajout de la prévision à la table finale :
            UnePrev = pd.DataFrame(data=[UnePrev],columns = ["PAX_NP"])
            PrevisionsNP = pd.concat([PrevisionsNP , pd.concat([UnePrev , pd.DataFrame([datePrev]) , pd.DataFrame(histoMod[["ArrDep" , "Faisceau"]]).head(1).reset_index().drop(columns = ['index'])] , axis = 1)])


    return PrevisionsNP.rename(columns = {0 : "Date"})
    
   


    
def rmse (serie1 , serie2) :
    """
    Fonction qui calcule la racine de l'erreur quad moyenne entre deux séries passées en argument

    Parameters
    ----------
    serie1, serie2 : list
        Séries de valeurs (prédites et réelles ici) utilisées pour calculer le RMSE

    Returns
    -------
    rmse : float
        valeur de l'erreur RMSE entre les deux séries passées en argument

    """
    rmse = 0 
    n = len(serie1)
    
    for i in range(n) :
        rmse += (serie1[i]-serie2[i])**2    
    
    return math.sqrt(rmse/n)
   
    
 
    
    
def meilleur_h (histoMod, Calendrier, dateFinMod, hPrev, tailleBlocs) :
    """
    Fonction qui recherche le meilleur h : 
        - sélectionne des périodes de test : périodes présentes dans l'histo, commençant aux mêmes dates 
            que la période à prédire, mais pour des années antérieures (de sorte qu'il y ait toujours au moins un an de dispo dans l'histo)
        - pour chaque période de test, cherche le meilleur h parmi les différents candidats choisis arbitrairement : 
            [5,10,15,20,25,30,35,40,45,50]
        - prend la moyenne des meilleurs h sélectionnés à l'étape précédente 

    Parameters
    ----------
    histoMod, Calendrier, dateFinMod, hPrev, tailleBlocs : 
        idem que dans la fonction previsions_NP et previsions_NP_h_fixe

    Returns
    -------
    h : int
        Meilleur h qui permet de faire les meilleurs prédictions

    """
    candidats_h = [i for i in range(5,55,5)]
    
    meilleurs_h = [] # Liste qui contiendra les meilleurs h retenus pour chaque période testée
    
    nb = len(histoMod)//365 - 1  # Nombre de périodes testées
    
    for i in range(1,nb) : 
        
        # On choisit d'essayer de prédire la même période k années avant :
        dateFinMod2 = dateFinMod - timedelta(days=i*365) 
        
        histoMod2 = histoMod[histoMod['Date']<=dateFinMod2]
        infosBlocs2 = infos_blocs (histoMod2,tailleBlocs)
        
        realise =  histoMod[(histoMod['Date']>dateFinMod2)&(histoMod['Date']<=dateFinMod2+timedelta(days=hPrev))]
        realise = list(realise['PAX'])
    
        # Test de chacun des candidats h, et choix de celui avec la plus petite erreur RMSE :
    
        meilleur_h = candidats_h[0]
        previsions = previsions_NP_h_fixe (histoMod2, Calendrier, dateFinMod2, infosBlocs2 , hPrev, meilleur_h) 
        meilleure_erreur = rmse(realise, list(previsions['PAX_NP']))
    
        for k in range(1,len(candidats_h)) :
            h = candidats_h[k]
            previsions = previsions_NP_h_fixe (histoMod2, Calendrier, dateFinMod2, infosBlocs2 , 365, h)
            erreur = rmse(realise, list(previsions['PAX_NP']))
        
            if erreur < meilleure_erreur :
                meilleur_h = h
                meilleure_erreur = erreur
        meilleurs_h.append(meilleur_h)
        
    # print(histoMod2['Faisceau'][0],histoMod2['ArrDep'][0],hPrev,sum(meilleurs_h)/len(meilleurs_h))
    return sum(meilleurs_h)/len(meilleurs_h)
    

    


def previsions_NP (histoMod, Calendrier, dateDebMod, dateFinMod, hPrev, ic = 0.95, tailleBlocs = 365) :
    """
    Fonction qui réalise les prédictions selon le modèle non-paramètrique 
    à partir de l'historique donné (déjà filtré pour un faisceau et un type de mouvement),
    en choisissant le meilleur hyper paramètre h par cross validation

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
    ic : float, optional
        Correspond au seuil de l'intervalle de confiance souhaité (mettre 0 pour ne pas calculer d'intervalle de confiance). The default is 0.95.

    Returns
    -------
    PrevisionsNP : DataFrame
        Prévisions journalières du modèle (contient 'PAX_NP', 'Date', 'Faisceau', 'ArrDep') + intervalle de confiance 'IC_ic_inf_NP' et 'IC_ic_sup_NP' si ic != 0

    """
    
    # Sélection colonnes intérêt Calendrier + dans les colonnes Pont_LunF, .., Pont_VenF, on remplace par 0 si c'est les vacances en même temps
    colonnesCalendrier = ["Date" , "Mois" , "JourSem" , "Semaine" , "Semaine_Perso" , "Pont_LunF" , "Pont_MarF" , 
                              "Pont_Mer1F" , "Pont_Mer2F" , "Pont_JeuF" , "Pont_VenF" ,"Pont_SamF" , "Pont_DimF" , "Vac_Toussaint" , "Vac_Noel" ,
                              "Vac_Hiver_A" , "Vac_Hiver_B" , "Vac_Hiver_C" ,"Vac_Printemps_A" , "Vac_Printemps_B" , "Vac_Printemps_C" ,"Vac_Ete"]
    Calendrier = np.array(Calendrier[colonnesCalendrier])
    vacances = Calendrier[:,13:].sum(axis=1) == 0 #Contient True si on n'est pas en vacances et False si on est en vacances
    for i in range(5,11) :
        Calendrier[:,i] *= vacances
    Calendrier = pd.DataFrame(Calendrier, columns=colonnesCalendrier)
    
    
    # Augmentation de l'historique avec le calendrier : 
    histoMod = pd.merge(histoMod, Calendrier, left_on = ['Date'], right_on = ['Date'], how = 'left')
    histoMod = histoMod.sort_values(by='Date')
    

    # Création des blocs et des informations utiles :
    infosBlocs = infos_blocs (histoMod,tailleBlocs)
    
    
    # Choix de la meilleure largeur de la fenêtre : on fait appel à une recherche de type cross validation
    h = meilleur_h (histoMod, Calendrier, dateFinMod, hPrev, tailleBlocs) 
    
    
    # Réalisation des prévisions avec le h sélectionné précédemment :
    PrevisionsNP = previsions_NP_h_fixe (histoMod, Calendrier, dateFinMod, infosBlocs , hPrev, h, ic, tailleBlocs)
   
    
    return PrevisionsNP
    
   



#### TEST ####

# dateDebMod = pd.to_datetime("2007-01-01")
# dateFinMod = pd.to_datetime("2016-01-15")

# hPrev = 365
    
# database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
# database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
# database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

# Calendrier = pd.read_csv("Calendrier.csv", dayfirst = True , sep = ';' , parse_dates = ['Date'])

# histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]

# histoPrev = database[(database['Date']>dateFinMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]
    
            
# histoMod_2 = histoMod[(histoMod['Faisceau']=='Schengen') & (histoMod['ArrDep']=='Arrivée')]
           

# test = previsions_NP(histoMod_2, Calendrier, dateDebMod, dateFinMod, hPrev)

