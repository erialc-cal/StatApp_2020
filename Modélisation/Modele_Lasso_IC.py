"""
Implémentation du modèle Lasso 

Plusieurs fonctions implémentées :

    
"""

from datetime import timedelta

import numpy as np
import pandas as pd
#% pip install mlinsights

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LassoCV
#pour obtenir l'IC en bootstrap
from mlinsights.mlmodel import IntervalRegressor
from sklearn.ensemble import BaggingRegressor


def previsions_Lasso (histoMod, Calendrier, dateDebMod, dateFinMod, hPrev, ic=0.95) :
    """
    Fonction qui réalise les prédictions selon le modèle de régression Lasso
    sur des indicatrices temporelles crées à partir du Calendrier,
    à partir de l'historique donné (déjà filtré pour un faisceau et un type de mouvement)

    Parameters
    ----------
    histoMod : DataFrame
        Historique du trafic journalier, sur un seul faisceau et un seul type de mouvement, contenant : 'ArrDep', 'Faisceau', 'Date', 'PAX'
    Calendrier : DataFrame
        Calendrier à utiliser pour créer les features de la régression
    dateDebMod : datetime64[ns]
        Date de début de l'historique
    dateFinMod : datetime64[ns]
        Date de fin de l'historique
    hPrev : int
        Nombre de jours pour lesquels faire une prédiction du trafic

    Returns
    -------
    PrevisionsLasso : DataFrame
        Prévisions journalières du modèle (contient 'PAX_NP', 'Date', 'Faisceau', 'ArrDep')

    """
    
    # Sélection des variables d'intérêt du Calendrier : 
    colonnesCalendrier = ["Mois" , "JourSem" , "Semaine" , "LunF" , "MarF" , 
                              "Mer1F" , "Mer2F" , "JeuF" , "VenF" ,"SamF" , "DimF" , "Vac_Toussaint" , "Vac_Noel" ,
                              "Vac_Hiver_A" , "Vac_Hiver_B" , "Vac_Hiver_C" ,"Vac_Printemps_A" , "Vac_Printemps_B" , "Vac_Printemps_C" ,"Vac_Ete"]
    Calendrier = Calendrier[["Date"]+colonnesCalendrier]
    Calendrier = Calendrier.fillna(0)
    
    
    # Idée : on transforme les colonnes de vacances en indicatrices qui valent 1 si on est en première semaine de vacances (variable <= med(variable))
    # for k in range(1,10) :
    #     mediane_k = Calendrier[colonnesCalendrier[-k]].median()
    #     Calendrier[colonnesCalendrier[-k]] = Calendrier[colonnesCalendrier[-k]] <= mediane_k
    # => sur le test Schengen/Arrivées/hPrev=7j, ça diminue le score du modèle...
    
    
    # Augmentation histo & Récupération données utilisées pour l'entrainement :
    histoMod = pd.concat([histoMod.set_index('Date'), Calendrier.set_index('Date')], axis=1, join="inner")
    histoMod = histoMod.reset_index()
    y_mod = histoMod['PAX']
    X_mod = histoMod[colonnesCalendrier]
    
    
    # Récupération données utilisées pour les prédictions :
    X_pred = Calendrier[(Calendrier['Date']>dateFinMod)&(Calendrier['Date']<=dateFinMod+timedelta(days=hPrev))]
    X_pred = X_pred[colonnesCalendrier]
    
    
    # Création des indicatrices pour chaque colonne du calendrier ayant plus de 2 modalités (JourSem, Mois, ...) :
    encodeur = OneHotEncoder(drop='first').fit(X_mod)
    X_mod = encodeur.transform(X_mod).toarray()
    X_pred = encodeur.transform(X_pred).toarray()
            # la commande drop='first' permet d'enlever l'une des indicatrices à chaque fois (la première arbitrairement)

    
    # On peut désormais entrainer la régression Lasso et calculer les prédictions : 
    modele = LassoCV(cv=5,eps=0.001, n_alphas=100, normalize=True)
            # L'argument normalize=True va normaliser toutes les colonnes de X
    modele.fit(X_mod,y_mod)
    
    predictions = modele.predict(X_pred)
    
    # Calcul des IC par bootstrap
    
    reg= BaggingRegressor(LassoCV(cv=5, random_state=0),bootstrap=True)
    
    ic_model = IntervalRegressor(alpha=1-ic, estimator=reg)
    ic_model.fit(X_mod, y_mod)
    
    sorted_X = np.array(list(sorted(X_mod)))
    bootstrapped_pred = ic_model.predict_sorted(sorted_X)
    min_pred = bootstrapped_pred[:, 0]
    max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1]-1]
    
    # Mise en forme des prédictions :
    faisceau = histoMod['Faisceau'][0] 
    mvt = histoMod['ArrDep'][0]
    
    PrevisionsLasso=[]
    for k in range(hPrev) :
        PrevisionsLasso.append([dateFinMod + timedelta(days=k+1) , faisceau, mvt, predictions[k]])
    
    PrevisionsLasso = pd.DataFrame(data=PrevisionsLasso,columns = ["Date", "Faisceau", "ArrDep", "PAX_Lasso"])
    PrevisionsLasso["IC"+str(int(ic*100))+"_low_LASSO"] = min_pred
    PrevisionsLasso["IC"+str(int(ic*100))+"_up_LASSO"] = max_pred
    
    # print((faisceau,mvt,modele.alpha_))
    # print("Score entrainement : "+str(modele.score(X_mod,y_mod)))

    return PrevisionsLasso

    
    
    


#### TEST ####

# dateDebMod = pd.to_datetime("2008-01-01")
# dateFinMod = pd.to_datetime("2016-01-15")

# hPrev = 7
    
# database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
# database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
# database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

# Calendrier = pd.read_csv("Calendrier.csv", dayfirst = True , sep = ';' , parse_dates = ['Date'])

# # Données d'entrainement : 
# histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
# histoMod_2 = histoMod[(histoMod['Faisceau']=='Schengen') & (histoMod['ArrDep']=='Arrivée')]
# # histoMod_2.to_csv('histoMod.csv')

# # Données de test : 
# histoPrev = database[(database['Date']>dateFinMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]
# histoPrev_2 = histoPrev[(histoPrev['Faisceau']=='Schengen')&(histoPrev['ArrDep']=='Arrivée')]
# # histoPrev_2.to_csv('histoPrev.csv')



# test = previsions_Lasso(histoMod_2, Calendrier, dateDebMod, dateFinMod, hPrev)







## TEST  2 : Ajout aux prédictions NP 

# dateDebMod = pd.to_datetime("2008-01-01")
# dateFinMod = pd.to_datetime("2015-12-31")
# horizonsPrev = [7] #, 31+29+31, 365]

# database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
# database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
# database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

# Calendrier = pd.read_csv("Calendrier.csv", dayfirst = True , sep = ';' , parse_dates = ['Date'])

# histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]

# for hPrev in horizonsPrev :
        
#         # on va ajouter les prévisions à l'historique précédent + sur la période de prévisions
#     histoPrev = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]
#             # ( sans historique précédent : histoPrev = database[(database['Date']>dateFinMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]   )
    
#     Prev_Lasso = pd.DataFrame()

    
#     for faisceau in ['National', 'Schengen', 'Autre UE', 'International', 'Dom Tom'] :
#         for mvt in ['Arrivée', 'Départ'] :
            
#             histoMod_2 = histoMod[(histoMod['Faisceau']==faisceau) & (histoMod['ArrDep']==mvt)]  
                
#                 # Modèle Lasso : 
#             prev_Lasso = previsions_Lasso (histoMod_2, Calendrier, dateDebMod, dateFinMod, hPrev)
#             Prev_Lasso = pd.concat([Prev_Lasso,prev_Lasso],ignore_index=True)
                    
                    
#     # Ajout des prévisions du modèle NP à histoPrev  

#     predNP = pd.read_csv("Previsions_"+str(hPrev)+"j.csv",parse_dates = ['Date'])
#     predNP.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','PAX_NP':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
#     histoPrev = pd.concat([predNP.set_index(['Date','Faisceau','ArrDep']),Prev_Lasso.set_index(['Date','Faisceau','ArrDep'])],axis=1)
#     histoPrev = histoPrev.reset_index()
    
#     histoPrev.to_csv("Previsions_"+str(hPrev)+"j.csv")
    