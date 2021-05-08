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
from tqdm import trange


def prediction_interval(model, X_train, y_train, x0, alpha):
  ''' Compute a prediction interval around the model's prediction of x0.

  INPUT
    model
      A predictive model with `fit` and `predict` methods
    X_train: numpy array of shape (n_samples, n_features)
      A numpy array containing the training input data
    y_train: numpy array of shape (n_samples,)
      A numpy array containing the training target data
    x0 : A new data point, of shape (n_features,)
    alpha: 1-ic
      The prediction uncertainty

  OUTPUT
    A triple (`lower`, `pred`, `upper`) with `pred` being the prediction
    of the model and `lower` and `upper` constituting the lower- and upper
    bounds for the prediction interval around `pred`, respectively. '''

  # Number of training samples
  n = X_train.shape[0]

  # We choose to make 1000 bootstraps
  nbootstraps=1000

  # Compute the m_i's and the validation residuals
  bootstrap_preds, val_residuals = pd.DataFrame(), []
  for b in trange(nbootstraps):
    train_idxs = np.random.choice(range(n), size = n, replace = True)
    val_idxs = np.array([idx for idx in range(n) if idx not in train_idxs])
    model.fit(X_train[train_idxs, :], y_train[train_idxs])
    preds = model.predict(X_train[val_idxs])
    val_residuals.append(y_train[val_idxs] - preds)
    

    bootstrap_preds[b] = model.predict(x0)
  bootstrap_preds -= np.mean(bootstrap_preds)
  val_residuals = np.concatenate(val_residuals)

  # Compute the prediction and the training residuals
  model.fit(X_train, y_train)
  preds = model.predict(X_train)
  train_residuals = y_train - preds

  # Take percentiles of the training- and validation residuals to enable
  # comparisons between them
  val_residuals = np.percentile(val_residuals, q = np.arange(100))
  train_residuals = np.percentile(train_residuals, q = np.arange(100))

  # Compute the .632+ bootstrap estimate for the sample noise and bias
  no_information_error = np.mean(np.abs(np.random.permutation(y_train) - \
    np.random.permutation(preds)))
      
  generalisation = np.abs(val_residuals.mean() - train_residuals.mean())
  no_information_val = np.abs(no_information_error - train_residuals)
  relative_overfitting_rate = np.mean(generalisation / no_information_val)
  weight = .632 / (1 - .368 * relative_overfitting_rate)
  residuals = (1 - weight) * train_residuals + weight * val_residuals

  # Construct the C set and get the percentiles
  C = np.array([m + o for m in bootstrap_preds for o in residuals])
  qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]
  percentiles = np.percentile(C, q = qs)

  return percentiles[0], model.predict(x0), percentiles[1]
    




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
    
    if ic != 0 :
        intervals = prediction_interval(modele, X_mod, y_mod, X_pred, 1-ic)

    
    # Mise en forme des prédictions :
    faisceau = histoMod['Faisceau'][0] 
    mvt = histoMod['ArrDep'][0]
    
    PrevisionsLasso=[]
    for k in range(hPrev) :
        PrevisionsLasso.append([dateFinMod + timedelta(days=k+1) , faisceau, mvt, predictions[k]])
    
    PrevisionsLasso = pd.DataFrame(data=PrevisionsLasso,columns = ["Date", "Faisceau", "ArrDep", "PAX_Lasso"])
    
    if ic == 0 :
        PrevisionsLasso["IC"+str(int(ic*100))+"_low_LASSO"] = [0 for k in range(hPrev)]
        PrevisionsLasso["IC"+str(int(ic*100))+"_up_LASSO"] = [0 for k in range(hPrev)]
    else : 
        PrevisionsLasso["IC"+str(int(ic*100))+"_low_LASSO"] = PrevisionsLasso["PAX_Lasso"]+intervals[0]
        PrevisionsLasso["IC"+str(int(ic*100))+"_up_LASSO"] = PrevisionsLasso["PAX_Lasso"]+ intervals[2]
    
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







## TEST  2 : Prévisions Lasso sur tous les faisceaux, hPrev = 7, mvt = Départ :

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
#         for mvt in ['Départ'] : #,"Arrivée"] :
            
#             histoMod_2 = histoMod[(histoMod['Faisceau']==faisceau) & (histoMod['ArrDep']==mvt)]  
                
#                 # Modèle Lasso : 
#             prev_Lasso = previsions_Lasso (histoMod_2, Calendrier, dateDebMod, dateFinMod, hPrev)
#             Prev_Lasso = pd.concat([Prev_Lasso,prev_Lasso],ignore_index=True)
                    
# Prev_Lasso.to_csv("Prev_Lasso_IC.csv")
 
             
