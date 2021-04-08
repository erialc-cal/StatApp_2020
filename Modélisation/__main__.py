"""
Le code suivant réalise des prédictions selon les différents modèles implémentés,
pour chaque faisceau et type de mouvement 

Pour l'utiliser, modifier :
    # - les chemins des dossiers contenant : les donnees, les modeles et où ranger les previsions finales
    - les dates de l'historique
    - le nombre de prévisions souhaitées
    - le seuil de l'intervalle de confiance souhaité
    
Le code crée alors un dataFrame qui contient les prédictions de nombre de passagers, 
avec le réalisé et les prédictions des FQMs.
"""


import pandas as pd
from datetime import timedelta


from Modele_ARIMA_final import previsions_ARIMA
from Modele_SARIMA_final import previsions_SARIMA
from Modele_NP_final import previsions_NP
from Modele_Lasso import previsions_Lasso



dateDebMod = pd.to_datetime("2008-01-01")
dateFinMod = pd.to_datetime("2015-12-31")


horizonsPrev = [365] # 7, 31+29+31, 365]  # (en jours)
ic = 0.95   # Seuil de l'intervalle de confiance souhaité
            


if __name__ == '__main__':
    
    database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
    database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
    database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

    Calendrier = pd.read_csv("Calendrier.csv", dayfirst = True , sep = ';' , parse_dates = ['Date'])

    histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
    # histoMod.to_csv("HistoMod.csv")
    
    
    for hPrev in horizonsPrev :
        
        # on va ajouter les prévisions à l'historique précédent + sur la période de prévisions
        histoPrev = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]
            # ( sans historique précédent : histoPrev = database[(database['Date']>dateFinMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]   )
    
        Prev_Arima = pd.DataFrame()
        Prev_Sarima = pd.DataFrame()
        Prev_NP = pd.DataFrame()
        Prev_Lasso = pd.DataFrame()

    
        for faisceau in ['National', 'Schengen', 'Autre UE', 'International', 'Dom Tom'] :
            for mvt in ['Départ']: #, 'Arrivée']: 
            
                histoMod_2 = histoMod[(histoMod['Faisceau']==faisceau) & (histoMod['ArrDep']==mvt)]

                # Modèle Arima :
                prev_Arima = previsions_ARIMA(histoMod_2, dateDebMod, dateFinMod, hPrev, ic)
                Prev_Arima = pd.concat([Prev_Arima, prev_Arima],ignore_index=True) 


                # Modèle Sarima :
                if faisceau!='International' :
                    prev_Sarima = previsions_SARIMA(histoMod_2, dateDebMod, dateFinMod, hPrev, ic)
                    Prev_Sarima = pd.concat([Prev_Sarima, prev_Sarima],ignore_index=True) 
                
                
                # Modèle Non-Paramétrique :
                prev_NP = previsions_NP(histoMod_2, Calendrier, dateDebMod, dateFinMod, hPrev, ic)
                Prev_NP = pd.concat([Prev_NP, prev_NP],ignore_index=True)   

                
                # Modèle Lasso : 
                prev_Lasso = previsions_Lasso (histoMod_2, Calendrier, dateDebMod, dateFinMod, hPrev)
                Prev_Lasso = pd.concat([Prev_Lasso,prev_Lasso],ignore_index=True)
                    
                    
        # Ajout des prévisions des différents modèles à histoPrev           
        histoPrev = pd.concat([histoPrev.set_index(['Date','Faisceau','ArrDep']),
                               Prev_Arima.set_index(['Date','Faisceau','ArrDep']),
                               Prev_Sarima.set_index(['Date','Faisceau','ArrDep']),
                               Prev_NP.set_index(['Date','Faisceau','ArrDep']),
                               Prev_Lasso.set_index(['Date','Faisceau','ArrDep'])],axis=1)
        histoPrev = histoPrev.reset_index()

        histoPrev.to_csv("Previsions_"+str(hPrev)+"j.csv")

