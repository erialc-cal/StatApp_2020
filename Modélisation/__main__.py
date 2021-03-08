"""
Le code suivant réalise des prédictions selon les différents modèles implémentés,
pour chaque faisceau et type de mouvement 

Pour l'utiliser, modifier :
    # - les chemins des dossiers contenant : les donnees, les modeles et où ranger les previsions finales
    - les dates de l'historique
    - le nombre de prévisions souhaitées
    
Le code crée alors un dataFrame qui contient les prédictions de nombre de passagers, avec le réalisé et les prédictions des FQMs
"""

import pandas as pd
from datetime import timedelta

from Modele_NP_final import previsions_NP


dateDebMod = pd.to_datetime("2007-01-01")
dateFinMod = pd.to_datetime("2016-01-15")

horizonsPrev = [7,30]
            


if __name__ == '__main__':
    
    database = pd.read_csv("database_sieges.csv",low_memory=False,decimal=',')
    database = database.astype({'Date': 'datetime64[ns]','PAX_FQM':'float','Sièges Corrections_ICI':'float','Coeff_Rempl':'float','Coeff_Rempl_FQM':'float'})
    database = database.groupby(['Date','Faisceau','ArrDep']).agg({'PAX':'sum','PAX_FQM':'sum','Sièges Corrections_ICI':'sum','Coeff_Rempl':'mean','Coeff_Rempl_FQM':'mean'}).reset_index()

    Calendrier = pd.read_csv("Calendrier.csv", dayfirst = True , sep = ';' , parse_dates = ['Date'])

    histoMod = database[(database['Date']>=dateDebMod) & (database['Date']<=dateFinMod)]
    
    for hPrev in horizonsPrev :
        
        histoPrev = database[(database['Date']>dateFinMod) & (database['Date']<=dateFinMod+timedelta(days = hPrev))]
    
        Prev_NP = pd.DataFrame()
    
        for faisceau in ['National', 'Schengen', 'Autre UE', 'International', 'Dom Tom'] :
            for mvt in ['Arrivée', 'Départ'] :
            
                histoMod_2 = histoMod[(histoMod['Faisceau']==faisceau) & (histoMod['ArrDep']==mvt)]
           
                # Modèle Non-Paramétrique :

                prev_NP = previsions_NP(histoMod_2, Calendrier, dateDebMod, dateFinMod, hPrev)
                Prev_NP = pd.concat([Prev_NP, prev_NP],ignore_index=True)

            
        # Ajout des prévisions du modèle NP à histoPrev           
        histoPrev = pd.concat([histoPrev.set_index(['Date','Faisceau','ArrDep']),Prev_NP.set_index(['Date','Faisceau','ArrDep'])],axis=1)
        histoPrev = histoPrev.reset_index()

        histoPrev.to_csv("Previsions_"+str(hPrev)+"j.csv")

