"""
Ajout du nombre de sièges (provenant des FQMs) aux données réelles (provenant des histos) 
"""

import pandas as pd
import numpy as np

database = pd.read_csv("DATABASE.csv",low_memory=False,decimal=',')
database = database[database['Terrain'].isin(['ORLY'])]

base_ref_2 = pd.read_csv("base_ref_2.csv")

#Suppression d'une colonne inutile : 
for base in [database,base_ref_2] :
    base.drop(['Unnamed: 0'], axis='columns', inplace=True)
    
#On renomme les colonnes des bases de référence pour pouvoir faire un merge :
base_ref_2.rename(columns={"Tav": 'TypAv'}, inplace=True)


"""
On utilise la base de référence base_ref_2 
(car les compagnies ne correspondent pas pour utiliser base_ref_1) :
    - on fait correspondre le type avion "Tav" avec la colonne 'TypAv' de la base 
"""

database = database.merge(base_ref_2,how='left', on=['TypAv'])



"""
On peut ajouter la création d'une estiamtion du taux de remplissage des vols 'Coeff_Rempl'
"""

database['Coeff_Rempl'] = database['PAX']/database['Sièges Corrections_ICI']


"""
Ensuite, on exporte la base complétée
"""

if __name__ == '__main__':
    valeurs_tot = database['NumVol'].count()
    valeurs_renseignées = database['Sièges Corrections_ICI'].count()

    print("Nombre de valeurs de sièges renseignés :",valeurs_renseignées,"(sur ",valeurs_tot," soit ",valeurs_renseignées*100/valeurs_tot,"% )")
    # --> Nombre de valeurs de sièges renseignés : 2 034 834 (sur  2 074 533  soit  environ 98.09 % )
    
    print("Nombre de types avions non référencés :",database['TypAv'].drop_duplicates().count())
    # --> 192 sous-types avions n'ont pas d'estimations du nombre de sièges dispos...


database.to_csv("database_sieges.csv",encoding = 'utf-8')


#Création d'une base pour 2016 uniquement :
    
database = database.astype({'Date': 'datetime64[ns]'})

database_2016 = database[database['Date']>np.datetime64('2015-12-31')]
database_2016 = database_2016[database_2016['Date']<np.datetime64('2017-01-01')]

database_2016.to_csv("database_2016.csv",encoding = 'utf-8')


    