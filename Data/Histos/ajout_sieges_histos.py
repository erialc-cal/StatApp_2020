"""
Ajout du nombre de sièges (provenant des FQMs) aux données réelles (provenant des histos) 
"""

import pandas as pd
import numpy as np

histos = pd.read_csv("historique_concat.csv")
histos = histos[histos['Plateforme'].isin(['ORLY'])]

fqms = pd.read_csv("fqms_concat.csv", low_memory = False)

base_ref_1 = pd.read_csv("base_ref_1.csv")
base_ref_2 = pd.read_csv("base_ref_2.csv")

#Suppression d'une colonne inutile : 
for base in [histos,fqms,base_ref_1,base_ref_2] :
    base.drop(['Unnamed: 0'], axis='columns', inplace=True)
    
#On renomme les colonnes des bases de référence pour pouvoir faire un merge :
base_ref_1.rename(columns={"Cie": "Code IATA compagnie", "Tav": "Sous-type avion"}, inplace=True)
base_ref_2.rename(columns={"Tav": "Sous-type avion"}, inplace=True)


"""
On commence par utiliser la première base de référence base_ref_1 :
    - on fait correspondre le type avion "Tav" avec la colonne "Sous-type avion" des histos, 
    - on fait correspondre la compagnie "Cie" avec la colonne "Code IATA compagnie" des histos
"""

# histos.count()  
# --> il y a 46 valeurs manquantes de la colonne "Compagnie" et 25 681 valeurs manquantes de "Code IATA Compagnie"

# histos_2 = histos[['Compagnie','Code IATA compagnie']]
# print(histos_2[histos_2.isnull().any(axis=1)].drop_duplicates().count())
# --> Il y a 694 compagnies qui ne se voient pas attribuer de Code IATA, et donc sur lesquelles il faudra utiliser base_ref_2


histos = histos.merge(base_ref_1,how='left', on=["Sous-type avion","Code IATA compagnie"])
# On a pu complété que 65 582 valeurs...


"""
Ensuite, on isole la partie de la base histos qui n'a pas été complétée lors du premier merge.

On fait un second merge sur la partie incomplète, avec base_ref_2
"""

histos_complet_1 = histos[~histos['Sièges Corrections_ICI'].isnull()]
histos_incomplet_1 = histos[histos['Sièges Corrections_ICI'].isnull()]

histos_incomplet_1.drop(['Sièges Corrections_ICI'], axis='columns', inplace=True)
histos_incomplet_1 = histos_incomplet_1.merge(base_ref_2,how='left', on=["Sous-type avion"])

histos_complet_2 = histos_incomplet_1[~histos_incomplet_1['Sièges Corrections_ICI'].isnull()]
histos_incomplet_2 = histos_incomplet_1[histos_incomplet_1['Sièges Corrections_ICI'].isnull()]


"""
Enfin, on "recolle" les morceaux : on joint histos_complets_1 et histos_complets_2, 
auxquels on rajoute la partie des histos que l'on a pas pu compléter histos_incomplet_2
"""

histos = pd.concat([histos_complet_1, histos_complet_2, histos_incomplet_2])


if __name__ == '__main__':
    valeurs_tot = histos['Numéro de vol'].count()
    valeurs_renseignées = histos['Sièges Corrections_ICI'].count()

    print("Nombre de valeurs de sièges renseignés :",valeurs_renseignées,"(sur ",valeurs_tot," soit ",valeurs_renseignées*100/valeurs_tot,"% )")
    # --> Nombre de valeurs de sièges renseignés : 1 616 539 (sur  1 634 166  soit  environ 98.92 %)
    
    print("Nombre de types avions non référencés :",histos_incomplet_2['Sous-type avion'].drop_duplicates().count())
    # --> 129 sous-types avions n'ont pas d'estimations du nombre de sièges dispos...


histos.to_csv("histos_sieges.csv",encoding = 'utf-8')


#Création d'une base pour 2016 uniquement :
    
histos = histos.astype({'Jour': 'datetime64[ns]'})

histos_2016 = histos[histos['Jour']>np.datetime64('2015-12-31')]
histos_2016 = histos_2016[histos_2016['Jour']<np.datetime64('2017-01-01')]

histos_2016.to_csv("histo_2016.csv",encoding = 'utf-8')

    


