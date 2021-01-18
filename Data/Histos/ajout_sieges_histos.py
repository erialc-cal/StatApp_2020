"""
Ajout du nombre de sièges estimés (provenant des FQMs) aux données réelles (provenant des histos) 
"""

import pandas as pd

histos = pd.read_csv("historique_concat.csv")
fqms = pd.read_csv("fqms_concat.csv", low_memory = False)

base_ref_1 = pd.read_csv("base_ref_1.csv")
base_ref_2 = pd.read_csv("base_ref_2.csv")

#Suppression d'une colonne inutile : 
for base in [histos,fqms,base_ref_1,base_ref_2] :
    base.drop(['Unnamed: 0'], axis='columns', inplace=True)
    
#On renomme les colonnes des bases de référence pour pouvoir faire un merge :
base_ref_1.rename(columns={"Cie": "Code IATA compagnie", "Tav": "Type avion"}, inplace=True)
base_ref_2.rename(columns={"Tav": "Type avion"}, inplace=True)


"""
On commence par utiliser la première base de référence base_ref_1 :
    - on fait correspondre le type avion "Tav" avec la colonne "Type avion" des histos, 
    - on fait correspondre la compagnie "Cie" avec la colonne "Code IATA compagnie" des histos
"""

# histos.count()  
# --> il y a 46 valeurs manquantes de la colonne "Compagnie" et 25 681 valeurs manquantes de "Code IATA Compagnie"

# histos_2 = histos[['Compagnie','Code IATA compagnie']]
# print(histos_2[histos_2.isnull().any(axis=1)].drop_duplicates().count())
# --> Il y a 694 compagnies qui ne se voient pas attribuer de Code IATA, et donc sur lesquelles il faudra utiliser base_ref_2


histos = histos.merge(base_ref_1,how='left', on=["Type avion","Code IATA compagnie"])
# On a pu complété que 65 582 valeurs...


"""
Ensuite, on isole la partie de la base histos qui n'a pas été complétée lors du premier merge.

On fait un second merge sur la partie incomplète, avec base_ref_2
"""

histos_complet_1 = histos[~histos['Sièges Corrections_ICI'].isnull()]
histos_incomplet_1 = histos[histos['Sièges Corrections_ICI'].isnull()]

histos_incomplet_1.drop(['Sièges Corrections_ICI'], axis='columns', inplace=True)
histos_incomplet_1 = histos_incomplet_1.merge(base_ref_2,how='left', on=["Type avion"])

histos_complet_2 = histos_incomplet_1[~histos_incomplet_1['Sièges Corrections_ICI'].isnull()]
histos_incomplet_2 = histos_incomplet_1[histos_incomplet_1['Sièges Corrections_ICI'].isnull()]


"""
Enfin, on "recolle" les morceaux : on joint histos_complets_1 et histos_complets_2, 
auxquels on rajoute la partie des histos que l'on a pas pu compléter histos_incomplet_2
"""

histos = pd.concat([histos_complet_1, histos_complet_2, histos_incomplet_2])

histos.to_csv("histos_sieges.csv",encoding = 'utf-8')

    


