"""
Création de bases de références à partir des FQMS :
    
    - une base "base_ref_1" contenant le nombre moyens de sièges par type avion et compagnie
    - une base "base_ref_2" contenant le nombre moyens de sièges par type avion seulement

Dans les deux cas, on utilise la colonne "Sièges Corrections_ICI" pour le nombre de sièges
"""

import pandas as pd

fqms = pd.read_csv("fqms_concat.csv",low_memory=False)


base_ref_1 = fqms[['Tav','Cie','Sièges Corrections_ICI']]
base_ref_1 = base_ref_1.groupby(by=['Tav','Cie'],as_index=False).mean('Sièges Corrections_ICI')
base_ref_1.to_csv("base_ref_1.csv",encoding = 'utf-8')

base_ref_2 = fqms[['Tav','Sièges Corrections_ICI']]
base_ref_2 = base_ref_2.groupby(by=['Tav'],as_index=False).mean('Sièges Corrections_ICI')
base_ref_2.to_csv("base_ref_2.csv",encoding = 'utf-8')