import pandas as pd

"""
!!! A runner dans le dossier contenant les FQms

Concaténation des fichiers FQMS :
    
On a les fichiers : fqm_[1-53]_2015, fqm_[1-52]_2016 et fqm_[1-41]_2017
On a donc les estimations depuis le 01/01/2015 jusqu'au 22/10/2017

Fichiers modifiés avant la concaténation : 
    - fqm_13_2015 : tableau supplémentaire à supprimer en bas de l'onglet FQM_Definitive

Fichiers enlevés de la concaténation pour cause d'erreur/valeurs manquantes : 
    - fqm_35_2017 : pas d'estimations ... et d'onglet FQM_Definitive
    - fqm_12_2015 : pas de colonne LF%
    - fqm_15_2017 et fqm_34_2017 : fichiers incomplets, avec dates équivalentes étranges (00/01/1900)  
            --> fqm_15_2017 : 2365 sur 4748 valeurs de LF% et Pax Estimés sont manquantes
            --> fqm_34_2017 : 3350 sur 4488 valeurs de Pax Estimés ont été mises à 0
"""

annees = ["2015","2016","2017"]
fichiers = [53,52,41]
exclus = [("2017",35),("2015",12),("2017",15),("2017",34)]

df=[]

for j,annee in enumerate(annees) :
    for k in range(1,fichiers[j]+1) :
        if not((annee,k) in exclus) :
            xls = pd.ExcelFile("fqm_"+str(k)+"_"+annee+".xls") 
            fqm = pd.read_excel(xls, 'FQM_Definitive')
            df.append(fqm)
        
donnees_fqm = pd.concat(df, join = 'inner')

#Réinitialisation de l'index :
donnees_fqm = donnees_fqm.reset_index()

#Suppression des colonnes inutiles : 
donnees_fqm.drop(['Courbe','index'],axis=1, inplace=True) 

donnees_fqm.to_csv("fqms_concat.csv",encoding = 'utf-8')
