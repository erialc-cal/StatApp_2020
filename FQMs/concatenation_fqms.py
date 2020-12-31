import pandas as pd

"""
!!! A runner dans le dossier contenant les FQms

Concaténation des fichiers FQMS :
    
On a les fichiers : fqm_[1-53]_2015, fqm_[1-52]_2016 et fqm_[1-41]_2017
On a donc les estimations depuis le 01/01/2015 jusqu'au 22/10/2017

Fichiers enlevés pour cause d'erreur : 
    - fqm_35_2017 : pas d'estimations ... et d'onglet FQM_Definitive
    - fqm_12_2015 : pas de colonne LF%
    
Autres problèmes :    
    - fqm_13_2015 : réglé, il y avait un tableau supplémentaire à supprimer en bas du tableau
    - fqm_15_2017 : il manque 2365 valeurs de LF% et Pax Estimés 
    - fqm_15_2017 et fqm_34_2017 : dates équivalentes étranges (00/01/1900)
"""

annees = ["2015","2016","2017"]
fichiers = [53,52,41]

df=[]

for j,annee in enumerate(annees) :
    for k in range(1,fichiers[j]+1) :
        if not(annee == "2017" and k == 35) and not(annee == "2015" and k == 12) :
            xls = pd.ExcelFile("fqm_"+str(k)+"_"+annee+".xls") 
            fqm = pd.read_excel(xls, 'FQM_Definitive')
            df.append(fqm)
        
donnees_fqm = pd.concat(df, join = 'inner')

#Réinitialisation de l'index :
donnees_fqm = donnees_fqm.reset_index()

#Suppression des colonnes inutiles : 
donnees_fqm.drop(['Courbe','index'],axis=1, inplace=True) 

donnees_fqm.to_csv("fqms_concat.csv",encoding = 'utf-8')