# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math as math
from datetime import timedelta

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)



TRAFIC = pd.DataFrame()
for An in range(2007 , 2022):   
    print(An)     
    TRAFIC_Temp  = pd.read_csv("C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\Histo Trafic\\Histo Trafic " + str(An) + ".csv"  , dayfirst = True , sep = ';' , parse_dates = ['Jour' , 'Horaire théorique' , 'Horaire bloc'])
    TRAFIC_Temp = TRAFIC_Temp.rename(columns = {'Numéro de vol' : 'NumVol' , 'Compagnie' : 'Cie' , 'Code OACI compagnie' : 'Code Cie' , 'Aéroport' : 'Aerop' , 'Aérogare' : 'Aerog' , 'Code aéroport OACI' : 'Code Aerop' , 'Faisceau facturation' : 'Faisceau' , 'Type de mouvement' : 'ArrDep' , 'Sous-type avion' : 'TypAv' , 'Plateforme' : 'Terrain' , 'Jour' : 'Date' , 'Horaire théorique' : 'Heure Théorique' , 'Horaire bloc' : 'Heure Bloc' , 'Nombre de mouvements réalisés' : 'MVT' , 'Nombre de passagers réalisés' : 'PAX'}) 
    TRAFIC = pd.concat([TRAFIC , TRAFIC_Temp])
    
print(TRAFIC.columns)
TRAFIC["Test_Compare"] = TRAFIC['NumVol'].str[0:3].str.strip().str.lower() == TRAFIC['Code Cie'].str.strip().str.lower()
TRAFIC['NumVol']       = (TRAFIC['Code Cie'] + np.where((TRAFIC["Test_Compare"] == True) , TRAFIC['NumVol'].str[3:].str.lstrip('0') , TRAFIC['NumVol'].str[2:].str.lstrip('0')))


print(TRAFIC.columns)

TRAFIC = TRAFIC[['Date' , 'Faisceau', 'Pays', 'Ville' , 'Terrain' , 'Terminal' , 'Aerog' , 'Aerop' , 'Code Aerop' , 'Cie' , 'Code Cie' , 'ArrDep' , 'NumVol' , 'TypAv' , 'MVT' , 'PAX']]
print(TRAFIC)

DATABASE_NP = TRAFIC[(TRAFIC['ArrDep'] == 'Départ') & (TRAFIC['Faisceau'] == 'National') & (TRAFIC['Aerog'] == 'ORYO')]
DATABASE_NP = DATABASE_NP.groupby(['ArrDep' , 'Faisceau' , 'Aerog' , 'Date'] , sort = True , observed = True).agg({"PAX" : lambda x: x.sum(skipna = False) , "MVT" : lambda x: x.sum(skipna = False)}).reset_index()       
DATABASE_NP['EMPORT'] = DATABASE_NP['PAX'] / DATABASE_NP['MVT']







FQMS = pd.DataFrame()
for An in range (2015 , 2017):
    for Semaine in range ( 1 , 54):
        if (An == 2015 and Semaine == 53) or (An == 2016 and Semaine <= 52) or (An == 2017 and (Semaine <= 41 and (Semaine != 10 and Semaine != 13 and Semaine != 15 and Semaine != 35 and Semaine <= 41))):
            print(An , Semaine)
            FQM_Temp = pd.read_excel('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\FQMS\\fqm_' + str(Semaine) + '_' + str(An) + '.xls' , parse_dates = True, sheet_name  = 'FQM_Definitive') 
            FQM_Temp = FQM_Temp.drop(columns = ['Faisceau'])
            FQM_Temp = FQM_Temp.rename(columns = {'Date du vol' : 'Date' , 'Qualité' : 'Faisceau' , 'Prov/Dst' : 'Ville' , 'Aérog' : 'Aerog' , 'Tav' : 'TypAv' , 'A/D' : 'ArrDep' , 'LF%' : 'Coeff Rempl' , 'Pax Estimés' : 'PAX' , 'Cpt Mvt' : 'MVT' , 'Sièges Corrections_ICI' : 'Sieges'})
            FQMS = pd.concat([FQMS , FQM_Temp])

print(FQMS.columns)

FQMS["Date"] = pd.to_datetime(FQMS['Date'].dt.date)

FQMS = FQMS[['Date' , 'Hor' , 'Date Equiv' , 'Faisceau', 'Pays', 'Ville' , 'Aerog' , 'Cie' , 'ArrDep' , 'Ligne' , 'TypAv' , 'MVT' , 'PAX' , 'Sieges' , 'Coeff Rempl']]

FQMS['ArrDep'] = np.where(FQMS['ArrDep'] == "A" , "Arrivée" , "Départ")
FQMS['Faisceau'] = np.where(FQMS['Faisceau'] == "MTP" , "National" , np.where(FQMS['Faisceau'] == "SCH" , "Schengen" , np.where(FQMS['Faisceau'] == "DTM" , "Dom Tom" , np.where(FQMS['Faisceau'] == "CEE" , "Autre UE" , np.where(FQMS['Faisceau'] == "INT" , "International" , FQMS['Faisceau'])))))
FQMS['Aerog'] = np.where(FQMS['Aerog'] == "OS" , "ORYS" , np.where(FQMS['Aerog'] == "OW" , "ORYO" , FQMS['Aerog']))

print(FQMS.drop_duplicates(subset=['Aerog']))
print(TRAFIC.drop_duplicates(subset=['Faisceau']))


DATABASE_FQMS = FQMS[(FQMS['ArrDep'] == 'Départ') & (FQMS['Faisceau'] == 'National') & (FQMS['Aerog'] == 'ORYO')]
DATABASE_FQMS = DATABASE_FQMS.groupby(['ArrDep' , 'Faisceau' , 'Aerog' , 'Date'] , sort = True , observed = True).agg({"PAX" : lambda x: x.sum(skipna = False) , "MVT" : lambda x: x.sum(skipna = False) , "Sieges" : lambda x: x.sum(skipna = False)}).reset_index()       


print(DATABASE_FQMS)
print(FQMS)


Calendrier = pd.read_csv("C:\\Users\\boudra\\Desktop\\Projet Previs\\Calendrier\\Calendrier.csv"  , dayfirst = True , sep = ';' , parse_dates = ['Date'])
Calendrier = Calendrier[["Date" , "Mois" , "JourSem" , "Semaine" , "Semaine_Perso" , "Pont_LunF" , "Pont_MarF" , "Pont_Mer1F" , "Pont_Mer2F" , "Pont_JeuF" , "Pont_VenF" , "Pont_SamF" , "Pont_DimF" , "Vac_Toussaint" , "Vac_Noel" , "Vac_Hiver_A" , "Vac_Hiver_B" , "Vac_Hiver_C" , "Vac_Printemps_A" , "Vac_Printemps_B" , "Vac_Printemps_C" , "Vac_Ete"]]
print(Calendrier)










DtDebMod = pd.to_datetime("2007-01-01")
DtFinMod = pd.to_datetime("2016-10-23")


HPrev = 2
Taille_Des_Blocs = 365
#NbEchVC = 40
#Grille = np.array([i for i in np.arange(0.1,10.1,0.1)]).reshape(-1,1)
h = 20



HistoMod = DATABASE_NP[(DATABASE_NP['Date'] >= DtDebMod) & (DATABASE_NP['Date'] <= DtFinMod)]
HistoMod.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\Histo.csv' , sep = ';' , decimal = ',')
print(HistoMod)


HistoMod = pd.merge(HistoMod , Calendrier , left_on = ["Date"] , right_on = ["Date"] , how='left')





as_strided = np.lib.stride_tricks.as_strided  
Blocs = pd.DataFrame(as_strided(HistoMod["EMPORT"], (len(HistoMod["EMPORT"]) - (Taille_Des_Blocs - 1), Taille_Des_Blocs), (HistoMod["EMPORT"].values.strides * 2)))
Blocs.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\Blocs.csv' , sep = ';' , decimal = ',')




Stats = pd.DataFrame() 
Stats['Mean'] = Blocs.mean(axis = 1)
Stats['Stds'] = Blocs.std(axis = 1)
Stats.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\Stats.csv' , sep = ';' , decimal = ',')


        
Blocs_CR = (Blocs - np.array(Stats['Mean']).reshape(-1,1)) / np.array(Stats['Stds']).reshape(-1,1)
Blocs_CR.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\Blocs_CR.csv' , sep = ';' , decimal = ',')


LastBloc_CR  = pd.DataFrame(np.array(Blocs_CR)[len(Blocs_CR) - 1].reshape(1,-1))
LastBloc_CR.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\LastBloc_CR.csv' , sep = ';' , decimal = ',')
 
print(np.array(Stats['Mean'])[len(Stats['Mean']) - 1].reshape(1,1))
StatsLastBloc = pd.DataFrame(np.array(Stats)[len(Stats['Mean']) - 1].reshape(1,-1))
StatsLastBloc = StatsLastBloc.rename(columns = {0 : 'Mean' , 1 : 'Std'})

StatsLastBloc.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\StatsLastBloc.csv' , sep = ';' , decimal = ',')
 
print(StatsLastBloc)


Weights = pd.DataFrame((1 / (math.sqrt(2 * math.pi) ** (Taille_Des_Blocs))) * np.exp(-(1 / (2 * h)) * np.sum((np.array(LastBloc_CR) - np.array(Blocs_CR[:len(Blocs_CR)])) ** 2 , axis = 1)))
Weights.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\Weights.csv' , sep = ';' , decimal = ',')
print(np.array(LastBloc_CR) - np.array(Blocs_CR[:len(Blocs_CR)]))

print(Calendrier)
      
TsPrevisions = pd.DataFrame()
for HorizonPrev in range(1,HPrev):
    DatePrev = DtFinMod + timedelta(days = HorizonPrev)
    print(HorizonPrev)

    CalPrevtemp = np.array(Calendrier)[[Calendrier["Date"] == DatePrev]]
    CalPrevtemp[:,5] = np.where(CalPrevtemp[:,5] > 0 and (CalPrevtemp[:,13] > 0 or CalPrevtemp[:,14] > 0 or CalPrevtemp[:,15] > 0 or CalPrevtemp[:,16] > 0 or CalPrevtemp[:,17] > 0 or CalPrevtemp[:,18] or CalPrevtemp[:,19] > 0 or CalPrevtemp[:,20] > 0 or CalPrevtemp[:,21] > 0), 0 , CalPrevtemp[:,5])
    CalPrevtemp[:,6] = np.where(CalPrevtemp[:,6] > 0 and (CalPrevtemp[:,13] > 0 or CalPrevtemp[:,14] > 0 or CalPrevtemp[:,15] > 0 or CalPrevtemp[:,16] > 0 or CalPrevtemp[:,17] > 0 or CalPrevtemp[:,18] or CalPrevtemp[:,19] > 0 or CalPrevtemp[:,20] > 0 or CalPrevtemp[:,21] > 0), 0 , CalPrevtemp[:,6])
    CalPrevtemp[:,7] = np.where(CalPrevtemp[:,7] > 0 and (CalPrevtemp[:,13] > 0 or CalPrevtemp[:,14] > 0 or CalPrevtemp[:,15] > 0 or CalPrevtemp[:,16] > 0 or CalPrevtemp[:,17] > 0 or CalPrevtemp[:,18] or CalPrevtemp[:,19] > 0 or CalPrevtemp[:,20] > 0 or CalPrevtemp[:,21] > 0), 0 , CalPrevtemp[:,7])
    CalPrevtemp[:,8] = np.where(CalPrevtemp[:,8] > 0 and (CalPrevtemp[:,13] > 0 or CalPrevtemp[:,14] > 0 or CalPrevtemp[:,15] > 0 or CalPrevtemp[:,16] > 0 or CalPrevtemp[:,17] > 0 or CalPrevtemp[:,18] or CalPrevtemp[:,19] > 0 or CalPrevtemp[:,20] > 0 or CalPrevtemp[:,21] > 0), 0 , CalPrevtemp[:,8])
    CalPrevtemp[:,9] = np.where(CalPrevtemp[:,9] > 0 and (CalPrevtemp[:,13] > 0 or CalPrevtemp[:,14] > 0 or CalPrevtemp[:,15] > 0 or CalPrevtemp[:,16] > 0 or CalPrevtemp[:,17] > 0 or CalPrevtemp[:,18] or CalPrevtemp[:,19] > 0 or CalPrevtemp[:,20] > 0 or CalPrevtemp[:,21] > 0), 0 , CalPrevtemp[:,9])
    CalPrevtemp[:,10] = np.where(CalPrevtemp[:,10] > 0 and (CalPrevtemp[:,13] > 0 or CalPrevtemp[:,14] > 0 or CalPrevtemp[:,15] > 0 or CalPrevtemp[:,16] > 0 or CalPrevtemp[:,17] > 0 or CalPrevtemp[:,18] or CalPrevtemp[:,19] > 0 or CalPrevtemp[:,20] > 0 or CalPrevtemp[:,21] > 0), 0 , CalPrevtemp[:,10])
   
    
    
    print(CalPrevtemp)
    
    print(Calendrier)
    
    JourPrev         = CalPrevtemp[:,2]
    Pont_LunF        = CalPrevtemp[:,5]
    Pont_MarF        = CalPrevtemp[:,6]
    Pont_Mer1F       = CalPrevtemp[:,7]
    Pont_Mer2F       = CalPrevtemp[:,8]
    Pont_JeuF        = CalPrevtemp[:,9]
    Pont_VenF        = CalPrevtemp[:,10]
    Vac_Toussaint    = CalPrevtemp[:,13]
    Vac_Noel         = CalPrevtemp[:,14]
    Vac_Hiver_A      = CalPrevtemp[:,15]
    Vac_Hiver_B      = CalPrevtemp[:,16]
    Vac_Hiver_C      = CalPrevtemp[:,17]
    Vac_Printemps_A  = CalPrevtemp[:,18]
    Vac_Printemps_B  = CalPrevtemp[:,19]
    Vac_Printemps_C  = CalPrevtemp[:,20]
    Vac_Ete          = CalPrevtemp[:,21]    
    
    
    
    
    HistoMod["Pont_LunF"] = np.where(HistoMod["Pont_LunF"] > 0 and (HistoMod["Vac_Toussaint"] > 0 or HistoMod["Vac_Noel"] > 0 or HistoMod["Vac_Hiver_A"] > 0 or HistoMod["Vac_Hiver_B"] > 0 or HistoMod["Vac_Hiver_C"] > 0 or HistoMod["Vac_Printemps_A"] > 0 or HistoMod["Vac_Printemps_B"] > 0 or HistoMod["Vac_Printemps_C"] > 0 or HistoMod["Vac_Ete"] > 0), 0 , HistoMod["Pont_LunF"])
    HistoMod["Pont_MarF"] = np.where(HistoMod["Pont_MarF"] > 0 and (HistoMod["Vac_Toussaint"] > 0 or HistoMod["Vac_Noel"] > 0 or HistoMod["Vac_Hiver_A"] > 0 or HistoMod["Vac_Hiver_B"] > 0 or HistoMod["Vac_Hiver_C"] > 0 or HistoMod["Vac_Printemps_A"] > 0 or HistoMod["Vac_Printemps_B"] > 0 or HistoMod["Vac_Printemps_C"] > 0 or HistoMod["Vac_Ete"] > 0), 0 , HistoMod["Pont_MarF"])
    HistoMod["Pont_Mer1F"] = np.where(HistoMod["Pont_Mer1F"] > 0 and (HistoMod["Vac_Toussaint"] > 0 or HistoMod["Vac_Noel"] > 0 or HistoMod["Vac_Hiver_A"] > 0 or HistoMod["Vac_Hiver_B"] > 0 or HistoMod["Vac_Hiver_C"] > 0 or HistoMod["Vac_Printemps_A"] > 0 or HistoMod["Vac_Printemps_B"] > 0 or HistoMod["Vac_Printemps_C"] > 0 or HistoMod["Vac_Ete"] > 0), 0 , HistoMod["Pont_Mer1F"])
    HistoMod["Pont_Mer2F"] = np.where(HistoMod["Pont_Mer2F"] > 0 and (HistoMod["Vac_Toussaint"] > 0 or HistoMod["Vac_Noel"] > 0 or HistoMod["Vac_Hiver_A"] > 0 or HistoMod["Vac_Hiver_B"] > 0 or HistoMod["Vac_Hiver_C"] > 0 or HistoMod["Vac_Printemps_A"] > 0 or HistoMod["Vac_Printemps_B"] > 0 or HistoMod["Vac_Printemps_C"] > 0 or HistoMod["Vac_Ete"] > 0), 0 , HistoMod["Pont_Mer2F"])
    HistoMod["Pont_JeuF"] = np.where(HistoMod["Pont_JeuF"] > 0 and (HistoMod["Vac_Toussaint"] > 0 or HistoMod["Vac_Noel"] > 0 or HistoMod["Vac_Hiver_A"] > 0 or HistoMod["Vac_Hiver_B"] > 0 or HistoMod["Vac_Hiver_C"] > 0 or HistoMod["Vac_Printemps_A"] > 0 or HistoMod["Vac_Printemps_B"] > 0 or HistoMod["Vac_Printemps_C"] > 0 or HistoMod["Vac_Ete"] > 0), 0 , HistoMod["Pont_JeuF"])
    HistoMod["Pont_VenF"] = np.where(HistoMod["Pont_VenF"] > 0 and (HistoMod["Vac_Toussaint"] > 0 or HistoMod["Vac_Noel"] > 0 or HistoMod["Vac_Hiver_A"] > 0 or HistoMod["Vac_Hiver_B"] > 0 or HistoMod["Vac_Hiver_C"] > 0 or HistoMod["Vac_Printemps_A"] > 0 or HistoMod["Vac_Printemps_B"] > 0 or HistoMod["Vac_Printemps_C"] > 0 or HistoMod["Vac_Ete"] > 0), 0 , HistoMod["Pont_VenF"])

    
    
    
    
    
    
    HistoPrev = HistoMod[Taille_Des_Blocs - 1 + HorizonPrev : ]
    IndJourPrev = np.where((np.array(HistoPrev["JourSem"]) == JourPrev) & (np.array(HistoPrev["Pont_LunF"]) == Pont_LunF) & (np.array(HistoPrev["Pont_MarF"]) == Pont_MarF) & (np.array(HistoPrev["Pont_Mer1F"]) == Pont_Mer1F) & (np.array(HistoPrev["Pont_Mer2F"]) == Pont_Mer2F) & (np.array(HistoPrev["Pont_JeuF"]) == Pont_JeuF) & (np.array(HistoPrev["Pont_VenF"]) == Pont_VenF) & (np.array(HistoPrev["Vac_Toussaint"]) == Vac_Toussaint) & (np.array(HistoPrev["Vac_Noel"]) == Vac_Noel) & (np.array(HistoPrev["Vac_Hiver_A"]) == Vac_Hiver_A) & (np.array(HistoPrev["Vac_Hiver_B"]) == Vac_Hiver_B) & (np.array(HistoPrev["Vac_Hiver_C"]) == Vac_Hiver_C) & (np.array(HistoPrev["Vac_Printemps_A"]) == Vac_Printemps_A) & (np.array(HistoPrev["Vac_Printemps_B"]) == Vac_Printemps_B) & (np.array(HistoPrev["Vac_Printemps_C"]) == Vac_Printemps_C)  == True , 1 , 0).reshape(-1,1)
     
    
    print(IndJourPrev)
    
    WeightsPrev = IndJourPrev * np.array(Weights)[:len(Weights) - HorizonPrev]
    #WeightsPrev = np.array(Weights)[:len(Weights) - HorizonPrev]
    WeightsPrev = pd.DataFrame(WeightsPrev)
    print(WeightsPrev)

    Sim = WeightsPrev / np.sum(WeightsPrev , axis = 0)
    
    pd.DataFrame(Sim).to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\Sim.csv' , sep = ';' , decimal = ',')
    
    
    UnePrevision = (np.sum(((((np.array(HistoMod["EMPORT"][Taille_Des_Blocs - 1 + HorizonPrev : ]).reshape(-1,1) - np.array(Stats['Mean'][ : len(Stats['Mean']) - HorizonPrev]).reshape(-1,1)) / np.array(Stats['Stds'][ : len(Stats['Stds']) - HorizonPrev]).reshape(-1,1)) * np.array(StatsLastBloc['Std']).reshape(-1,1)) + np.array(StatsLastBloc['Mean']).reshape(-1,1)) * Sim))
    
    UnePrevision = pd.DataFrame(UnePrevision).rename(columns = {0 : "Prévisions"})
    print("prevision" , UnePrevision)

    TsPrevisions = pd.concat([TsPrevisions , pd.concat([UnePrevision , pd.DataFrame([DatePrev]) , pd.DataFrame(HistoMod[["ArrDep" , "Faisceau"]]).head(1).reset_index().drop(columns = ['index'])] , axis = 1)])


print(TsPrevisions)


TsPrevisions = TsPrevisions.rename(columns = {0 : "Date"})
HistoMod = DATABASE_NP[(DATABASE_NP['Date'] >= DtDebMod) & (DATABASE_NP['Date'] <= DtFinMod + timedelta(days = HPrev - 1))]
TsPrevisions = pd.merge(HistoMod , TsPrevisions , on = ["ArrDep" , "Faisceau" , "Date"] , how='left')

print(TsPrevisions)

TsPrevisions['Prévisions'] = np.where(TsPrevisions['Date'] == DtFinMod , TsPrevisions['EMPORT'] , TsPrevisions['Prévisions'])

print(TsPrevisions)
print(DATABASE_FQMS)


DATABASE_FQMS = DATABASE_FQMS.rename(columns = {"PAX" : "Prevision_FQM" , "MVT" : "MVT_FQM" , "Sieges" : "Sieges_FQM"})

TsPrevisions = pd.merge(TsPrevisions , DATABASE_FQMS , on = ["ArrDep" , "Faisceau" , "Aerog" , "Date"] , how = 'left')

print(TsPrevisions)

TsPrevisions['PAX_NP'] = TsPrevisions['Prévisions'] * TsPrevisions['MVT_FQM']
TsPrevisions['Rempl'] = TsPrevisions['PAX'] / TsPrevisions['Sieges_FQM'] * 100
TsPrevisions['Rempl_NP'] = TsPrevisions['PAX_NP'] / TsPrevisions['Sieges_FQM'] * 100
TsPrevisions['Rempl_FQM'] = TsPrevisions['Prevision_FQM'] / TsPrevisions['Sieges_FQM'] * 100



TsPrevisions.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\TsPrevisions.csv' , sep = ';' , decimal = ',')


print(FQMS)
print(TsPrevisions)


DATABASE_Previs = TsPrevisions[(pd.isna(TsPrevisions['Prévisions']) == False) & (TsPrevisions['Date'] != DtFinMod)]
print(DATABASE_Previs)
DATABASE_Previs = DATABASE_Previs[["ArrDep" , "Faisceau" , "Aerog" , "Date" , "Rempl" , "Rempl_NP" , "Rempl_FQM"]]

FQMS_Result = pd.merge(FQMS , DATABASE_Previs , on = ["ArrDep" , "Faisceau" , "Aerog" , "Date"] , how = 'right')
FQMS_Result['jour'] = FQMS_Result["Date"].dt.weekday

FQMS_Result.to_csv('C:\\Users\\BOUDRA\\Desktop\\Projet Previs\\test\\FQMS_Result.csv' , sep = ';' , decimal = ',')

print(FQMS_Result["Date"])















