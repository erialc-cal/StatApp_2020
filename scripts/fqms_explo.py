# -*- coding: utf-8 -*-
"""
Visualisations des FQMS

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
import bokeh 
fqms = pd.read_csv(r"C:\Users\heleo\Desktop\StatApp_2020\Documents\fqms_concat_def.csv", low_memory=False)


"""
On dispose des fichiers : fqm_[1-53]_2015, fqm_[1-52]_2016 et fqm_[1-41]_2017.

On a donc les estimations depuis le 01/01/2015 jusqu'au 22/10/2017

#   Column                  Non-Null Count   Dtype  
---  ------                  --------------   -----  
 1   Jour                    656768 non-null  float64   -----> date du vol (jour)
 2   Tav                     656768 non-null  object    ----->
 3   Cie                     656768 non-null  object    -----> code de compagnie ex. EZY
 4   Ligne                   656768 non-null  object    -----> numéro de ligne ex. 4060
 5   A/D                     656768 non-null  object    -----> arrivée ou départ 
 6   Aérog                   656768 non-null  object    -----> code aérogare ex. OS pour orly sud
 7   Hor                     656768 non-null  object    -----> heure du vol
 8   Sièges Corrections_ICI  656768 non-null  float64   -----> correction du nombre de sièges estimé (?) ex. 114
 9   Pax Estimés             654403 non-null  float64   -----> nombre de passagers estimés ex.167.22
 10  Prov/Dst                656768 non-null  object    -----> code aéroport de provenance ex. JFK
 11  Pays                    656768 non-null  object    -----> code pays d'A/D ex. US
 12  Qualité                 656768 non-null  object    -----> Faisceau qualité (code MTP, CEE, INT etc.) 
 13  Crit. Douanier          656768 non-null  object    -----> critère douanier H ou D
 14  Faisceau                656768 non-null  float64   -----> code Faisceau de 1 à 5
 15  LF%                     654403 non-null  float64   -----> LF% taux de remplissage estimé ex. 0.929 pour 92.9%
 16  IATA                    656768 non-null  object    -----> 
 17  Cpt Mvt                 656768 non-null  float64   -----> 
 18  Faisc. IATA             656768 non-null  object    -----> 
 19  Date du vol             656812 non-null  object    -----> date du vol (jour + hor)
 20  NB Sieges specif        656812 non-null  object    -----> nombre de sièges spécifiques au vol
 21  NB Sieges Standard      656768 non-null  object    -----> nombre de sièges standards (pas toujours inf à nb sieges specif)
 22  Date Equiv              656768 non-null  object    -----> date equiv avec laquelle on a fait les prévisions
 23  CLE SYNTH.              656768 non-null  object    -----> clef = code aerogarde+A/D+faisceau
 24  Semaine                 656768 non-null  float64   -----> numéro de la semaine
 25  CLE S/T                 656768 non-null  object    -----> 
dtypes: float64(7), int64(1), object(18)
memory usage: 130.3+ MB


On cherche à agréger les estimations au niveau faisceau. 

"""


# %%

fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(fqms.sort_values(by='Date du vol', ascending=False).isnull(), cbar=False)
plt.title('Complétude de la base, en blanc les valeurs manquantes')
print("Complétude totale en pourcentage de la base :", (1-fqms.isnull().sum().sum()/(len(fqms.index)*len(fqms.columns)))*100,'%')


# %% 

"""

Comparaison des différentes données sièges et remplissage

"""


from bokeh.plotting import figure, output_file, show
#fqms = fqms[fqms['Date du vol'].notna()].reset_index(drop = True)
#fqms = fqms[fqms['Date du vol'].str.contains('20')].reset_index(drop = True)

fqms = fqms[fqms['NB Sieges specif']!='nc']

fqms = fqms[fqms['NB Sieges Standard']!='nc']


# %% 

#Maintenant ça marche
fqms = fqms.astype({'Date du vol': 'datetime64[ns]','Date Equiv': 'datetime64[ns]', 'NB Sieges specif' : 'float64', 'NB Sieges Standard': 'float64'})
fqms['Date du vol'] = pd.to_datetime(fqms['Date du vol'], format='%Y%m%d')
fqms_jour =fqms.groupby(fqms['Date du vol']).sum()
print(fqms_jour.info())
# %% 


# données à observer
NBSSP = np.array(fqms_jour['NB Sieges specif'])
NBSST = np.array(fqms_jour['NB Sieges Standard'])
NBSCor =  np.array(fqms_jour['Sièges Corrections_ICI'])

dates = np.array(fqms_jour.index, dtype=np.datetime64)

window_size = 100
window = np.ones(window_size)/float(window_size)

# # données moyennées
# nbssp_avg = np.convolve(NBSSP, window, 'same')
# nbsst_avg = np.convolve(NBSST, window, 'same')
# nbcor_avg = np.convolve(NBSCor, window, 'same')

# output to static HTML file
output_file("seats.html", title="fqms_explo.py seats")

# create a new plot with a datetime axis type
p = figure(plot_width=800, plot_height=350, x_axis_type="datetime")

# add renderers
p.circle(dates, NBSSP, size=4, color='green', alpha=0.2, legend_label='Sièges spécifiques')
#p.line(dates, nbssp_avg, color='navy', legend_label='Sièges spécifiques')
p.circle(dates, NBSST, size=4, color='navy', alpha=0.2, legend_label='Sièges standard')
#p.line(dates, nbsst_avg, color='navy', legend_label='Sièges standard')
p.circle(dates, NBSCor, size=4, color='red', alpha=0.2, legend_label='Sièges corrigés')
#p.line(dates, nbcor_avg, color='navy', legend_label='Sièges corrigés')

# NEW: customize by setting attributes
p.title.text = "Visualisation des sièges renseignés"
p.legend.location = "top_left"
p.grid.grid_line_alpha = 0
p.xaxis.axis_label = 'Date'
p.yaxis.axis_label = 'Nombre de sièges'
p.ygrid.band_fill_color = "olive"
p.ygrid.band_fill_alpha = 0.1

# show the results
show(p)









