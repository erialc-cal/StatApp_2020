#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 19:46:21 2020

@author: victorhuynh
"""

import pandas as pd

annees = ["2012","2013","2014","2015","2016","2017","2018"]

df=pd.DataFrame()

for annee in annees :
    xls = pd.read_csv("/Users/victorhuynh/Documents/ENSAE/ENSAE 2A/2A S2/Stat App/Historique Trafic/Histo Trafic " + "" + str(annee) + ".csv", sep = ";") 
    df = df.append(xls)

df = df.reset_index(drop = True)

df.to_csv("historique_concat.csv",encoding = 'utf-8')