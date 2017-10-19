#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:23:48 2017

@author: onegm
"""
import pandas as pd 
import numpy as np
import math as m
import sys
import glob

sys.path.insert(0, '/Users/onegm/Desktop/Arqam/PY Validation')
import Events as e

from scipy.optimize import least_squares
import matplotlib.pyplot as plt

file1 = '/Users/onegm/Desktop/Arqam/Consolidation/Consolidated Shots.xlsx'
df = pd.read_excel(file1, sheetname = 0, header = 0, index_col = None)

df.rename(columns={'X=': 'X', 'Y=': 'Y'}, inplace=True)
df.X = df.X +60
df.Y = df.Y +40





files = glob.glob('/Users/onegm/Desktop/Arqam/16.17/*.csv')

l = len(df)
for file in files:
    df2 = pd.read_csv(file,index_col=None, header=0, encoding = 'utf-16le', delimiter = '\t')
    # Add Big chance column if not there.
    if 'Big Chance' not in df2.columns:
        df2.insert(13, 'Big Chance', pd.Series([0]*len(df2)))

    # Turn all strings to lowercase
    for column in df2.columns:
        if df2[column].dtype == 'O': df2[column] = df2[column].str.lower()

    for i in range(len(df2)):
        
        if (df2['Event'][i]== 'shot' and isinstance(df2['Start Location'][i], str)):
            event = e.Shot(df2, i)
            l = l+1
            
            df.loc[l, 'X'] = event.startx
            df.loc[l, 'Y'] = event.starty
            df.loc[l, 'EventEn='] = event.source
            if event.isbchance: df.loc[l, 'EventEn='] = 'OneONOne'
        
            if event.isgoal:
                df.loc[l, 'ResultID='] = 17
            else: 
                df.loc[l, 'ResultID='] = 16
            

df.to_csv('/Users/onegm/Desktop/Arqam/consolidatedshotsbig.csv', 
          index = False, columns = df.columns)

#df = df[df.X <= 120]; df = df[df.X >= 0]
#df = df[df.Y <= 80]; df = df[df.Y >= 0]
df.reset_index(drop=True, inplace=True)


#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(df.X, df.Y, c = 'b', marker = '.', s = 10, alpha = 0.25)
#ax1.scatter([120, 120], [44, 36], c = 'r', marker = 'o', s = 20, alpha = 1)
#plt.show()

yrds = 5



shots = []
shotx = df.X
shoty = df.Y


shots = np.array(shots)

result = df['ResultID=']==17       

binx = shotx//yrds
biny = shoty//yrds




bin2 = binx+ biny *(12)

binresult = np.zeros((120*80//yrds,2))

for i in range(len(shotx)):
    binresult[bin2[i]][0] = binresult[bin2[i]][0] +1
    if result[i]==1:
        binresult[bin2[i]][1] = binresult[bin2[i]][1] + 1

np.seterr(divide='ignore', invalid='ignore')                 
binprob = np.divide(binresult[:,1],binresult[:,0])
binprob = np.nan_to_num(binprob)

shotprob = []
for i in range(len(shotx)):
    shotprob.append(binprob[bin2[i]])
    df.loc[i, 'True Values'] = shotprob[i]
          
df.to_csv('/Users/onegm/Desktop/Arqam/Consolidation/consolidatedshotsbig.csv', 
          index = False, columns = df.columns)


goalx = 120;
goaly = 40;
distx = shotx - 120
disty = shoty - 40
dist = np.sqrt(distx**2 + disty**2)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(dist, shotprob, c = 'b', marker = 'o', s = 5, alpha = 0.75)






#
#H = [binprob[0:11], binprob[12:23], binprob[24:35],
#    binprob[36:47], binprob[48:59], binprob[60:71],
#    binprob[72:83], binprob[84:95]]
#
#
#fig1 = plt.figure(figsize=(12, 8))
#
#ax = fig1.add_subplot(111)
#ax.set_title('colorMap')
#plt.imshow(H, vmax =0.7)
#ax.set_aspect('equal')
#
#cax = fig1.add_axes([0, 120, 0, 80])
#cax.get_xaxis().set_visible(False)
#cax.get_yaxis().set_visible(False)
#cax.patch.set_alpha(0)
#cax.set_frame_on(False)
#plt.colorbar(orientation='vertical')
#plt.show()



x = range(0, int(np.sqrt(120**2+40**2)))
#a = 0.76468465; c = 0; k = 0.1140749 #Small Sample 
#a = 0.675; c = 0; k = 0.111 # Large Sample
#a = 0.69473466; c = 0; k = 0.09604; o = 0.894177 # Large Sample w/ adjusted dist

#xG model
a = 0.689588202; c = 0.0; k = 0.098172893; o = 0.799545363
y = []
for i in range(len(x)):
    y.append(a*m.exp(-k*x[i])+c)
    
ax1.scatter(x,y, c = 'r', marker = '^', s = 7, alpha = 0.5)

x = range(0, int(np.sqrt(120**2+40**2)))
a = 0.921985; c = 0.036212; k = 1/4.79
y = []
for i in range(len(x)):
    y.append(a*m.exp(-k*x[i])+c)

ax1.scatter(x,y, c = 'g', marker = '*', s = 7, alpha = 0.5)

plt.show()

angles = []
for i in range(len(shotx)):
    x = shotx[i]
    y = shoty[i]
    
    #Angle to goal
    if (y>=36 and y <= 44): angle =1
    else:
        angle1 = 1-abs(m.degrees(m.acos((y-44)/m.sqrt((120-x)**2 + (44-y)**2)))/180 - 0.5)
        angle2 = 1-abs(m.degrees(m.acos((y-36)/m.sqrt((120-x)**2 + (36-y)**2)))/180 - 0.5)
#       angle1 = abs(m.degrees(m.acos((44-y)/m.sqrt((120-x)**2 + (44-y)**2)))/90 - 1)
#       angle2 = abs(m.degrees(m.acos((36-y)/m.sqrt((120-x)**2 + (36-y)**2)))/90 - 1)
        angle = max([angle1, angle2])
        angles.append(angle)

#fig1 = plt.figure()
#ax2 = fig1.add_subplot(111)
#ax2.scatter(angles, shotprob, c = 'b', marker = 'o', s = 5, alpha = 0.75)
#plt.show()
#
#dfangles = pd.DataFrame(data = {'angles': angles, 'shotprob': shotprob}, 
#                        columns = ['angles', 'shotprob'])


