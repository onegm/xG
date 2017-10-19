#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:42:01 2017

@author: onegm
"""
import numpy as np 
import math as m
import pandas as pd
from sklearn.neighbors import RadiusNeighborsRegressor as rnr
import matplotlib.pyplot as plt






# Calculate adj dist of shot 
def adjdist(x,y):

    dist = m.sqrt((120-x)**2 + (40-y)**2)
    
    #Angle to goal
    if (y>=36 and y <= 44): angle =1
    else:
        angle1 = 1-abs(m.degrees(m.acos((y-44)/m.sqrt((120-x)**2 + (44-y)**2)))/180 - 0.5)
        angle2 = 1-abs(m.degrees(m.acos((y-36)/m.sqrt((120-x)**2 + (36-y)**2)))/180 - 0.5)
        angle = max([angle1, angle2])
        
    #Adjusted Distance
    o = 0.799545363
    adjdist = dist/(angle**o)
    
        
    return adjdist







def lowerdf(df):
    # Turn all strings in df to lowercase
    for column in df.columns:
        if df[column].dtype == 'O': df[column] = df[column].str.lower()
    
    return df







def radiusreg(r, x, y):
    # Radius regression nearest neighbors
    model = rnr(radius = r, weights = 'uniform')
    model.fit(x, y)
    return model






def scatplot(x, y, title = ''):
    # scatter plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, c = 'b', marker = 'o', s = 5, alpha = 0.75)
    plt.show()



def plotmodel(adjdists, prediction):

    scatplot(adjdists, prediction, title = str(r) + ' yards')
        
    rmse = m.sqrt(sum((truearray - prediction)**2)/len(truearray))
    
    print(rmse)
    

#####################################################################


filename = '/Users/onegm/Desktop/Arqam/Consolidation/consolidatedshotsbig.csv'
df = pd.read_csv(filename, header = 0, index_col = None)
df = lowerdf(df)


adjdists = []
trueval = df['True Values'].tolist()
truearray = np.asarray(df['True Values'])


for i in range(len(df)):
    x = df.X[i]
    y = df.Y[i]
    
    dist = adjdist(x,y)
    
    adjdists.append([dist])

        

for r in [1, 3, 5]: 

    model = radiusreg(r, adjdists, trueval)
    
    prediction = []
    for i in range(len(df)):
        prediction.append(float(model.predict(adjdists[i][0])))
        
#    plotmodel(adjdists, prediction)
    
    prediction = np.asarray(prediction)


adjdists = np.array(adjdists).flatten()
df1 = pd.DataFrame(data = {'AdjDist': adjdists, 'True Values': trueval, 
                           'Prediction': prediction})
    
    
df2 = df1[df1['AdjDist']<40]
rmse2 = m.sqrt(sum((df2.Prediction - df2['True Values'])**2)/len(df2))


