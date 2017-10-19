#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:47:47 2017

@author: onegm
"""

import numpy as np 
import math as m
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib  import cm




def lowerdf(df):
    # Turn all strings in df to lowercase
    for column in df.columns:
        if df[column].dtype == 'O': df[column] = df[column].str.lower()
    
    return df











def realx(x, teamdir):
    # Convert shot locations to standard form for 2016-17
    
    if (teamdir == 'right to left' and isinstance(x, int)):
        x = 120 + x
        
    return x


def realy(y, teamdir):
    # Convert shot locations to standard form for 2016-17
    
    if (teamdir == 'left to right' and isinstance(y, int)):
        y = 80 - y
    elif (teamdir == 'right to left' and isinstance(y, int)):
        y = abs(y)
        
    return y


def realxy(df):
    # Convert shot locations to standard form for 2016-17
    
    df['X'] = df.apply(lambda row: realx(row['X'], row['Team Direction']), axis = 1)
    df['Y'] = df.apply(lambda row: realy(row['Y'], row['Team Direction']), axis = 1)
    
    return df










def standardfoot(foot):
    # Rename foot column from old format to new
    
    if foot == 'header':
        foot = 'head'
        
    elif foot == 'rightfoot':
        foot = 'right foot'
        
    elif foot == 'leftfoot':
        foot = 'left foot'
        
    return foot




def standardteams(team):
    
    if team in ['al ahly', 'ahl']:
        return 'ahly'
    
    elif team in ['asw']:
        return 'aswan'
    
    elif team in ['asi']:
        return 'asiut'
    
    elif team in ['dak']:
        return 'dakhlia'
    
    elif team in ['dam']:
        return 'damanhoor'
    
    elif team in ['wadi degla', 'deg']:
        return 'degla'
    
    elif team in ['enp']:
        return 'enppi'
    
    elif team in ['gei']:
        return 'geish'
    
    elif team in ['ghazl', 'ghazl al mahala', 'mah', 'ghazl al mahla']:
        return 'mahala'
    
    elif team in ['gou']:
        return 'gouna'
    
    elif team in ['har']:
        return 'haras'
    
    elif team in ['int']:
        return 'intag'
    
    elif team in ['ism', 'ismaili']:
        return 'ismaily'
    
    elif team in ['iti', 'ittihad']:
        return 'itihad skandary'
    
    elif team in ['mak']:
        return 'makasa'
    
    elif team in ['mas']:
        return 'masry'
    
    elif team in ['mok', 'moqaouloun', 'mokaweloun']:
        return 'mokawloon'
    
    elif team in ['nas']:
        return 'nasr'
    
    elif team in ['nasr lel_taadin']:
        return 'nasr lel-taadin'
    
    elif team in ['pet']:
        return 'petrojet'
    
    elif team in ['raj']:
        return 'raja'
    
    elif team in ['sho']:
        return 'shorta'
    
    elif team in ['smo', 'smoha']:
        return 'smouha'
    
    elif team in ['zam']:
        return 'zamalek'
    
    else: return team
    


def standardresults(result):
    
    if result == 'goalkick':
        return 'off t'
    
    elif result == 'cornerkick':
        return 'saved'
    
    elif result == 'blockbydefense':
        return 'blocked'
    
    elif result == 'bars':
        return 'post'
    
    else: return result




def standardevents(row):
    
    if row['Event'] == 'oneonone':
        row['Event'] = 'shot'
        row['Big Chance'] = 'big chance'
        row['Source'] = 'open play'
        
    elif row['Event'] == 'freekick':
        row['Event'] = 'shot'
        row['Source'] = 'free kick'
        
    elif row['Event'] == 'penalty':
        row['Event'] = 'shot'
        row['Source'] = 'penalty'
        
    elif row['Event'] == 'shoot':
        row['Event'] = 'shot'
        row['Source'] = 'open play'
        
    return row
        


        
        
def standarddf(filename):
    # Convert x and y to one format. Rename 'header' to 'head'
    df1 = pd.read_excel(filename, sheetname = '2014-2016')
    df2 = pd.read_excel(filename, sheetname = '2016-2017')
    
    df1 = lowerdf(df1)
    df2 = lowerdf(df2)
    

    # Converting shot locations to standard form 
    
    #2014-2016
    df1.X = df1.X + 60
    df1.Y = df1.Y + 40
    
    # 2016-2017
    df2 = realxy(df2).drop('Team Direction', axis = 1)
    
    
    df = pd.concat([df1, df2])
    
    df = df[(df.X <= 120) & (df.Y <= 80)]
    
    df.reset_index(drop=True, inplace=True)
    
    # Standardizing format
    df['Foot'] = df['Foot'].apply(lambda foot: standardfoot(foot))
    df['Result'] = df['Result'].apply(lambda result: standardresults(result))
    df['Team'] = df['Team'].apply(lambda team: standardteams(team))
    df['Opposition'] = df['Opposition'].apply(lambda team: standardteams(team))
    df = df.apply(lambda row: standardevents(row), axis = 1)

    
    return df


def rename(foot):
    # Rename foot column from old format to new
    
    if foot == 'header':
        foot = 'head'
        
    elif foot == 'rightfoot':
        foot = 'right foot'
        
    elif foot == 'leftfoot':
        foot = 'left foot'
        
    return foot







def dist(x, y):
    # Get the distance of a shot
        
    dist = np.sqrt((120-x)**2 + (40-y)**2)        
    
    return dist



def angle(x, y):
    # Get the angle of a shot 
    
    if (y>=36 and y <= 44): 
        angle =1
        
    else:
        angle1 = 1-abs(m.degrees(m.acos((y-44)/m.sqrt((120-x)**2 + (44-y)**2)))/180 - 0.5)
        angle2 = 1-abs(m.degrees(m.acos((y-36)/m.sqrt((120-x)**2 + (36-y)**2)))/180 - 0.5)
        angle = max([angle1, angle2])
        
    return angle




def get_dist_ang(df):
    # Populate new columns 'Distance' and 'Angle'
    df['Distance'] = dist(df.X, df.Y) #df.apply(lambda row: dist(row['X'], row['Y']), axis = 1)
    df['Angle'] = df.apply(lambda row: angle(row['X'], row['Y']), axis = 1) 
    
    return df







def shotfilter(shot_type, df ):
    # Filter all shots to specific shot type
    # Valid types: shot, big chance, head, freekick, penalty, right/left foot
    s = shot_type.lower()
    
    if s == 'shot':
        shotfilter = (df['Source'] == 'open play') & (df['Foot'] != 'head')
    
    elif s in ['head', 'right foot', 'left foot']:
        shotfilter = df['Foot'] == s
    
    elif s == 'big chance':
        
        shotfilter = df['Big Chance' == s]
    
    else:
        shotfilter = df['Source'] == s

        
    df = df[shotfilter]
    df.reset_index(drop=True, inplace=True)
    

    return df












def colormap(prob, title = ''):
    fig = plt.figure(figsize=(10, 6))
    
    ax = fig.add_subplot(111)
    ax.set_title(title.title())
    plt.imshow(prob, interpolation = 'nearest', vmax = 0.4)
    ax.set_aspect('equal')
    
    #cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    #cax.get_xaxis().set_visible(False)
    #cax.get_yaxis().set_visible(False)
    plt.colorbar(orientation='vertical')
    plt.show()















def scatplot(x, y, title = '', color = 'b', colmap = False):
    # scatter plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    if colmap:
        ax1.scatter(x, y, c = color, marker = 'o', s = 5, cmap = cm.YlOrRd)
    else:
        ax1.scatter(x, y, c = color, marker = 'o', s = 5, alpha = 0.75)
    plt.title(title)
    plt.show()













def expdecay(x, a, b, k = None):
    if len(x) > 2:
        return a*np.exp(-b*x)
    
    else:
        return a*np.exp(-b*x[0]/(x[1]**k))
    
    




def curvefit(func, X, Y, initial_p):
    
    
    param, pcov = curve_fit(func, X, Y, initial_p)        
    
    if len(X) > 2:
        rmse = m.sqrt(sum((Y - func(X, param[0], param[1]))**2)/len(X))
        
    else:
        rmse = m.sqrt(sum((Y - func(X, param[0], param[1], param[2]))**2)/len(X))
    
    return param, rmse









def adjdist(dist, angle):   
    # Distance adjusted to angle (not used)
    o = 0.896899
    adjdist = dist/(angle**o)
    
        
    return adjdist





    





def plotmodel(adjdists, prediction, truearray, r):

    scatplot(adjdists, prediction, title = str(r) + ' yards')
        
    rmse = m.sqrt(sum((truearray - prediction)**2)/len(truearray))
    
    print(rmse)
    
    



def calc_xG(df):
    
    dfgoal = df[df.isGoal == 1]
    
    x_edges = [0, 40, 60, 75, 85, 90, 95, 100,
               102.5, 105, 107.5, 110, 112.5, 115, 117.5, 120]
    
    y_edges = [0, 10, 17, 23, 29, 35, 40, 45, 51, 57, 63, 70, 80]


    H, x_edges, y_edges = np.histogram2d(df.X, df.Y, bins = (x_edges, y_edges))
    
    H[H < 20] = 0
    
    Hgoal, x_edges, y_edges = np.histogram2d(dfgoal.X, dfgoal.Y, bins = (x_edges, y_edges))
    
    
    Hdiv = np.divide(Hgoal, H)
    Hdiv[~np.isfinite(Hdiv)] = 0
    
    X, Y = np.meshgrid(x_edges, y_edges)
    
    fig = plt.figure(figsize=(10, 6))
    plt.subplot()
    plt.pcolor(X, Y, Hdiv.T, cmap = 'jet')
    plt.colorbar()

    
    df['raw_xG'] = df.apply(lambda row: get_bin_value(row, Hdiv, x_edges, y_edges), axis = 1)
    
    return df['raw_xG']



def get_bin_value(row, Hdiv, x_edges, y_edges):
    
    x = row.X
    y = row.Y
    
    x_index = next(a[0] for a in enumerate(x_edges) if a[1] > x - 1)
    
    if x_index > 0: 
        x_index -= 1
    
    
    y_index = next(a[0] for a in enumerate(y_edges) if a[1] > y) - 1
    
    
    return Hdiv[x_index, y_index]
    
    
    







###############################################################################

# Importing consolidated file and standardizing format
filename = '/Users/onegm/Desktop/Arqam/Consolidation/Consolidated Shots Full 14.17.xlsx'

df = standarddf(filename)

# Probability matrix of each bin for each shot type
df['isGoal'] = (df['Result'] == 'goal') * 1
df = get_dist_ang(df)

dfshot = shotfilter('shot', df)

dfshot['raw_xG'] = calc_xG(dfshot)



