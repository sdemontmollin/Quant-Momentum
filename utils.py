# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:04:41 2021

@author: smont
"""

import numpy as np
import pandas as pd


def longShort(vec):
    # Remove nan
    temp = vec.transpose().dropna()    
    # Sort
    temp = temp.sort_values()    
    # Size of quantiles
    quantileLength = round(len(temp)*0.1)    
    # Get top tickers
    top = list(temp[len(temp)-quantileLength:len(temp)].index)    
    # Get bottom tickers
    bottom = list(temp[0:quantileLength].index)
    # Return list of lists
    return [top,bottom]

def longOnly(vec, quant, transpose = True):
    
    # transpose
    if transpose:
        temp = vec.transpose()
    # Sort
    temp = temp.sort_values()    
    # Size of quantiles
    quantileLength = round(len(temp)*quant)    
    # Get top tickers
    top = list(temp[len(temp)-quantileLength:len(temp)].index)    
    # Return list of lists
    return top
    
def getRet(vec):
    # convert back to simple returns
    norm_ret = np.exp(vec) - 1
    # sum returns
    return sum(norm_ret)/len(norm_ret)

def readList(path, sep):
    content = open(path).read()
    content_list = content.split(sep)
    content_list[0] = content_list[0][2:]
    content_list[-1] = content_list[-1][:-2]
    return content_list

def fillGap(df):
    # Input df with nan values (holes) and bound by nan values
    # It will fill only the holes (forward fill)
    
    temp = df.fillna(method='ffill')
    rev_mask = df.fillna(method='bfill')
    rev_mask = rev_mask/rev_mask

    return temp * rev_mask

def createReturns(dat1):
    # Input should be daily prices in matrix form
    dat2 = dat1.groupby([dat1.index.year, dat1.index.month]).tail(1)
    monthly = np.log(dat2)-np.log(dat2.shift(1))
    daily = np.log(dat1)-np.log(dat1.shift(1))
    return {'monthly':monthly, 'daily':daily}

def createMomentum(returns):
    # Input should be matrix of monthly returns
    l = [returns.shift(i) for i in range(1,37)]
    mom1 = l[0]
    mom6 = sum(l[0:6])
    mom12 = sum(l[0:12])
    mom36 = sum(l)
    mom6rev = mom6 - mom6.shift(1)
    return {'mom1':mom1, 'mom6':mom6, 'mom12':mom12, 'mom36':mom36, 'mom6rev':mom6rev}

def getSPI():
    spi = pd.read_csv('SPI.csv',sep = ';')
    spi.index = pd.to_datetime(spi['DATE'])
    spi.drop('DATE', inplace = True, axis = 1)
    spi.sort_index(inplace = True)
    return createReturns(spi)

def infoDisc(mom,pos,date,dailyret,momret):
    
    # define lookback
    if mom == 'mom12':
        lookback = pd.DateOffset(years=1, months=1)
    elif mom == 'mom1':
        lookback = pd.DateOffset(months=2)
    elif mom == 'mom6':
        lookback = pd.DateOffset(months=7)
    elif mom == 'mom36':
        lookback = pd.DateOffset(years=3, months=1)
    
    # adjust date to get the dates of the mom period
    mom_adj = pd.DateOffset(months=1)

    # binary return
    temp = momret[pos].loc[date]
    temp[temp>0] = 1
    temp[temp<0] = -1
    pret = temp
    
    # get path
    lookback_date = date - lookback
    until = date - mom_adj
    temp_df = dailyret.loc[lookback_date:until][pos]
    per_pos = temp_df.gt(0).sum() / temp_df.count()
    per_neg = temp_df.lt(0).sum() / temp_df.count()
    
    return pret * (per_neg-per_pos)