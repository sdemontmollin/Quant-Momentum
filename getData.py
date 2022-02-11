import yfinance as yf
from pandas_datareader import data
from datetime import datetime
import pandas as pd
import pickle

from utils import readList, fillGap, createReturns, createMomentum

# script is for importing data

# tickers available on yahoo finance
tics = readList('tickers_yf.txt', "', '")

# import data with yfinance 
yf_hist = {}
yf_info = {}
pd_hist = {}
fail_info = []
for i in tics:
    temp = yf.Ticker(i)
    hist = temp.history(period="max")
    yf_hist[i] = hist
    try:
        carac = temp.info
        yf_info[i] = carac
    except:
        fail_info.append(i)
    panel_data = data.DataReader(i,  "yahoo", datetime(1990,1,1), datetime(2020,12,31))
    pd_hist[i] = panel_data
    
# Create matrix of price 
new_dict = {}
for i in pd_hist:
    new_dict[i] = pd_hist[i]['Adj Close']
temp = pd.concat({k: pd.Series(v) for k, v in new_dict.items()}).reset_index()
adj_price = temp.pivot(index ='Date', columns = 'level_0', values = 'Adj Close')

# fill na
adj_price = fillGap(adj_price)

# compute returns
rets = createReturns(adj_price)
# compute momentum
moms = createMomentum(rets['monthly'])

a_file = open("rets.pkl", "wb")
pickle.dump(rets, a_file)
a_file.close()
a_file = open("moms.pkl", "wb")
pickle.dump(moms, a_file)
a_file.close()


