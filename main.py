import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot
import statsmodels.api as sm
from fredapi import Fred

from utils import  longOnly, getRet, getSPI, infoDisc

# Login to API
fred = Fred(api_key='bb02f973af8d7813a10815f0a2e9bb81')

# load getData (presaved)
f = open("rets.pkl", "rb")
rets = pickle.load(f)
f = open("moms.pkl", "rb")
moms = pickle.load(f)

# Set variables
minObs = 100
periods = rets['monthly'].index
# Get quarterly rebalancing dates
rebal_mask = periods.month.isin([3,6,9,12])
rebal_periods = periods[rebal_mask]

# Load SPI returns
spi = getSPI()['monthly']

# Load swiss bond yiels
rf = fred.get_series('IRLTLT01CHM156N')
rf = rf/1200 # /12 for monthly and 100 for %

# Monthly Adjustments
moms.pop('mom6rev', None)
res_moms = {}
res_moms_frog = {}
res_moms_quart = {}
res_moms_quart_frog = {}
for j in moms:
    strat_ret = {}
    position = {}
    strat_ret_frog = {}
    position_frog = {}
    strat_ret_quart = {}
    position_quart = {}
    strat_ret_quart_frog = {}
    position_quart_frog = {}
    # see if filter create a hole in the series of returns
    test_periods = []
    for i in periods:    
        # get tickers in scope
        tics_inscope = moms[j].loc[i].transpose().dropna().index.intersection(rets['monthly'].loc[i].transpose().dropna().index)
        # enough tickers in scope ?
        if len(tics_inscope) >= minObs:
            # Without frog in pan
            pos = longOnly(moms[j][tics_inscope].loc[i], 0.2)
            temp = getRet(rets['monthly'][pos].loc[i])
            strat_ret[i] = temp
            position[i] = pos
            
            # With frog in pan
            inf_disc = -infoDisc(j,pos,i,rets['daily'],moms[j])
            pos_frog = longOnly(inf_disc, 0.5)
            temp_frog = getRet(rets['monthly'][pos_frog].loc[i])
            strat_ret_frog[i] = temp_frog
            position_frog[i] = pos_frog
            
            # Quarter
            if i in rebal_periods:
                pos_quart = pos
            # if pos_quart in locals(): # if first period is not a quarter where we rebalance

            temp_quart = getRet(rets['monthly'][pos_quart].loc[i].fillna(0))
            strat_ret_quart[i] = temp_quart
            position_quart[i] = pos_quart
            
            # Quarter with frog
            if i in rebal_periods:
                pos_quart_frog = pos_frog
            # if pos_quart in locals(): # if first period is not a quarter where we rebalance
            temp_quart_frog = getRet(rets['monthly'][pos_quart_frog].loc[i].fillna(0))
            strat_ret_quart_frog[i] = temp_quart_frog
            position_quart_frog[i] = pos_quart_frog
        else:
            test_periods.append(i)
            
    for i in [(strat_ret,position,res_moms),
              (strat_ret_frog,position_frog,res_moms_frog),
              (strat_ret_quart,position_quart,res_moms_quart),
              (strat_ret_quart_frog,position_quart_frog,res_moms_quart_frog)]:
        
        # Calculate market returns
        #from_date = next(iter(i[0]))
        p = 0
        from_date = list(i[0])[p]
        while np.isnan(i[0][from_date]):
            p += 1
            from_date = list(i[0])[p]

        to_date = list(i[0])[-1]
        spi_mask = (spi.index >= from_date) & (spi.index <= to_date)
        spi_insample = spi[spi_mask]
        # convert back to norm returns
        spi_norm_ret = np.exp(spi_insample) - 1
        # calculate market excess returns
        # reset indices for merge
        df_temp1 = spi_norm_ret.reset_index()
        df_temp2 = rf.rename('risk_free_rate').reset_index()
        # merge on month as day does not match (apply fn)
        df_mkt = pd.merge(df_temp1,
                          df_temp2,
                          left_on=df_temp1['DATE'].apply(lambda x: (x.year, x.month)),
                          right_on=df_temp2['index'].apply(lambda x: (x.year, x.month)),
                          how = 'left').set_index('DATE')
        df_mkt['excess_return'] = df_mkt['Close']-df_mkt['risk_free_rate']
        
        df_ret = pd.DataFrame.from_dict(i[0], orient = 'index')
        mens = (((df_ret+1).prod()).pow(1/len(df_ret))-1)[0]
        ann = (mens+1)**12 -1
        sharpe_men = df_ret.mean()/df_ret.std(ddof=0)
        sharpe_ann = (12**0.5) * sharpe_men
        # Calculate alpha & beta
        x = df_mkt['excess_return']
        x = sm.add_constant(x).to_numpy()
        y = df_ret.to_numpy()
        model = sm.OLS(y,x)
        fit = model.fit() 
        beta = fit.params[1]
        avg_rf = np.power((df_mkt['risk_free_rate']+1).prod(),1/len(df_mkt['risk_free_rate']))**12 -1
        avg_mkt = np.power((df_mkt['excess_return']+1).prod(),1/len(df_mkt['excess_return']))**12 -1
        alpha = ann - avg_rf - beta*avg_mkt
     
        i[2][j] = [i[0], i[1], 
                   {'Monthly return': mens,
                    'Annual return': ann,
                    'Monthly Sharpe': sharpe_men,
                    'Annual Sharpe': sharpe_ann,
                    'Annualized Alpha': alpha,
                    'Annualized Beta': beta}]
        
 
# extreme values appear in plot 
to_plot = pd.DataFrame.from_dict(res_moms['mom6'][0], orient = 'index')
to_plot.plot()
pyplot.show()

# TO DO:
    # Transaction costs
        # ideas ex-ante and ex-post transaction costs 
        # ways to optimise the turnover ?
    # Presentations (vizu)



