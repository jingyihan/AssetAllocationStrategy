import pandas as pd
import numpy as np

from numpy.linalg import inv
from scipy.stats.mstats import gmean
from carhart import carhart

from scipy.linalg import cholesky, block_diag
from statsmodels.stats.moment_helpers import cov2corr
from stochastic_mvo import stochastic_mvo
from mvo import mvo
from cvar_2 import CVaR

import cvxpy as cvx

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'figure.figsize': (15, 10),
          'legend.loc':'best',
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10}
pylab.rcParams.update(params)

prices_org = pd.read_csv("adjClose.csv")
factor_org = pd.read_csv("factors.csv")
factor_org = factor_org.dropna()

dt = 12
#N = int(round(len(factor_org)/dt, 0))
N = 5

prices_org['Date'] = prices_org['date'].apply(lambda x: x[:7])
prices = pd.merge(prices_org, factor_org['Date'], on='Date', how='inner')
prices  = prices.iloc[len(prices)-dt*(N-1)-1:, :]
factor = factor_org.iloc[len(factor_org)-dt*(N-1):, :]

#risk adversion coefficient
risk_weight_coefficient = 5000
    
#transaction cost
tcost = 0.05
initialVal = 100
tcost_negate = 0

currentPortVal = [[initialVal, initialVal, initialVal, initialVal]]

#save weights
w = [[] for i in range(4)]
trans_cost = [[] for i in range(4)]
pval = [[] for i in range(4)]
NoShares = [[] for i in range(4)]
w_p = [np.zeros((7,N-1)) for i in range(4)]
plot_prices = [[] for i in range(4)]
plot_dates = [prices.iloc[0,:].Date]
plot_dates_index = [prices.index[0]]

def plot_price_function(prices, NoShares, w, i):
    
    plot_price = prices.drop(['date','Date'], axis=1)
    
    fig, ax = plt.subplots()
    ax.plot(np.dot(plot_price,NoShares[0][i-2]), label='Stochastic MVO')
    ax.plot(np.dot(plot_price,NoShares[1][i-2]), label='Benchmark MVO')
    ax.plot(np.dot(plot_price,NoShares[2][i-2]), label='Equal Weight')
    ax.legend()
    plt.show()
    
    sns.heatmap(w[0][i-2][:,0].reshape(len(w[0][i-2][:,0]),1), annot=True, fmt='.2f')
    plt.show()
    
    sns.heatmap(w[1][i-2].reshape(len(w[1][i-2]),1), annot=True, fmt='.2f')
    plt.show()
    return

for i in range(0, N-1):
    
    train_prices = prices.iloc[i * dt:(i+1) * dt,:]
    train_factors = factor.iloc[i * dt:(i+1) * dt-1, :]
    
    print("i: {}, N left to model: {}".format(i, N))
    
    # remove dates in data  
    current_price = train_prices.iloc[0, :].drop(['date','Date'])

    mu, Q = carhart(train_prices, train_factors)

    # target return
    targetRet = np.mean(mu)
    
    # transaction cost
    if(i>0):
        #tcost_negate = NoShares[0][i-1] * current_price / np.dot(NoShares[0][i-1], current_price) 
        tcost_negate = 0
    else:
        tcost_negate = 0
    
    N = N-1
    
    weight, tcost_sto = stochastic_mvo(Q, mu, dt*(N), N, targetRet, risk_weight_coefficient, tcost, tcost_negate, current_price, currentPortVal[0][0])
    mvo_weight = mvo(Q, mu, targetRet)
    cvar_weight = CVaR(mu, Q)
    
    # save weights
    w[0].append(weight[:,0])
    w[1].append(mvo_weight)
    w[2].append(np.ones(mu.shape)*(1/len(mu)))
    w[3].append(cvar_weight)
    
    ## save plotable weights
    w_p[0][:,i] = weight[:,0]
    w_p[1][:,i] = mvo_weight
    w_p[2][:,i] = np.ones(mu.shape)*(1/len(mu))
    w_p[3][:,i] = cvar_weight
    
    # Portfolio Value
    NoShares[0].append((weight[:,0] * currentPortVal[i][0] / current_price).T)
    NoShares[1].append((w[1][i] * currentPortVal[i][1] / current_price).T)
    NoShares[2].append((w[2][i] * currentPortVal[i][2] / current_price).T)
    NoShares[3].append((w[3][i] * currentPortVal[i][3] / current_price).T)
    
    current_price = train_prices.iloc[-1, :].drop(['date','Date'])
    
    currentPortVal.append([np.dot(NoShares[0][i], current_price), 
                           np.dot(NoShares[1][i], current_price),
                           np.dot(NoShares[2][i], current_price),
                           np.dot(NoShares[3][i], current_price)])
    
    #save trans_cost
    if i > 0:
        trans_cost[0].append(np.dot((tcost_sto * currentPortVal[i][0] / current_price), current_price)*tcost)
        trans_cost[1].append(currentPortVal[i][1] * tcost)
        trans_cost[2].append(currentPortVal[i][2] * tcost)
        trans_cost[3].append(currentPortVal[i][3] * tcost)
    
    plot_prices[0] = np.concatenate((plot_prices[0],np.dot(train_prices.drop(['date','Date'], axis=1),NoShares[0][i])), axis=None)
    plot_prices[1] = np.concatenate((plot_prices[1],np.dot(train_prices.drop(['date','Date'], axis=1),NoShares[1][i])), axis=None)
    plot_prices[2] = np.concatenate((plot_prices[2],np.dot(train_prices.drop(['date','Date'], axis=1),NoShares[2][i])), axis=None)
    plot_prices[3] = np.concatenate((plot_prices[3],np.dot(train_prices.drop(['date','Date'], axis=1),NoShares[3][i])), axis=None)

    plot_dates = np.concatenate((plot_dates,train_prices.iloc[-1].Date), axis=None)
    temp_index = train_prices.index
    plot_dates_index = np.concatenate((plot_dates_index,temp_index[-1]), axis=None)
   
plot_price = prices[-dt:].drop(['date','Date'], axis=1)
    
#plt_prices_df = pd.DataFrame(plot_prices, columns = ['Stochastic MVO', 'Benchmark MVO', 'Equal Weight', 'CVaR'])

fig, ax = plt.subplots()
ax.plot(plot_prices[0], label='Stochastic MVO')
ax.plot(plot_prices[1], label='Benchmark MVO')
ax.plot(plot_prices[2], label='Equal Weight')
ax.plot(plot_prices[3], label='CVaR')
ax.plot(prices.iloc[:-1].SPY.reset_index(drop=True)*100/prices.iloc[0].SPY, label='S&P500')
temp_index = prices.index
plt.setp(ax, xticks=plot_dates_index-temp_index[0], xticklabels=plot_dates)

legend = ax.legend()
plt.show()

sns.heatmap(w_p[0], annot=True, fmt='.2f')
plt.show()
sns.heatmap(w_p[1], annot=True, fmt='.2f')
plt.show()
sns.heatmap(w_p[3], annot=True, fmt='.2f')
plt.show()

#sns.heatmap(w[1][i-2].reshape(len(w[1][i-2]),1), annot=True, fmt='.2f')
#plt.show()
