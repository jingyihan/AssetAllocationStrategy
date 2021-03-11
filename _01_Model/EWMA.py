import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

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
riskfree = factor_org.loc[:,'RF']
riskfree = riskfree.reset_index(drop=True)

prices = prices_org
prices = prices.drop(['date'], axis=1)
returns = prices[1:].values / prices[:-1] - 1
returns = returns.apply(lambda x: x-riskfree).dropna()
T = returns.shape[0]
T = int(T/12)

for i in range(1,T):
    returns_temp = returns.iloc[(i-1)*12:i*12,:]
    return_test = returns_temp.ewm(span = 12)
    covs = returns_temp.ewm(span = 12).cov()
    covs = covs.iloc[-len(returns_temp.columns):]
    covs = covs.reset_index(level = 0, drop = True)
    if not np.all(np.linalg.eigvals(covs) > 0):
        print(np.all(np.linalg.eigvals(covs) > 0))
    #print(covs)
    
lag = 6
returns_ewma = returns.ewm(span = lag, adjust = True).mean()
returns_test = returns[1:].values
returns_ewma = returns_ewma[:-1].values

rolling_vars = returns.rolling(lag).var()
rolling_vars = rolling_vars[lag+1:].values
vars_ewma = returns.ewm(span = lag, adjust = True).var()
vars_ewma = vars_ewma[lag:len(vars_ewma)-1].values

error = mean_squared_error(returns_test[:,3], returns_ewma[:,3])
print(error)
plt.plot(returns_test[:,3], label='actual')
plt.plot(returns_ewma[:,3],label='forcast')
plt.legend()
plt.show()

error = mean_squared_error(rolling_vars[:,3],vars_ewma[:,3])
print(error)
plt.plot(rolling_vars[:,3], label='actual')
plt.plot(vars_ewma[:,3],label='forcast')
plt.legend()
plt.show()