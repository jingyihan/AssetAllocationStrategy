import pandas as pd
import numpy as np

from numpy.linalg import inv
from scipy.stats.mstats import gmean

from sklearn.decomposition import PCA

def carhart(prices: pd.DataFrame, factors: pd.DataFrame):
    # write code here
    # make sure fama-french factors and prices have the same length of datas
    prices_org = prices
    factor_org = factors
    prices_org['Date'] = prices_org['date'].apply(lambda x: x[:7])
    prices = pd.merge(prices_org, factor_org['Date'], on='Date', how='inner')

    # remove dates in data
    prices = prices_org.drop(['date','Date'], axis=1)
    factors = factor_org.drop('Date', axis=1)

    # organize data into riskfree, factors, and returns
    #factors = factors[1:]
    factors = factors.apply(lambda x: x/100)
    riskfree = factors.loc[:,'RF']
    riskfree = riskfree.reset_index(drop=True)
    factors = factors.drop('RF', axis=1)
    returns = prices[:-1] / prices[1:].values - 1
    returns = returns.reset_index(drop=True)
    returns = returns.apply(lambda x: x-riskfree)

    pca = PCA(n_components=factors.shape[1])
    factors = pca.fit_transform(factors)

    # Slove for a and beta
    B = np.ones(len(factors))
    B = np.c_[B, factors]
    
    loadings = np.dot(np.dot(inv(np.dot(B.T,B)), B.T), returns.to_numpy())
        
    #generate n_stock x 1 vector of alphas
    A = loadings[0,:]
        
    #generate n_factor x n_stock matrix of betas
    V = loadings[1:,:]
    
    #factor expected returns, f_bar, n_factor x 1 
    f_bar = gmean(B[:,1:]+1)-1
    f_bar = f_bar.T
        
    #factor covariance matrix, F, n_factor x n_factor
    F = np.cov(factors.T)
    
    #Regression residuals, epsilon
    epsilon = returns.to_numpy() - np.dot(B, loadings)
        
    #Diagonal n_stock x n_stock matrix of residual variance
    D = np.diag(np.diag(np.cov(epsilon.T)))
        
    #1 x n_stock vector of asset exp. returns
    mu = np.dot(V.T, f_bar) + A
    mean_returns = mu.T
    
    #n_stock x n_stock asset covariance matrix
    variance_covariance = np.dot(np.dot(V.T,F),V) + D
    
    ###### No Used, mean_returns and variance_covariance return as numpy array
    # mean_returns = pd.DataFrame()
    # variance_covariance = pd.DataFrame()
    ###
    return mean_returns, variance_covariance

prices = pd.read_csv("adjClose.csv")
factor = pd.read_csv("factors.csv")

sample_prices = pd.DataFrame(data = [['2018-08-31 00:00:00+00:00', 1, 1, 2], \
                                     ['2018-09-28 00:00:00+00:00', 3, 1, 1], \
                                     ['2018-10-30 00:00:00+00:00', 1, 2, 1], \
                                     ['2018-11-30 00:00:00+00:00', 2, 1, 1]], \
                                columns = ['date', 'AAPL', 'GOOGL', 'MSFT'])
sample_factors = pd.DataFrame(data = [['2018-09', 2, 2, 2, 0, 2], \
                                      ['2018-10', 1, 2, 2, 0, 1], \
                                      ['2018-11', 0, 2, 1, 0, 2]], \
                                columns = ['Date','Mkt-RF', 'SMB', 'HML','RF', 'Mom'])

mean_returns, variance_covariance = carhart(sample_prices, sample_factors)
