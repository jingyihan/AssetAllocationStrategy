import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy.stats.mstats import gmean

from sklearn.decomposition import PCA

def estimate(prices: pd.DataFrame, factors: pd.DataFrame):
    # Modify code here
    # write code here
    # make sure fama-french factors and prices have the same length of datas
    prices_org = prices
    factor_org = factors
    # prices_org["datestr"] = prices_org["date"].map(lambda x: str(x)[:7])
    # factor_org["datestr"] = factor_org["date"].map(lambda x: str(x)[:7])
    # prices = pd.merge(prices_org, factor_org["datestr"], on="datestr", how="inner")

    # remove dates in data
    prices = prices_org.reset_index(drop=True)
    factors = factor_org.reset_index(drop=True)

    # organize data into riskfree, factors, and returns
    factors = factors.apply(lambda x: x / 100)
    riskfree = factors.loc[:, "RF"]
    riskfree = riskfree.reset_index(drop=True)
    factors = factors.drop("RF", axis=1)
    returns = prices[1:] / prices[:-1].values - 1
    returns = returns.reset_index(drop=True)[:-1]
    returns = (returns.T - riskfree).T

    pca = PCA(n_components=factors.shape[1])
    factors = pca.fit_transform(factors)

    # Slove for a and beta
    B = np.ones(len(factors))
    B = np.c_[B, factors]

    loadings = np.dot(np.dot(inv(np.dot(B.T, B)), B.T), returns.to_numpy())

    # generate n_stock x 1 vector of alphas
    A = loadings[0, :]

    # generate n_factor x n_stock matrix of betas
    V = loadings[1:, :]

    # factor expected returns, f_bar, n_factor x 1
    f_bar = gmean(B[:, 1:] + 1) - 1
    f_bar = f_bar.T

    # factor covariance matrix, F, n_factor x n_factor
    F = np.cov(factors.T)

    # Regression residuals, epsilon
    epsilon = returns.to_numpy() - np.dot(B, loadings)

    # Diagonal n_stock x n_stock matrix of residual variance
    D = np.diag(np.diag(np.cov(epsilon.T)))

    # 1 x n_stock vector of asset exp. returns
    mu = np.dot(V.T, f_bar) + A
    mean_returns = pd.Series(mu.T, index=prices.columns)

    # n_stock x n_stock asset covariance matrix
    variance_covariance = np.dot(np.dot(V.T, F), V) + D

    ###### No Used, mean_returns and variance_covariance return as numpy array
    # mean_returns = pd.DataFrame()
    # variance_covariance = pd.DataFrame()
    ###
    return mean_returns, variance_covariance


def test():
    """run this function to test"""
    prices = pd.read_csv("../data/adjClose.csv", index_col=["date"])
    factors = pd.read_csv("../data/factors.csv", index_col=["Date"])
    mu, Q = estimate(prices, factors)
    # check one example
    assert mu == []
    assert Q == [[]]
    raw_mu, raw_Q = raw_harry.estimate(prices, factors)
    # plot some differences


if __name__ == "__main__":
    test()
