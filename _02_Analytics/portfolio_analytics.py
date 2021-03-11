from pandas import DataFrame, Series
import numpy as np

def drawdown(portfolio_value: pd.DataFrame):
    cv = portfolio_value.cumsum()
    i = np.argmax(np.maximum.accumulate(cv) - cv) # end of the period
    j = np.argmax(cv[:i]) # start of period
    print("MDD starts from" + j + "ends at" + i)
    return portfolio_value[i] - portfolio_value[j]

def beta_to_mkt(portfolio_returns:pd.DataFrame, benchmark_returns:pd.DataFrame):
    # code written here to measure the beta to market of the portfolio
    # assume that portfolio returns is a DataFrame with "Date" and porfolio returns
    portfolio_returns.drop("Date", axis = 1) # drop "Date" column 
    benchmark_returns.drop("Date", axis = 1) # drop "Date" column
    assert len(portfolio_returns) == len(benchmark_returns), "Dimension mismatch"  #length needs to match 
    portfolio_rerturns = portfolio_returns.iloc[:, 0].values   # cast the portfolio returns into a 1d array
    benchmark_returns = benchmark_returns.iloc[:, 0].values    # cast the benchmark returns into a 1d array
    portfolio_var = np.var(portfolio_rerturns)
    portfolio_benchmark_covar = np.cov(portfolio_returns, benchmark_returns)[0,1]
    beta = portfolio_benchmark_covar / portfolio_var
    return beta

def treynor(portfolio_returns:pd.DataFrame, benchmark_returns:pd.DataFrame, factors:pd.DataFrame):
    # code written here to measure the treynor ratio of the portfolio
    # assume that portfolio returns is a DataFrame with "Date" and porfolio returns
    factors = pd.merge(factor, portfolio_returns["Date"], on='Date', how='inner')
    factors = factors.drop("Date", axis = 1)
    factors = factors.apply(lambda x: x / 100)
    rf = factors.loc[:, "RF"]
    portfolio_returns.drop("Date", axis = 1) # drop "Date" column 
    benchmark_returns.drop("Date", axis = 1) # drop "Date" column
    assert len(portfolio_returns) == len(riskfree), "Dimension mismatch"  #length needs to match 
    portfolio_rerturns = portfolio_returns.iloc[:, 0].values   # cast the portfolio returns into a 1d array
    benchmark_returns = benchmark_returns.iloc[:, 0].values    # cast the benchmark returns into a 1d array
    rf = rf.iloc[:,"RF"].values  # cast the riskfree returns into a 1d array
    diff = portfolio_returns - rf
    portfolio_var = np.var(portfolio_returns)
    portfolio_benchmark_covar = np.cov(portfolio_returns, benchmark_returns)[0,1]
    beta = portfolio_benchmark_covar / portfolio_var
    return diff/beta

def sharpe_ratio(portfolio_returns:pd.Series):]
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std()
    return sharpe_ratio

def jensens_alpha(historical_prices:DataFrame,weights:Series,factors:pd.DataFrame,benchmark_returns:pd.DataFrame):
    #input is the prices of all the stocks invested within the period
    #weight is the portfolio weights at the beginning of the period
    factors = factors[1:]
    factors = factors.apply(lambda x: x/100)
    riskfree = factors.loc[:,'RF']
    riskfree = riskfree.reset_index(drop=True)
    portfolio_returns.drop("Date", axis = 1) # drop "Date" column 
    benchmark_returns.drop("Date", axis = 1) # drop "Date" column
    assert len(portfolio_returns) == len(riskfree), "Dimension mismatch"  #length needs to match 
    portfolio_rerturns = portfolio_returns.iloc[:, 0].values   # cast the portfolio returns into a 1d array
    benchmark_returns = benchmark_returns.iloc[:, 0].values    # cast the benchmark returns into a 1d array
    rf = rf.iloc[:, 0].values  # cast the riskfree returns into a 1d array
    portfolio_var = np.var(portfolio_rerturns)
    portfolio_benchmark_covar = np.cov(portfolio_returns, benchmark_returns)[0,1]
    beta = portfolio_benchmark_covar / portfolio_var
    returns = historical_prices / historical_prices.iloc[0]

    for i in range(len(weights)):
    returns[returns.columns[i]]=returns[returns.columns[i]].multiply(weights[i])
    returns = returns *100000
    # add a total portfolio column
    returns['Total'] = returns.sum(axis=1)

    # Daily Return
    returns['Daily Return'] = returns['Total'].pct_change(1)

    #jensens_alpha 
    ja = portfolio_val['Daily Return'].mean()-(riskfree.mean()+beta*(benchmark_returns-riskfree.mean()))

    #Annual jensens_alpha 
    Aja = (252**0.5) * ja

    return sharpe_ratio, ASR