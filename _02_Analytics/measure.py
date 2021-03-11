from pandas import DataFrame, Series


def portfolio_performance(portfolio: Series, historical_prices: DataFrame):
    # Write code here to measure the total_return of the portfolio
    # assume start with $1
    shares = portfolio / historical_prices.iloc[0, :]
    portfolio_values = (shares * historical_prices).sum(axis=1)
    return portfolio_values


def scenarios(portfolio_values: DataFrame, actual_years: int):
    # take any 5 yr interval & put them into a percentile basis
    five_year_returns = portfolio_values[60:].values / portfolio_values[:-60]
    worse_case = five_year_returns.quantile(0.05) ** (actual_years / 5)
    expected_case = five_year_returns.mean() ** (actual_years / 5)
    better_case = five_year_returns.quantile(0.95) ** (actual_years / 5)
    return worse_case, expected_case, better_case
