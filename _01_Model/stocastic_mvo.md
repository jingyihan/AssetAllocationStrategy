## Stocastic Mean-Variance Optimization
A two-stage stochastic programming approach is implemented to the classic multi-period MVO strategy. This formulation incorporates the uncertainties in parameter estimations in the optimization problem and enables the multi-period rebalancing forecasting for periodic rebalancing framework adopting two-stage stochastic programming formulation and multi-period framework. Transaction cost and holding cost is considered in the model. Short selling can be either allowed or disallowed in the model based on user’s selection.

### Derivation
Base on the single-period Hybrid MVO, the following stochastic program is formulated:

![image](https://user-images.githubusercontent.com/24922489/111053051-3fdcf100-8426-11eb-941a-5c8c0dbcf32a.png)

where <img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " />  is a  matrix of expected returns for scenario  at time step .  is the number of scenarios. 

 is the objective variable which is a  matrix that consists of the optimal weights for the n assets to construct the portfolio. *When short selling is not allowed, the lower bound of  is set to be 0. Else, this constrain is removed.
 
 is the second stage variable measuring the amount by which expected return of portfolio is below the target return for scenario .  is the second stage variable measuring the amount by which the expected return of portfolio is above the target return for scenario .
 
 is the reward parameter that rewards surplus from target return.  is the penalty parameter that punishes shortfall from target return. In the case to optimize the sharpe ratio,  can be set relatively small and  can be set relatively large. R is an arbitrary benchmark return which can be set to risk free rate if not specified.

is a discount factor in the objective function with period step t at the exponent, such at first period, t=0, the discount factor is 1.  is a positive number smaller than 1. The formulation implies that optimizing with the most recent parameters is more important than the future parameters. 

is the risk aversion coefficient that describes the importance of variance minimization.
 stands for transaction cost generated when rebalancing the portfolio.
 
![image](https://user-images.githubusercontent.com/24922489/111053062-508d6700-8426-11eb-809e-2623740ed8a7.png)
