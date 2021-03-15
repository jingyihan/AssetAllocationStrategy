## CVAR Optimization
CVaR optimization is a strategy that focused on minimizing the loss of the portfolio. This approach emphasizes on reducing the risk of the portfolio using Value at risk. CVaR extends from concept of VaR, also known as the expected shortfall. It attempts to address the shortcomings of the VaR model, which is a statistical technique used to measure the level of financial risk within a firm or an investment portfolio over a specific time frame. CVaR is the expected loss if that worst-case threshold is ever crossed. It quantified the expected losses that occur beyond the VaR breakpoint. 

### Parameter Uncertainty Consideration and Monte Carlo simulation
Price scenarios are generated with Monte Carlo simulation where single step prediction is taken for each rebalance period assuming the price of stocks follows geometric Brownian motion.

![image](https://user-images.githubusercontent.com/24922489/111163110-6f911380-8562-11eb-899d-28449038aa03.png)

### Formulation
![image](https://user-images.githubusercontent.com/24922489/111163448-bc74ea00-8562-11eb-9785-52bf344a63d8.png)
