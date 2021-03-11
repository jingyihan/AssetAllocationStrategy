import pandas as pd
import numpy as np

from scipy.linalg import cholesky, block_diag
from statsmodels.stats.moment_helpers import cov2corr

import cvxpy as cvx

def stochastic_mvo(Q, mu, T, N, targetRet, risk_weight_coefficient, tcost, tcost_negate, currentPrices, currentPortVal):
    
    num_asset = len(mu)
    rho = cov2corr(Q)
    
    nPaths = 100
    L = cholesky(rho, lower=True)
    dt = T/N
    
    # Because it is in a minimization problem, so minus means reward
    reward_per_dollor_surplus = 0
    # Because it is in a minimization problem, so positive means punishment
    punishment_per_dollor_shortfall = 0

    #transaction cost
    tcost_coef = 1
    
    # ######################
    # monte carlo simulaton 
    # ######################
    
    # Matrix of simulated price paths num_asset * nPeriods+1 * nPaths, set initial price to be $100
    S = np.array([[[0.0 for k in range(nPaths)] for j in range(N+1)] for i in range(num_asset)])
    S[:, 0, :] = 100
    
    # Generate paths
    for i in range(nPaths):
        for j in range(N):
            xi = np.dot(L,np.random.randn(num_asset, 1))
            for k in range(num_asset):
                S[k, j+1, i] = S[k, j, i] * np.exp( ( mu[k] - 0.5 * Q[k, k] ) * dt \
                                + np.sqrt(Q[k, k]) * np.sqrt(dt) * xi[k] )
    
    # returns_sample n_asset * nPeriod * nPaths
    returns_sample = np.array([[[0.0 for k in range(nPaths)] for j in range(N)] for i in range(num_asset)])
    
    for i in range(num_asset):
        returns_sample[i] = S[i,:-1,:] / S[i, 1:,:] - 1
    
    '''
    Have nPath scenarios and each scenario has a surplus and shortfall
    Also has num_asset * N (n_period) weight variables
    In total, have 2 * nPath * num_asset * N + num_asset * N (n_period) variables.
    First nPath * N variables are surplus variables.
    Second nPath * num_asset * N  are shortfall variables.
    N-1 transaction cost variables.
    The last num_asset * N are weight variables.
    '''
    
    #### objective function
    
    ## linear portion: surplus and shortfall
    total_vars = 2 * nPaths * N + num_asset + (N-1) + num_asset * N
    q = np.zeros(total_vars)
    q[:nPaths * N] = (1 / nPaths) * reward_per_dollor_surplus
    q[nPaths * N : 2 * nPaths * N] = (1 / nPaths) * punishment_per_dollor_shortfall
    # transcation cost
    q[2 * nPaths * N : 2 * nPaths * N + num_asset + (N-1)] = 0
    
    ## quadratic portion: minimize risk
    P = np.zeros((total_vars, total_vars))
    blocks = risk_weight_coefficient * Q
    for i in range(N-1):
        blocks = block_diag(blocks, risk_weight_coefficient * Q)
    
    P[2 * nPaths * N + num_asset + (N-1):, 2 * nPaths * N + num_asset + (N-1):] = blocks
    
    ## Linear equal constrain
    Aeq = np.zeros((nPaths * N + num_asset + (N-1) + N, 2 * nPaths * N + num_asset + (N-1) + num_asset * N))
    blocks = np.eye(nPaths)
    for i in range(N-1):
        blocks = block_diag(blocks, np.eye(nPaths))
    Aeq[:nPaths * N, :nPaths * N] = -1 * blocks
    Aeq[:nPaths * N, nPaths * N : 2 * nPaths * N] = blocks
    Aeq[nPaths * N : nPaths * N + num_asset, 2 * nPaths * N : 2 * nPaths * N + num_asset] = -1 * np.eye(num_asset)
    Aeq[nPaths * N + num_asset: nPaths * N + num_asset + N-1, 2 * nPaths * N + num_asset: 2 * nPaths * N + num_asset + N-1] \
        = -1 * np.eye(N-1)
    
    # transcation cost
    blocks = np.ones(num_asset) * tcost_coef
    Aeq[nPaths * N : nPaths * N + num_asset, 2 * nPaths * N + num_asset + (N-1) : 2 * nPaths * N + num_asset + (N-1) + num_asset] = np.eye(num_asset)
    if N>1:
        for i in range(1, N):
            print(i)
            blocks_1 = -1 * np.ones(num_asset)
            blocks_2 = np.ones(num_asset)
            blocks = np.hstack((blocks_1, blocks_2))
            Aeq[nPaths * N + num_asset + (i-1), 2 * nPaths * N + num_asset + (N-1) + (i-1) * num_asset : 2 * nPaths * N + num_asset + (N-1) + (i+1) * num_asset] = blocks
    Aeq[Aeq==0.] = 0
    
    for i in range(N):
        for j in range(nPaths):
            Aeq[i*nPaths + j, 2 * nPaths * N+ num_asset + (N-1) + num_asset * i : 2 * nPaths * N + num_asset + (N-1) + num_asset * (i+1)] = returns_sample[:,i,j]
    
    blocks = np.ones(num_asset)
    for i in range(N-1):
        blocks = block_diag(blocks, np.ones(num_asset))
    Aeq[nPaths * N + num_asset + (N-1):, 2 * nPaths * N + num_asset + (N-1): 2 * nPaths * N + num_asset + (N-1) + num_asset * N] = blocks
    
    beq = np.ones(nPaths * N + num_asset + (N-1) + N)
    beq[nPaths * N : nPaths * N + num_asset] = tcost_negate
    beq[nPaths * N + num_asset : nPaths * N + num_asset + N - 1] = 0
    beq[: nPaths * N] = targetRet
    
    # upper and lower bound
    lb = np.zeros(2 * nPaths * N + num_asset + (N-1) + num_asset * N)
    ub = np.ones(2 * nPaths * N + num_asset + (N-1) + num_asset * N)
    b = np.hstack((ub,lb))
    A = np.vstack((np.eye(2 * nPaths * N + num_asset + (N-1) + num_asset * N),-1 * np.eye(2 * nPaths * N + num_asset + (N-1) + num_asset * N)))
    A[2 * nPaths * N : 2 * nPaths * N + num_asset + (N-1), 2 * nPaths * N : 2 * nPaths * N + num_asset + (N-1)] = 0
    
    x = cvx.Variable(2 * nPaths * N + num_asset + (N-1) + num_asset * N)
    prob = cvx.Problem(cvx.Minimize((1/2)*cvx.quad_form(x, P) + q.T@x), \
                     [A@x <= b, \
                      Aeq@x == beq])
    prob.solve()
    
    weight = x.value
    trans_cost = weight[-num_asset * N : -num_asset * N + num_asset]
    
    weight = weight[-num_asset * N:].reshape(N, num_asset)
    weight = weight.T

    return weight, trans_cost
