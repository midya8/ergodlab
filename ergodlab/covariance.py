"""Exact finite-lag covariances of stationary transition rewards."""
from fractions import Fraction

from .core import audit, pack


def audit_covariance(model, *, max_lag=15):
    """Decompose stationary block variance without assuming correlations decay.

    Rewards belong to (state, action, next_state) edges. Using only the
    conditional reward at the source loses dependence through the next state.
    Existing stationary reports and their schema are not changed.
    """
    if type(max_lag) is not int or not 0 <= max_lag <= 31:
        raise ValueError('max_lag must be an integer in 0..31')
    base = audit(model, block_limit=max_lag + 1)
    q = lambda value: Fraction(*value)
    mu = list(map(q, base['stationary']))
    transition = [list(map(q, row)) for row in base['controlled_transition']]
    expected = list(map(q, base['state_expected_rewards']))
    mean = q(base['average_reward'])
    # incoming[j] = E[R_0 * 1{S_1=j}], with the stationary edge law.
    incoming = [Fraction(0) for _ in mu]
    for edge in base['variance']['martingale_edges']:
        incoming[edge['next_state']] += q(edge['stationary_probability']) * q(edge['reward'])
    zero = q(base['variance']['one_step_stationary'])
    rows = [{'lag': 0, 'cross_moment': zero + mean**2, 'covariance': zero}]
    future = expected[:]
    for lag in range(1, max_lag + 1):
        cross = sum((mass * reward for mass, reward in zip(incoming, future)), Fraction(0))
        rows.append({'lag': lag, 'cross_moment': cross, 'covariance': cross - mean**2})
        future = [sum((p * reward for p, reward in zip(row, future)), Fraction(0))
                  for row in transition]
    blocks = []
    for original in base['variance']['stationary_blocks']:
        length = original['length']
        diagonal = length * zero
        lag_terms = [{'lag': lag, 'contribution': 2 * (length - lag) * rows[lag]['covariance']}
                     for lag in range(1, length)]
        off_diagonal = sum((row['contribution'] for row in lag_terms), Fraction(0))
        variance = diagonal + off_diagonal
        residual = variance - q(original['sum_variance'])
        if residual:
            raise ArithmeticError('covariance decomposition disagrees with block moment recurrence')
        blocks.append({'length': length, 'diagonal_contribution': diagonal,
                       'lag_contributions': lag_terms, 'off_diagonal_contribution': off_diagonal,
                       'sum_variance': variance, 'moment_recurrence_residual': residual})
    return pack({'format': 'ergodlab-covariance-v1', 'stationary_audit': base,
                 'max_lag': max_lag, 'reward_autocovariances': rows,
                 'block_decompositions': blocks,
                 'scope': 'Exact stationary scalar edge rewards for the supplied finite irreducible chain. '
                          'Finite lags only; periodic chains supported. No summability, mixing rate, '
                          'infinite covariance-sum, simulation or gradient-estimator variance claim.'})
