# Finite-lag covariance of edge rewards

Fix the supplied policy at its anchor. Let `P` be its controlled transition
matrix, `mu` its stationary distribution, and `R_t = R(S_t,A_t,S_(t+1))`.
All expectations below start in stationarity. This calculation concerns the
scalar reward process, not a policy-gradient estimator.

Write `r(s) = E[R_t | S_t=s]` and `rho = mu r`. Define a row vector

```
u(j) = sum_(s,a) mu(s) pi(a|s) T(j|s,a) R(s,a,j).
```

The vector `u` carries the reward on the incoming edge into state `j`.
It can be signed. It is not generally `mu(j) r(j)`. Conditional on that
edge ending in `j`, the expected reward `k` steps after its reward, for
`k >= 1`, is `(P^(k-1) r)(j)`. The Markov property therefore gives

```
E[R_0 R_k] = u P^(k-1) r              (k >= 1)
C_k = E[R_0 R_k] - rho^2
C_0 = E[R_0^2] - rho^2.
```

The implementation advances the vector `P^(k-1) r` one matrix-vector
multiplication at a time. It never enumerates paths in production. Tests
separately enumerate complete edge paths for short lags, using analytic
two-state stationary probabilities rather than the production linear solve.

For a block of length `N`, expand the variance of the sum. There are `N`
diagonal terms and `N-k` pairs at each positive lag `k`. Scalar products
commute, so reversing a pair does not require reversibility of the chain.
Thus the exact finite identity is

```
Var(R_0 + ... + R_(N-1)) = N C_0 + 2 sum_(k=1)^(N-1) (N-k) C_k.
```

The report returns each signed lag contribution, their sum, the diagonal
term and the residual against the existing conditional first/second-moment
recurrence. A nonzero residual raises an error rather than returning a report.
No infinite series or asymptotic limit is taken. A periodic chain can have
nondecaying covariances, such as `C_k=(-1)^k/4`; the finite identity still holds.

## An endpoint-conditioning check

Let the states be independent fair binary values and the reward be
`R_t = S_(t+1) - S_t`. Then `rho=0`, `C_0=1/2`, `C_1=-1/4`, and later
covariances vanish. The sum telescopes to `S_N-S_0`, whose variance is `1/2`
for every positive `N`. Replacing incoming edge rewards by source-state
conditional means would lose this negative lag-one contribution.

The stationary report included in the output retains the original model,
its rational bounds, all stationary/tree/Poisson checks and block moments.
Only `max_lag` is new: an integer `0..31`. All reported rational numbers use
reduced `[numerator, denominator]` pairs. No model fitting, empirical
calibration, summability, mixing-rate or confidence-interval claim is made.
