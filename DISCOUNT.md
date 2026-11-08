# Discounted values are not the stationary gradient proxy

This extension distinguishes three objects at the same anchored finite policy:

1. The true average-reward gradient `d rho` from the stationary audit.
2. A discounted stationary proxy `G_beta = mu dr + mu dP h_beta`.
3. The true gradient of a normalized discounted objective from a fixed starting state.

They need not agree for beta below one. A nonzero curl supplies an exact local obstruction to interpreting the proxy as a scalar-objective gradient. Zero curl by itself does not prove that a general vector field has a global potential.

## Centered discounted resolvent

Use the original model, stationary distribution `mu`, mean reward `rho`, transition matrix `P`, state expected reward `r`, and their parameter derivatives. Let `f=r-rho 1`. For `0<=beta<1`, solve

```text
(I-beta P) h_beta = f.
```

The inverse exists for a stochastic P in this range. Multiplying by `mu` gives `(1-beta) mu h_beta=mu f=0`, so the resolvent is centered. It differs from the usual discounted value by the constant `rho/(1-beta)`. Since every row of `dP` sums to zero, that constant disappears in `mu dP h_beta`.

At beta=1, do not invert the singular matrix `I-P`. Instead solve the centered Poisson matrix `I-P+1 mu`. The beta=1 row is an average-reward limit, not a finite discounted return evaluated at one. This is also valid for finite irreducible periodic chains: their nonconstant eigenvalues cannot equal one, even though some can have modulus one.

## Resolvent derivatives and exact curl

For coordinate i and beta below one:

```text
(I-beta P) dh_beta_i = dr_i - d_i rho 1 + beta (dP_i) h_beta.
```

At beta=1 the centered matrix itself changes with theta. Its derivative is `-dP_i+1 dmu_i`, so solve

```text
(I-P+1 mu) dh_i = dr_i-d_i rho 1+(dP_i)h-1(dmu_i h).
```

That final centering term matters. The audit returns every derivative residual and the differentiated centering identity `dmu_i h_beta + mu dh_beta_i=0`, including the endpoint.

Define the proxy component j as `G_j=mu dr_j+mu(dP_j)h_beta`. Its derivative contains symmetric second derivatives of `r` and `P`. They cancel from

```text
curl[i][j] = d_i G_j - d_j G_i.
```

It therefore suffices to compute

```text
C[i][j] = (dmu_i) dr_j
        + (dmu_i) (dP_j) h_beta
        + mu (dP_j) dh_beta_i
curl[i][j] = C[i][j] - C[j][i].
```

`C` is not the full Jacobian: the canceled symmetric terms are not returned. The output reports only the antisymmetric derivative matrix. A nonzero entry contradicts equality of mixed partial derivatives for a local scalar potential. At beta=1 the proxy equals `d rho`, and its curl is exactly zero by this gradient identity. Zero curl at other individual points is not a sufficient global-potential test.

## Fixed-start discounted objective

Let nu be the point mass at the supplied starting state. Hold nu fixed while differentiating. For beta below one:

```text
F_beta(theta) = (1-beta) nu (I-beta P)^-1 r
d_beta = (1-beta) nu (I-beta P)^-1
F_beta = d_beta r = rho+(1-beta) nu h_beta
```

Here `d_beta` is the normalized discounted occupancy, not the stationary distribution. Differentiation gives

```text
d_i F_beta = d_beta dr_i + beta d_beta (dP_i) h_beta
           = d_i rho + (1-beta) nu dh_beta_i.
```

Both forms are checked exactly. The implementation solves the transposed resolvent for `d_beta` and returns its full flow residual and normalization. At beta=0 the objective is the immediate reward at the fixed starting state. At beta=1 the row is the limit `rho` and gradient `d rho`, independent of that starting state. The audit does not differentiate a theta-dependent stationary initial distribution, and does not hold the stationary initial distribution fixed instead of nu.

The extra factor beta and the discounted occupancy both distinguish this gradient from the stationary proxy. Calling the proxy a discounted-objective gradient would silently change the objective.

## Exact two-coordinate counterexample

There are two states with rewards 0 and 1. Both actions initially have weight one. In state 0, action switch probabilities are `1/4,3/4`; action 1 has feature `[1,0]`. In state 1, action switch probabilities are `1/8,3/8`; action 1 has feature `[0,1]`. Action 0 has zero features in both states. Hence the controlled switch probabilities at the anchor are

```text
p=1/2, q=1/4, dp/dtheta_0=1/8, dq/dtheta_1=1/16.
```

The stationary distribution is `[1/3,2/3]`, `rho=2/3`, and `d rho=[1/18,-1/18]`. Let `u=p+q`, `D=1-beta+beta u`. For all beta in [0,1], the proxy is

```text
G_beta = [q dp/(u D), -p dq/(u D)].
curl[0][1] = -(1-beta) dp dq/(u D^2).
```

At beta=1/2 this gives `G=[1/21,-1/21]` and `curl[0][1]=-1/147`. Thus this proxy is not a scalar-objective gradient locally around the anchor. Starting from state 0, the actual normalized discounted objective is `2/7`, and its true gradient is `[5/98,-1/98]`, different from the proxy. All three vectors coincide only at the average-reward limit in this fixture's listed rows; no general monotone-bias claim is made.

## Bounds, verification and attribution

The existing model bounds remain unchanged. The API accepts at most eight distinct rational betas `[n,d]` with integers `0<=n<=d<=10000`. Input order is preserved; duplicates after reduction are refused. The starting state is a non-boolean in-range integer. The new version-one CLI payload has exactly `version`, `model`, `betas`, `initial_state`; existing audit/demo inputs and outputs are unchanged.

Independent Cramer/determinant references recompute stationary and resolvent values at nonzero floating theta. Central differences check both the true fixed-start objective gradient and the antisymmetric derivatives of the actual proxy, across three multi-state models, all their initial states and four beta values. The reference does not use production linear solves, tangent systems, Poisson derivatives or the curl formula. A separate 16-weight/four-beta grid checks the analytic two-state proxy and curl formulas. Tests cover beta=0, beta=1, 9999/10000, periodic chains, IID action rewards, transformations, residuals, isolation and strict validation.

[Greensmith, Bartlett and Baxter, JMLR 5 (2004), Section 3.2](https://jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf) explicitly distinguishes its discounted average-reward gradient approximation from a scalar-objective gradient, and its average-reward limit. That source treats the state-reward setting. The direct `mu dr` term, transition/action-reward implementation, centered derivative equations, exact local curl fixture, fixed-start comparison, tests and figure here are original implementations of standard differentiable finite-chain mathematics. ErgodLab is not GPOMDP, does not estimate an eligibility-trace gradient from data, and does not reproduce that paper's experiments. No optimizer, fitted policy, convergence rate, confidence interval, CLT or monotone-bias theorem is supplied.
