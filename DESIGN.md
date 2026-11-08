# Finite stationary audit

All vectors below use row orientation for distributions and column orientation for state functions. All production computations use rational arithmetic. Policy weights/features, transition probabilities and rewards are supplied, not inferred.

## Anchored policy and controlled chain

At `theta=0`, `pi(a|s)=w(s,a)/sum_b w(s,b)`. For coordinate `k`, the score is `z_k(s,a)=phi_k(s,a)-sum_b pi(b|s)phi_k(s,b)`. Thus `d_k pi=pi*z_k`, with each score mean zero.

The supplied transition `T(s,a,j)` and reward `R(s,a,j)` are independent of theta. Define:

```text
P(s,j) = sum_a pi(a|s) T(s,a,j)
dP_k(s,j) = sum_a pi(a|s) z_k(s,a) T(s,a,j)
r(s) = sum_a,j pi(a|s) T(s,a,j) R(s,a,j)
dr_k(s) = sum_a,j pi(a|s) z_k(s,a) T(s,a,j) R(s,a,j)
```

Weights are positive at every action, so the support graph is unchanged near the anchor. Strong connectivity is checked by reachability from each state. Reducible chains are outside scope even if a particular reducible chain has a unique stationary distribution.

The period is the gcd of `depth(s)+1-depth(j)` over all positive directed edges, using a rooted spanning search depth. In a strongly connected finite graph this equals the gcd of cycle lengths. Depth need not be a shortest-path distance. A self-loop gives period one. Period does not alter the stationary equations; the audit never claims pointwise mixing for periodic chains.

## Stationary sensitivity

Irreducibility gives a unique positive row distribution `mu` with `mu P=mu`, `mu 1=1`. Replace the final equation of `(P^T-I)mu^T=0` by normalization and solve by exact pivoted Gauss-Jordan elimination. Any one of the redundant stationary equations can be removed in an irreducible chain.

Differentiating the stationary equations:

```text
dmu_k (P-I) = -mu dP_k
dmu_k 1 = 0
```

The same normalized matrix solves this tangent system, now with normalization right-hand side zero. Full unmodified stationary and tangent residuals, including the removed equations, are returned and checked exactly.

For an independent reference, enumerate every rooted directed spanning tree. Every nonroot selects a distinct nonself parent candidate; a candidate is a tree iff each node reaches the root without cycling. Its weight is the product of `P(i,parent(i))`. Tree weights with impossible edges vanish. For root `j`, let `tau_j` be the sum. The Markov-chain tree theorem gives `mu_j=tau_j/sum_i tau_i`.

Differentiate every product with the product rule, without `dP/P`, then apply the quotient rule to the root sums. This independently checks every stationary derivative against the linear system. The product rule remains defined when an edge has zero probability. One-state chains use the empty product 1. For n=5, all roots together have `5*4^4=1280` parent assignments; the complete positive graph has `5^(5-2)=125` trees per root.

## Poisson bias and average-reward gradient

Let `rho=mu r`. Solve the centered Poisson equation:

```text
(I-P+1 mu) h = r-rho 1
```

Multiplication by `mu` gives `mu h=0`; hence this also solves `(I-P)h=r-rho 1`. The matrix is invertible for a finite irreducible chain, including periodic chains. The centering chooses one representative from the bias functions differing by a constant.

The derivative is `d rho=(dmu)r + mu dr`. Using `dmu 1=0`, the Poisson equation and the differentiated stationary relation:

```text
(dmu)r = (dmu)(I-P)h = mu (dP)h
d rho = mu dr + mu (dP)h
      = E_mu,pi,T [z(s,a) (R(s,a,j)+h(j))]
```

The implementation returns both evaluations and the split into direct/frozen-occupancy and occupancy terms. Adding an action-independent state baseline inside the final score expectation would not change its mean, but no score-estimator covariance or variance-optimal baseline is claimed here.

For the occupancy fixture, let `p(theta)` denote the state-0 switch probability and `q=1/4` the state-1 switch probability. At the anchor `p=1/2`, `dp=1/8`; therefore `mu_1=p/(p+q)=2/3` and `dmu_1=q dp/(p+q)^2=1/18`. Since `r=[0,1]` is policy-independent, the entire gradient comes from occupancy. A frozen-state distribution loses it.

## Exact long-run reward-sum variance

On each positive stationary edge define:

```text
D(s,a,j) = R(s,a,j)-rho+h(j)-h(s)
```

Conditioning on the history immediately before choosing the action in state `s`, the mean of `D` is `r(s)-rho+(Ph)(s)-h(s)=0`. Successive increments are martingale differences, and consequently uncorrelated. Under a stationary start their second moments are equal:

```text
sigma^2 = sum_s,a,j mu(s) pi(a|s) T(s,a,j) D(s,a,j)^2
```

Every term is nonnegative and returned. The centered reward sum `S_N=sum_t(R_t-rho)` satisfies

```text
S_N = sum_t D_t + h(X_0)-h(X_N).
```

Since the state space is finite, the boundary term is bounded. The martingale sum has variance `N sigma^2`. The boundary variance divided by N vanishes and its covariance with the martingale sum divided by N vanishes by Cauchy-Schwarz. Thus `Var(S_N)/N -> sigma^2`. This argument does not require aperiodicity, convergence of a covariance series, reversibility or a CLT. It is a stationary variance limit, not an interval, rate, mixing certificate or gradient-estimator variance.

A telescoping edge reward `R(s,a,j)=g(j)-g(s)` has `rho=0`, bias `h=-g+mu g` and every `D=0`. Its reward-sum variance is only an endpoint effect even when its one-step variance is positive. The tests include this edge-reward case, not only state rewards.

## Finite-block moment reference

Starting from a fixed state, let `m_N(s)` and `v_N(s)` denote the first and raw second moments of the uncentered N-step reward sum. Initialize both to zero at N=0, then:

```text
m_(N+1)(s) = sum_a,j pi T [R + m_N(j)]
v_(N+1)(s) = sum_a,j pi T [R^2 + 2 R m_N(j) + v_N(j)]
```

Mix the resulting moments with `mu`. The stationary mean is `N rho`, and the variance is `mu v_N-(N rho)^2`. These exact finite-block variances need not equal `N sigma^2`, need not be monotone after dividing by N, and do not imply a finite-N normal approximation. Independent explicit path enumeration in the tests checks all three fixture moment tables through length four.

For a symmetric two-state chain with switch probability `p`, state reward 0/1 and `lambda=1-2p`, the stationary reward variance is 1/4 and `sigma^2=(1/4)(1+lambda)/(1-lambda)`. The sticky case p=1/8 gives 7/4. The alternating endpoint p=1 gives zero; finite blocks have variance 0 for even N and 1/4 for odd N. These are different supplied chains, not an empirical variance-reduction experiment.

## Attribution

The tree theorem, Poisson equation, stationary sensitivity and martingale framework are standard. See the sourced references in [README.md](README.md). This finite-state derivation, edge-reward implementation, audit schema, chosen fixtures, independent oracles and figure were written for ErgodLab. It is not a novel algorithm, GPOMDP implementation, reproduction of a source experiment or claim about fitted policies.
