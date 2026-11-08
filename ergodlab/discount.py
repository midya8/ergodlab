"""Exact discounted values, stationary gradient proxies and local curl audits."""
from fractions import Fraction as F
from .core import audit,pack,solve


def beta_values(values):
    if type(values) is not list or not 1<=len(values)<=8:raise ValueError('betas must contain 1..8 rational pairs')
    out=[]
    for x in values:
        if type(x) is not list or len(x)!=2 or any(type(v) is not int for v in x) or not 1<=x[1]<=10000 or not 0<=x[0]<=x[1]:raise ValueError('beta must be [n,d], integers 0<=n<=d<=10000')
        b=F(*x)
        if b in out:raise ValueError('betas must be distinct after rational reduction')
        out.append(b)
    return out


def audit_discounted(model,*,betas=None,initial_state=0):
    bs=beta_values([[0,1],[1,2],[9,10],[99,100],[1,1]] if betas is None else betas)
    base=audit(model,block_limit=1);n=len(base['stationary']);dim=len(base['stationary_derivatives'])
    if type(initial_state) is not int or not 0<=initial_state<n:raise ValueError('initial_state must be in-range integer')
    def vec(v):return list(map(lambda x:F(*x),v))
    P=[vec(row) for row in base['controlled_transition']];dP=[[vec(row) for row in mat] for mat in base['transition_derivatives']];mu=vec(base['stationary']);dm=[vec(row) for row in base['stationary_derivatives']];r=vec(base['state_expected_rewards']);dr=[vec(row) for row in base['state_reward_derivatives']];rho=F(*base['average_reward']);grad=vec(base['gradient']['total']);f=[v-rho for v in r];rows=[]
    for b in bs:
        is_limit=b==1
        A=[[F(int(i==j))-b*P[i][j]+(mu[j] if is_limit else 0) for j in range(n)] for i in range(n)]
        h=solve(A,f)
        dh=[];dh_residual=[];centering_derivative=[]
        for k in range(dim):
            center=sum(dm[k][j]*h[j] for j in range(n)) if is_limit else F(0)
            rhs=[dr[k][s]-grad[k]+b*sum(dP[k][s][j]*h[j] for j in range(n))-center for s in range(n)]
            dh.append(solve(A,rhs))
            dh_residual.append([sum(A[s][j]*dh[k][j] for j in range(n))-rhs[s] for s in range(n)])
            centering_derivative.append(sum(dm[k][s]*h[s]+mu[s]*dh[k][s] for s in range(n)))
        direct=[sum(mu[s]*dr[k][s] for s in range(n)) for k in range(dim)]
        proxy=[direct[k]+sum(mu[s]*dP[k][s][j]*h[j] for s in range(n) for j in range(n)) for k in range(dim)]
        # Symmetric second derivatives cancel. Only first derivatives are needed.
        cross=[[sum(dm[i][s]*dr[j][s] for s in range(n))+sum(dm[i][s]*dP[j][s][u]*h[u]+mu[s]*dP[j][s][u]*dh[i][u] for s in range(n) for u in range(n)) for j in range(dim)] for i in range(dim)]
        curl=[[cross[i][j]-cross[j][i] for j in range(dim)] for i in range(dim)]
        if is_limit:
            occ=mu[:];value=rho;true=grad[:]
            assert proxy==grad and all(v==0 for row in curl for v in row)
        else:
            occ=solve([[F(int(i==j))-b*P[j][i] for j in range(n)] for i in range(n)],[(1-b)*int(i==initial_state) for i in range(n)])
            value=rho+(1-b)*h[initial_state]
            true=[sum(occ[s]*dr[k][s] for s in range(n))+b*sum(occ[s]*dP[k][s][j]*h[j] for s in range(n) for j in range(n)) for k in range(dim)]
            reference=[grad[k]+(1-b)*dh[k][initial_state] for k in range(dim)]
            assert true==reference and value==sum(occ[s]*r[s] for s in range(n))
        residual=[h[s]-b*sum(P[s][j]*h[j] for j in range(n))-f[s] for s in range(n)]
        occ_residual=[occ[j]-(1-b)*int(j==initial_state)-b*sum(occ[s]*P[s][j] for s in range(n)) for j in range(n)]
        assert not any(v for row in dh_residual for v in row) and not any(centering_derivative)
        assert sum(occ)==1 and all(v>=0 for v in occ) and not any(residual+occ_residual) and sum(mu[s]*h[s] for s in range(n))==0
        rows.append({'beta':b,'is_average_reward_limit':is_limit,'centered_resolvent':h,'centered_resolvent_derivatives':dh,'stationary_gradient_proxy':proxy,'proxy_bias_from_average_gradient':[proxy[k]-grad[k] for k in range(dim)],'proxy_curl':curl,'curl_is_nonzero':any(v!=0 for row in curl for v in row),'fixed_start_normalized_objective':value,'fixed_start_objective_gradient':true,'discounted_occupancy':occ,'residuals':{'centered_resolvent':residual,'discounted_occupancy':occ_residual,'centered_resolvent_derivatives':dh_residual,'differentiated_centering':centering_derivative}})
    return pack({'format':'ergodlab-discount-v1','stationary_audit':base,'initial_state':initial_state,'discounts':rows,'curl_orientation':'curl[i][j] = derivative_i(proxy_j) - derivative_j(proxy_i); not the full Jacobian; symmetric second-policy derivatives cancel','scope':'Proxy = mu dr + mu dP (I-beta P)^-1(r-rho) for beta<1, with centered Poisson limit at beta=1. It is not generally the gradient of a scalar objective. Fixed-start objective is (1-beta) E_start sum beta^t R_t with initial state held fixed while differentiating, not stationary initialization differentiated or held fixed. beta=1 reports the average-reward limit, not an invertible discounted resolvent or discounted sum at one. No sampled GPOMDP estimator, optimizer, mixing rate, CLT or monotone-bias guarantee.'})
