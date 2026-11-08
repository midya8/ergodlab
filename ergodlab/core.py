"""Finite rational stationary sensitivities and Markov-additive reward audits."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
from math import gcd


def pack(x):
    if isinstance(x,F):return [x.numerator,x.denominator]
    if isinstance(x,list):return [pack(v) for v in x]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    return x


def rational(x,prob=False):
    if type(x) is not list or len(x)!=2 or any(type(v) is not int for v in x) or not -32<=x[0]<=32 or not 1<=x[1]<=16:raise ValueError('rational must be [n,d], integers -32<=n<=32, 1<=d<=16')
    v=F(*x)
    if prob and not 0<=v<=1:raise ValueError('probability must be in [0,1]')
    return v


def validate(m,block_limit):
    if type(m) is not dict or set(m)!={'weights','features','transitions','rewards'}:raise ValueError('model requires exactly weights, features, transitions, rewards')
    w=m['weights'];f=m['features']
    if type(w) is not list or not 1<=len(w)<=5 or type(w[0]) is not list or not 1<=len(w[0])<=4:raise ValueError('weights require 1..5 states and 1..4 actions')
    n=len(w);a=len(w[0])
    if any(type(row) is not list or len(row)!=a or any(type(v) is not int or not 1<=v<=16 for v in row) for row in w):raise ValueError('weights must be rectangular integers in 1..16')
    if type(f) is not list or len(f)!=n or any(type(row) is not list or len(row)!=a for row in f) or type(f[0][0]) is not list or not 1<=len(f[0][0])<=3:raise ValueError('features require [states][actions][1..3 coordinates]')
    d=len(f[0][0])
    if any(type(v) is not list or len(v)!=d or any(type(c) is not int or not -8<=c<=8 for c in v) for row in f for v in row):raise ValueError('features must be bounded integers in -8..8 with fixed dimension')
    matrices=[]
    for name in ['transitions','rewards']:
        x=m[name]
        if type(x) is not list or len(x)!=n or any(type(row) is not list or len(row)!=a or any(type(v) is not list or len(v)!=n for v in row) for row in x):raise ValueError(name+' requires [states][actions][next states] rational pairs')
        matrices.append([[[rational(v,name=='transitions') for v in row] for row in s] for s in x])
    t,r=matrices
    if any(sum(row)!=1 for s in t for row in s):raise ValueError('transition rows must sum exactly to one')
    if type(block_limit) is not int or not 1<=block_limit<=32:raise ValueError('block_limit must be integer 1..32')
    return n,a,d,t,r


def solve(A,b):
    """Square exact Gauss-Jordan solve; rejects singular systems."""
    n=len(b);z=[[F(v) for v in row]+[F(b[i])] for i,row in enumerate(A)]
    for c in range(n):
        pivot=next((i for i in range(c,n) if z[i][c]),None)
        if pivot is None:raise ValueError('singular linear system')
        z[c],z[pivot]=z[pivot],z[c];v=z[c][c];z[c]=[x/v for x in z[c]]
        for i in range(n):
            if i!=c:
                v=z[i][c];z[i]=[x-v*y for x,y in zip(z[i],z[c])]
    return [row[-1] for row in z]


def graph(P):
    n=len(P);adj=[[j for j in range(n) if P[i][j]>0] for i in range(n)];reach=[]
    for root in range(n):
        seen={root};stack=[root]
        while stack:
            for j in adj[stack.pop()]:
                if j not in seen:seen.add(j);stack.append(j)
        reach.append(sorted(seen))
    if any(len(v)!=n for v in reach):raise ValueError('controlled chain must be irreducible; reducible chains have no supported audit')
    depth=[None]*n;depth[0]=0;stack=[0]
    while stack:
        i=stack.pop()
        for j in adj[i]:
            if depth[j] is None:depth[j]=depth[i]+1;stack.append(j)
    period=0
    for i in range(n):
        for j in adj[i]:period=gcd(period,abs(depth[i]+1-depth[j]))
    return {'adjacency':adj,'reachable':reach,'depth':depth,'period':period,'aperiodic':period==1}


def trees(P,dP):
    """Independent Markov-chain tree theorem, including product-rule tangents."""
    n=len(P);dim=len(dP);totals=[];tangents=[];counts=[];candidate=0
    for root in range(n):
        nodes=[i for i in range(n) if i!=root];weight=F(0);dw=[F(0)]*dim;count=0
        for parents in product(*[[j for j in range(n) if j!=i] for i in nodes]):
            candidate+=1;parent=dict(zip(nodes,parents));valid=True
            for start in nodes:
                cur=start;seen=set()
                while cur!=root:
                    if cur in seen:valid=False;break
                    seen.add(cur);cur=parent[cur]
                if not valid:break
            if not valid:continue
            edges=list(parent.items());factors=[P[i][j] for i,j in edges];v=F(1)
            for p in factors:v*=p
            weight+=v;count+=bool(v)
            for k in range(dim):
                for e,(i,j) in enumerate(edges):
                    term=dP[k][i][j]
                    for q,p in enumerate(factors):
                        if q!=e:term*=p
                    dw[k]+=term
        totals.append(weight);tangents.append(dw);counts.append(count)
    total=sum(totals);mu=[v/total for v in totals];dt=[sum(row[k] for row in tangents) for k in range(dim)]
    dmu=[[(tangents[i][k]*total-totals[i]*dt[k])/total**2 for i in range(n)] for k in range(dim)]
    return mu,dmu,{'root_tree_weights':totals,'positive_trees_by_root':counts,'parent_assignments_checked':candidate,'stationary':mu,'stationary_derivatives':dmu}


def audit(model,*,block_limit=16):
    n,a,d,t,r=validate(model,block_limit)
    pi=[[F(v,sum(row)) for v in row] for row in model['weights']]
    z=[[[F(model['features'][s][b][k])-sum(pi[s][c]*model['features'][s][c][k] for c in range(a)) for k in range(d)] for b in range(a)] for s in range(n)]
    P=[[sum(pi[s][b]*t[s][b][j] for b in range(a)) for j in range(n)] for s in range(n)]
    g=graph(P)
    dP=[[[sum(pi[s][b]*z[s][b][k]*t[s][b][j] for b in range(a)) for j in range(n)] for s in range(n)] for k in range(d)]
    A=[[P[j][i]-int(i==j) for j in range(n)] for i in range(n-1)]+[[F(1)]*n]
    mu=solve(A,[F(0)]*(n-1)+[F(1)])
    dm=[solve(A,[-sum(mu[s]*dP[k][s][j] for s in range(n)) for j in range(n-1)]+[F(0)]) for k in range(d)]
    mt,dt,tree=trees(P,dP);assert mu==mt and dm==dt
    rb=[sum(pi[s][b]*t[s][b][j]*r[s][b][j] for b in range(a) for j in range(n)) for s in range(n)]
    dr=[[sum(pi[s][b]*z[s][b][k]*t[s][b][j]*r[s][b][j] for b in range(a) for j in range(n)) for s in range(n)] for k in range(d)]
    rho=sum(mu[s]*rb[s] for s in range(n))
    H=[[F(int(i==j))-P[i][j]+mu[j] for j in range(n)] for i in range(n)]
    h=solve(H,[v-rho for v in rb]);assert sum(mu[i]*h[i] for i in range(n))==0
    direct=[sum(mu[s]*dr[k][s] for s in range(n)) for k in range(d)]
    occupancy=[sum(dm[k][s]*rb[s] for s in range(n)) for k in range(d)]
    gradient=[direct[k]+occupancy[k] for k in range(d)]
    poisson=[sum(mu[s]*pi[s][b]*t[s][b][j]*z[s][b][k]*(r[s][b][j]+h[j]) for s in range(n) for b in range(a) for j in range(n)) for k in range(d)]
    assert gradient==poisson
    mart=[];sigma=F(0);onevar=F(0)
    for s in range(n):
        conditional=F(0)
        for b in range(a):
            for j in range(n):
                if not t[s][b][j]:continue
                p=mu[s]*pi[s][b]*t[s][b][j];v=r[s][b][j]-rho+h[j]-h[s]
                conditional+=pi[s][b]*t[s][b][j]*v;sigma+=p*v*v;onevar+=p*(r[s][b][j]-rho)**2
                mart.append({'state':s,'action':b,'next_state':j,'stationary_probability':p,'reward':r[s][b][j],'martingale_increment':v,'variance_contribution':p*v*v})
        assert conditional==0
    # Exact conditional Markov-additive first/second moments, no trajectory sampling.
    first=[F(0)]*n;second=[F(0)]*n;blocks=[]
    for length in range(1,block_limit+1):
        nf=[sum(pi[s][b]*t[s][b][j]*(r[s][b][j]+first[j]) for b in range(a) for j in range(n)) for s in range(n)]
        ns=[sum(pi[s][b]*t[s][b][j]*(r[s][b][j]**2+2*r[s][b][j]*first[j]+second[j]) for b in range(a) for j in range(n)) for s in range(n)]
        first,second=nf,ns;mean=sum(mu[s]*first[s] for s in range(n));var=sum(mu[s]*second[s] for s in range(n))-mean**2
        assert mean==length*rho and var>=0
        blocks.append({'length':length,'sum_mean':mean,'sum_variance':var,'variance_per_step':var/length,'difference_from_asymptotic':var/length-sigma})
    stationary_residual=[sum(mu[s]*P[s][j] for s in range(n))-mu[j] for j in range(n)]
    tangent_residual=[[(sum(dm[k][s]*P[s][j]+mu[s]*dP[k][s][j] for s in range(n))-dm[k][j]) for j in range(n)] for k in range(d)]
    bias_residual=[h[s]-sum(P[s][j]*h[j] for j in range(n))-rb[s]+rho for s in range(n)]
    assert not any(stationary_residual+bias_residual) and not any(v for row in tangent_residual for v in row)
    return pack({'format':'ergodlab-audit-v1','model':deepcopy(model),'policy':pi,'scores':z,'controlled_transition':P,'transition_derivatives':dP,'graph':g,'stationary':mu,'stationary_derivatives':dm,'average_reward':rho,'state_expected_rewards':rb,'state_reward_derivatives':dr,'poisson_bias':h,'gradient':{'total':gradient,'frozen_occupancy':direct,'occupancy_term':occupancy,'poisson_reference':poisson},'variance':{'asymptotic_per_step':sigma,'one_step_stationary':onevar,'martingale_edges':mart,'stationary_blocks':blocks},'tree_reference':tree,'residuals':{'stationary':stationary_residual,'stationary_derivatives':tangent_residual,'poisson':bias_residual},'scope':'Exact supplied finite irreducible chain at theta=0; pi proportional to weight*exp(feature dot theta); fixed transitions and rewards; periodic chains supported but no pointwise mixing claim; scalar bounded Markov-additive reward, stationary finite-block variances and asymptotic variance of centered reward sums, not variance of a policy-gradient estimator, fitted policy, simulation, CLT, inference interval or generalization guarantee.'})
