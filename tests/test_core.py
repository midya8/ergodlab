import copy
from fractions import Fraction as F
from itertools import product,permutations
import math
import random
import unittest
from ergodlab import audit
from ergodlab.core import solve
from ergodlab.demo import occupancy_model,periodic_model,sticky_model


def q(x):return F(*x)
def det(A):
    n=len(A);out=0
    for p in permutations(range(n)):
        sign=(-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n));v=sign
        for i in range(n):v*=A[i][p[i]]
        out+=v
    return out


def enumerated(m,length,mu):
    n=len(mu);a=len(m['weights'][0]);out=[]
    def walk(s,left,p,total):
        if not left:out.append((p,total));return
        for b in range(a):
            for j in range(n):
                v=F(m['weights'][s][b],sum(m['weights'][s]))*q(m['transitions'][s][b][j])
                if v:walk(j,left-1,p*v,total+q(m['rewards'][s][b][j]))
    for s,p in enumerate(mu):walk(s,length,p,F(0))
    mean=sum(p*r for p,r in out);return mean,sum(p*(r-mean)**2 for p,r in out)


def float_reward(m,theta):
    n=len(m['weights']);na=len(m['weights'][0]);pi=[]
    for s in range(n):
        row=[m['weights'][s][b]*math.exp(sum(x*y for x,y in zip(m['features'][s][b],theta))) for b in range(na)];pi.append([v/sum(row) for v in row])
    P=[[sum(pi[s][b]*float(q(m['transitions'][s][b][j])) for b in range(na)) for j in range(n)] for s in range(n)]
    # Cofactors of I-P, not production's tree products or linear solve.
    weights=[det([[(int(i==j)-P[i][j]) for j in range(n) if j!=root] for i in range(n) if i!=root]) for root in range(n)]
    mu=[v/sum(weights) for v in weights]
    return sum(mu[s]*pi[s][b]*float(q(m['transitions'][s][b][j]))*float(q(m['rewards'][s][b][j])) for s in range(n) for b in range(na) for j in range(n))


class Core(unittest.TestCase):
    def test_occupancy_counterexample(self):
        a=audit(occupancy_model());self.assertEqual(a['stationary'],[[1,3],[2,3]]);self.assertEqual(a['stationary_derivatives'],[[[-1,18],[1,18]]]);self.assertEqual(a['gradient']['total'],[[1,18]]);self.assertEqual(a['gradient']['frozen_occupancy'],[[0,1]]);self.assertEqual(a['variance']['asymptotic_per_step'],[10,27])

    def test_twostate_cartesian_closed_forms(self):
        for p,z in product([F(1,8),F(1,4),F(1,2),F(3,4),F(1)],repeat=2):
            m=periodic_model();m['transitions']=[[[[*(1-p).as_integer_ratio()],[*p.as_integer_ratio()]]],[[[*z.as_integer_ratio()],[*(1-z).as_integer_ratio()]]]]
            a=audit(m,block_limit=3);mu=p/(p+z);v=mu*(1-mu);lam=1-p-z
            self.assertEqual(q(a['stationary'][1]),mu);self.assertEqual(q(a['variance']['asymptotic_per_step']),v*(1+lam)/(1-lam));self.assertEqual(a['graph']['period'],2 if p==z==1 else 1)

    def test_finite_block_independent_paths(self):
        for model in [occupancy_model(),sticky_model(),periodic_model()]:
            a=audit(model,block_limit=4)
            for b in a['variance']['stationary_blocks']:
                mean,var=enumerated(model,b['length'],list(map(q,a['stationary'])));self.assertEqual(mean,q(b['sum_mean']));self.assertEqual(var,q(b['sum_variance']))

    def test_periodic_odd_even(self):
        a=audit(periodic_model(),block_limit=32)
        self.assertFalse(a['graph']['aperiodic']);self.assertEqual(a['variance']['asymptotic_per_step'],[0,1])
        for b in a['variance']['stationary_blocks']:self.assertEqual(q(b['sum_variance']),F(b['length']%2,4))

    def test_threecycle_period(self):
        m={'weights':[[1]]*3,'features':[[[0]]]*3,'transitions':[[[[int(j==(s+1)%3),1] for j in range(3)]] for s in range(3)],'rewards':[[[[s,1] for _ in range(3)]] for s in range(3)]}
        a=audit(m);self.assertEqual(a['graph']['period'],3);self.assertEqual(a['variance']['asymptotic_per_step'],[0,1]);self.assertEqual(a['variance']['stationary_blocks'][2]['sum_variance'],[0,1])

    def test_independent_derivatives_and_determinants(self):
        for n in [2,3,4]:
            for seed in range(3):
                rng=random.Random(seed+n*10);na=3;dim=2
                m={'weights':[[rng.randrange(1,5) for _ in range(na)] for s in range(n)],'features':[[[rng.randrange(-2,3) for _ in range(dim)] for b in range(na)] for s in range(n)],'transitions':[[[[1,n] for j in range(n)] for b in range(na)] for s in range(n)],'rewards':[[[[rng.randrange(-2,4),2] for j in range(n)] for b in range(na)] for s in range(n)]}
                # Action- and state-dependent transitions, all strictly positive.
                for s,b in product(range(n),range(na)):
                    w=[rng.randrange(1,4) for _ in range(n)];total=sum(w);m['transitions'][s][b]=[[v,total] for v in w]
                a=audit(m,block_limit=2);P=[[q(v) for v in row] for row in a['controlled_transition']];cof=[det([[(int(i==j)-P[i][j]) for j in range(n) if j!=root] for i in range(n) if i!=root]) for root in range(n)]
                self.assertEqual([v/sum(cof) for v in cof],list(map(q,a['stationary'])))
                self.assertAlmostEqual(float_reward(m,[0,0]),float(q(a['average_reward'])),places=12)
                for k in range(dim):
                    plus=[0.,0.];minus=plus[:];plus[k]=1e-5;minus[k]=-1e-5;fd=(float_reward(m,plus)-float_reward(m,minus))/2e-5
                    self.assertAlmostEqual(fd,float(q(a['gradient']['total'][k])),places=9)
                for key in ['stationary','poisson']:self.assertTrue(all(q(v)==0 for v in a['residuals'][key]))

    def test_single_state_action_reward(self):
        m={'weights':[[1,3]],'features':[[[0],[1]]],'transitions':[[[[1,1]],[[1,1]]]],'rewards':[[[[0,1]],[[2,1]]]]}
        a=audit(m);self.assertEqual(a['gradient']['total'],[[3,8]]);self.assertEqual(a['average_reward'],[3,2]);self.assertEqual(a['variance']['asymptotic_per_step'],[3,4]);self.assertEqual(a['tree_reference']['root_tree_weights'],[[1,1]])
        for b in a['variance']['stationary_blocks']:self.assertEqual(q(b['variance_per_step']),F(3,4))

    def test_telescoping_edge_reward(self):
        m=occupancy_model()
        for s,b,j in product(range(2),range(2),range(2)):m['rewards'][s][b][j]=[j-s,1]
        a=audit(m);self.assertEqual(a['average_reward'],[0,1]);self.assertEqual(a['gradient']['total'],[[0,1]]);self.assertEqual(a['variance']['asymptotic_per_step'],[0,1]);self.assertGreater(q(a['variance']['one_step_stationary']),0)
        for edge in a['variance']['martingale_edges']:self.assertEqual(edge['martingale_increment'],[0,1])

    def test_shift_scale_reward(self):
        m=occupancy_model();old=audit(m)
        for s,b,j in product(range(2),range(2),range(2)):m['rewards'][s][b][j]=[3*q(m['rewards'][s][b][j])+2,1];m['rewards'][s][b][j][0]=int(m['rewards'][s][b][j][0])
        a=audit(m);self.assertEqual(q(a['average_reward']),3*q(old['average_reward'])+2);self.assertEqual(q(a['gradient']['total'][0]),3*q(old['gradient']['total'][0]));self.assertEqual(q(a['variance']['asymptotic_per_step']),9*q(old['variance']['asymptotic_per_step']));self.assertEqual([q(v) for v in a['poisson_bias']],[3*q(v) for v in old['poisson_bias']])

    def test_feature_shift_and_shared_parameter(self):
        m=occupancy_model();old=audit(m)
        for s,b in product(range(2),range(2)):m['features'][s][b]=[m['features'][s][b][0]+s+2]
        a=audit(m);self.assertEqual(a['scores'],old['scores']);self.assertEqual(a['gradient'],old['gradient'])

    def test_state_permutation(self):
        m=occupancy_model();old=audit(m);perm=[1,0]
        x={key:[copy.deepcopy(m[key][s]) for s in perm] for key in ['weights','features']}
        for key in ['transitions','rewards']:x[key]=[[[copy.deepcopy(m[key][s][b][j]) for j in perm] for b in range(2)] for s in perm]
        a=audit(x);self.assertEqual(a['stationary'],old['stationary'][::-1]);self.assertEqual(a['poisson_bias'],old['poisson_bias'][::-1]);self.assertEqual(a['gradient'],old['gradient']);self.assertEqual(a['variance']['asymptotic_per_step'],old['variance']['asymptotic_per_step'])

    def test_maximum_tree_budget(self):
        n=5;m={'weights':[[1]]*n,'features':[[[0]]]*n,'transitions':[[[[1,n]]*n] for _ in range(n)],'rewards':[[[[s,1]]*n] for s in range(n)]}
        a=audit(m,block_limit=1);self.assertEqual(a['tree_reference']['parent_assignments_checked'],1280);self.assertEqual(a['tree_reference']['positive_trees_by_root'],[125]*5);self.assertEqual(a['stationary'],[[1,5]]*5)

    def test_reducible_rejected(self):
        for transitions in [[[[[1,1],[0,1]]]]*2,[[[[1,1],[0,1]]],[[[0,1],[1,1]]]]]:
            m=periodic_model();m['transitions']=transitions
            with self.assertRaisesRegex(ValueError,'irreducible'):audit(m)

    def test_no_mutation_alias_rng(self):
        m=occupancy_model();before=copy.deepcopy(m);rng=random.getstate();a=audit(m);self.assertEqual(m,before);self.assertEqual(random.getstate(),rng);a['model']['weights'][0][0]=99;self.assertEqual(m,before)

    def test_linear_solver_pivot_and_singular(self):
        self.assertEqual(solve([[0,2],[1,1]],[4,3]),[1,2])
        with self.assertRaisesRegex(ValueError,'singular'):solve([[1,1],[2,2]],[1,2])

    def test_validation(self):
        cases=[('weights',[]),('weights',[[True,1],[1,1]]),('weights',[[0,1],[1,1]]),('features',None),('features',[[[0],[1]],[[0,1],[0]]]),('features',[[[0],[9]],[[0],[0]]]),('transitions',[]),('rewards',[[[[0,0]]]])]
        for key,value in cases:
            m=occupancy_model();m[key]=value
            with self.assertRaises(ValueError,msg=key):audit(m)
        for pair in [[True,1],[1,0],[33,1],[1,17],[-1,2],[2,1],None]:
            m=occupancy_model();m['transitions'][0][0][0]=pair
            with self.assertRaises(ValueError):audit(m)
        m=occupancy_model();m['transitions'][0][0][0]=[1,2]
        with self.assertRaisesRegex(ValueError,'sum'):audit(m)
        for value in [True,0,33,1.0]:
            with self.assertRaises(ValueError):audit(occupancy_model(),block_limit=value)
        m=occupancy_model();m['extra']=0
        with self.assertRaises(ValueError):audit(m)

    def test_exact_policy_gradient_cartesian(self):
        for w,u in product([1,2,3,4],repeat=2):
            m=occupancy_model();m['weights'][0]=[w,u];a=audit(m,block_limit=1);p=F(w+3*u,4*(w+u));q0=F(1,4);dp=F(w*u,2*(w+u)**2);truth=q0*dp/(p+q0)**2
            self.assertEqual(q(a['gradient']['total'][0]),truth);self.assertEqual(q(a['stationary_derivatives'][0][1]),truth)

    def test_zero_support_reward_inert(self):
        m=periodic_model();a=audit(m);m['rewards'][0][0][0]=[32,1];m['rewards'][1][0][1]=[-32,1];b=audit(m)
        self.assertEqual(a['variance'],b['variance']);self.assertEqual(a['gradient'],b['gradient']);self.assertEqual(a['average_reward'],b['average_reward'])

    def test_action_reorder_and_parameter_scaling(self):
        m=occupancy_model();a=audit(m)
        for key in m:
            for row in m[key]:row.reverse()
        b=audit(m);self.assertEqual(a['gradient'],b['gradient']);self.assertEqual(a['stationary'],b['stationary']);self.assertEqual(a['variance']['asymptotic_per_step'],b['variance']['asymptotic_per_step'])
        for s,row in enumerate(m['features']):
            for f in row:f[0]*=-2
        c=audit(m);self.assertEqual(q(c['gradient']['total'][0]),-2*q(a['gradient']['total'][0]))
