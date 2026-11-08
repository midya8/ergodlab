from copy import deepcopy
from fractions import Fraction as F
import json
import math
from itertools import product
import random
import subprocess
import sys
import tempfile
import unittest
from ergodlab import audit,audit_discounted
from ergodlab.demo import occupancy_model,periodic_model
from ergodlab.discount_demo import curl_model,discount_demo
from test_core import det


def q(v):return F(*v)
def cramer(A,b):
    determinant=det(A);out=[]
    for col in range(len(b)):
        B=[row[:] for row in A]
        for i in range(len(b)):B[i][col]=b[i]
        out.append(det(B)/determinant)
    return out


def reference(m,theta,beta,start):
    n=len(m['weights']);na=len(m['weights'][0]);dim=len(theta);pi=[];score=[]
    for s in range(n):
        w=[m['weights'][s][a]*math.exp(sum(x*y for x,y in zip(m['features'][s][a],theta))) for a in range(na)];p=[v/sum(w) for v in w];pi.append(p);score.append([[m['features'][s][a][k]-sum(p[c]*m['features'][s][c][k] for c in range(na)) for k in range(dim)] for a in range(na)])
    P=[[sum(pi[s][a]*float(q(m['transitions'][s][a][j])) for a in range(na)) for j in range(n)] for s in range(n)]
    r=[sum(pi[s][a]*float(q(m['transitions'][s][a][j]))*float(q(m['rewards'][s][a][j])) for a in range(na) for j in range(n)) for s in range(n)]
    weights=[det([[(int(i==j)-P[i][j]) for j in range(n) if j!=root] for i in range(n) if i!=root]) for root in range(n)];mu=[v/sum(weights) for v in weights];rho=sum(x*y for x,y in zip(mu,r))
    A=[[int(i==j)-beta*P[i][j]+(mu[j] if beta==1 else 0) for j in range(n)] for i in range(n)];h=cramer(A,[v-rho for v in r])
    proxy=[sum(mu[s]*pi[s][a]*float(q(m['transitions'][s][a][j]))*score[s][a][k]*(float(q(m['rewards'][s][a][j]))+h[j]) for s in range(n) for a in range(na) for j in range(n)) for k in range(dim)]
    return rho+(1-beta)*h[start],proxy


class Discount(unittest.TestCase):
    def test_exact_curl_fixture(self):
        d=discount_demo();row=d['discounts'][1]
        self.assertEqual(row['proxy_curl'][0][1],[-1,147]);self.assertTrue(row['curl_is_nonzero']);self.assertEqual(row['fixed_start_objective_gradient'],[[5,98],[-1,98]]);self.assertNotEqual(row['stationary_gradient_proxy'],row['fixed_start_objective_gradient'])

    def test_two_state_exact_grid(self):
        for w,u in product([1,2,3,4],repeat=2):
            m=curl_model();m['weights'][0]=[w,u];d=audit_discounted(m,betas=[[0,1],[1,2],[9,10],[1,1]]);p=F(w+3*u,4*(w+u));z=F(1,4);dp=F(w*u,2*(w+u)**2);dq=F(1,16);total=p+z
            for row in d['discounts']:
                b=q(row['beta']);den=1-b+b*total
                self.assertEqual([q(v) for v in row['stationary_gradient_proxy']],[z*dp/(total*den),-p*dq/(total*den)])
                self.assertEqual(q(row['proxy_curl'][0][1]),-(1-b)*dp*dq/(total*den**2))
                for i,j in product(range(2),repeat=2):self.assertEqual(q(row['proxy_curl'][i][j]),-q(row['proxy_curl'][j][i]))

    def test_independent_value_gradient_and_curl(self):
        models=[curl_model()]
        for n in [2,3]:
            rng=random.Random(n+81);na=2
            m={'weights':[[1,2] for _ in range(n)],'features':[[[rng.randrange(-2,3) for k in range(2)] for a in range(na)] for s in range(n)],'transitions':[[[[1,n] for j in range(n)] for a in range(na)] for s in range(n)],'rewards':[[[[rng.randrange(-3,4),2] for j in range(n)] for a in range(na)] for s in range(n)]}
            for s,a in product(range(n),range(na)):
                weights=[rng.randrange(1,4) for j in range(n)];m['transitions'][s][a]=[[v,sum(weights)] for v in weights]
            models.append(m)
        for m in models:
            for start in range(len(m['weights'])):
                d=audit_discounted(m,betas=[[0,1],[1,2],[9,10],[1,1]],initial_state=start)
                for row in d['discounts']:
                    b=float(q(row['beta']));value,proxy=reference(m,[0.,0.],b,start);self.assertAlmostEqual(value,float(q(row['fixed_start_normalized_objective'])),places=11)
                    for k in range(2):self.assertAlmostEqual(proxy[k],float(q(row['stationary_gradient_proxy'][k])),places=11)
                    jac=[]
                    for i in range(2):
                        plus=[0.,0.];minus=plus[:];plus[i]=1e-5;minus[i]=-1e-5;vp,gp=reference(m,plus,b,start);vm,gm=reference(m,minus,b,start);fd=(vp-vm)/2e-5
                        self.assertAlmostEqual(fd,float(q(row['fixed_start_objective_gradient'][i])),places=8);jac.append([(gp[j]-gm[j])/2e-5 for j in range(2)])
                    for i,j in product(range(2),repeat=2):self.assertAlmostEqual(jac[i][j]-jac[j][i],float(q(row['proxy_curl'][i][j])),places=8)

    def test_limit_not_singular_resolvent(self):
        d=audit_discounted(curl_model(),betas=[[9999,10000],[1,1]]);b=d['discounts'][-1];self.assertTrue(b['is_average_reward_limit']);self.assertFalse(b['curl_is_nonzero']);self.assertEqual(b['centered_resolvent'],d['stationary_audit']['poisson_bias']);self.assertEqual(b['stationary_gradient_proxy'],d['stationary_audit']['gradient']['total']);self.assertEqual(b['fixed_start_normalized_objective'],d['stationary_audit']['average_reward']);self.assertEqual(b['discounted_occupancy'],d['stationary_audit']['stationary'])

    def test_beta_zero_and_initial_state(self):
        m=curl_model();zero=audit_discounted(m,betas=[[0,1]],initial_state=0)['discounts'][0];one=audit_discounted(m,betas=[[0,1]],initial_state=1)['discounts'][0]
        self.assertEqual(zero['fixed_start_normalized_objective'],[0,1]);self.assertEqual(one['fixed_start_normalized_objective'],[1,1]);self.assertEqual(zero['discounted_occupancy'],[[1,1],[0,1]]);self.assertEqual(zero['stationary_gradient_proxy'],one['stationary_gradient_proxy']);self.assertEqual(zero['fixed_start_objective_gradient'],[[0,1],[0,1]])

    def test_single_state_direct_reward(self):
        m={'weights':[[1,3]],'features':[[[0,0],[1,2]]],'transitions':[[[[1,1]],[[1,1]]]],'rewards':[[[[0,1]],[[2,1]]]]}
        for row in audit_discounted(m)['discounts']:
            self.assertEqual(row['centered_resolvent'],[[0,1]]);self.assertEqual(row['stationary_gradient_proxy'],[[3,8],[3,4]]);self.assertEqual(row['fixed_start_objective_gradient'],[[3,8],[3,4]]);self.assertFalse(row['curl_is_nonzero'])

    def test_periodic_singular_endpoints(self):
        m=periodic_model();d=audit_discounted(m)
        for row in d['discounts']:
            b=q(row['beta']);self.assertEqual(q(row['fixed_start_normalized_objective']),b/(1+b));self.assertEqual([q(v) for v in row['centered_resolvent']],[-F(1,2)/(1+b),F(1,2)/(1+b)]);self.assertFalse(row['curl_is_nonzero'])

    def test_reward_shift_scale(self):
        m=curl_model();before=audit_discounted(m)
        for s,a,j in product(range(2),repeat=3):m['rewards'][s][a][j]=[3*m['rewards'][s][a][j][0]+2,1]
        after=audit_discounted(m)
        for old,new in zip(before['discounts'],after['discounts']):
            self.assertEqual(q(new['fixed_start_normalized_objective']),3*q(old['fixed_start_normalized_objective'])+2)
            for key in ['stationary_gradient_proxy','fixed_start_objective_gradient','proxy_bias_from_average_gradient','centered_resolvent']:self.assertEqual([q(v) for v in new[key]],[3*q(v) for v in old[key]])
            self.assertEqual(q(new['proxy_curl'][0][1]),3*q(old['proxy_curl'][0][1]))

    def test_validation_and_isolation(self):
        for value in [[],[[True,1]],[[1,0]],[[2,1]],[[1,10001]],[[1,2],[2,4]],[0],[[0,1]]*9]:
            with self.assertRaises(ValueError):audit_discounted(curl_model(),betas=value)
        for start in [True,-1,2,1.0]:
            with self.assertRaises(ValueError):audit_discounted(curl_model(),initial_state=start)
        m=curl_model();old=deepcopy(m);rng=random.getstate();d=audit_discounted(m);self.assertEqual(m,old);self.assertEqual(rng,random.getstate());d['stationary_audit']['model']['weights'][0][0]=99;self.assertEqual(m,old)
        for row in d['discounts']:
            for key,vals in row['residuals'].items():
                if key=='centered_resolvent_derivatives':self.assertTrue(all(q(v)==0 for values in vals for v in values))
                else:self.assertTrue(all(q(v)==0 for v in vals))

    def test_cli(self):
        p=subprocess.run([sys.executable,'-m','ergodlab','discount-demo'],text=True,capture_output=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),discount_demo())
        p=subprocess.run([sys.executable,'-m','ergodlab','discount','examples/discount.json'],text=True,capture_output=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),discount_demo())
        for patch in [{'extra':1},{'initial_state':True},{'version':True},{'betas':[[1,0]]},{'betas':None}]:
            payload={'version':1,'model':curl_model(),'initial_state':0,'betas':[[0,1]]};payload.update(patch)
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
                json.dump(payload,f);f.flush();p=subprocess.run([sys.executable,'-m','ergodlab','discount',f.name],text=True,capture_output=True);self.assertEqual(p.returncode,2);self.assertFalse(p.stdout)
