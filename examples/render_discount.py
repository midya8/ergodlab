"""Discounted proxy curl counterexample; all displayed entries remain exact."""
import argparse
import base64
import html
import json
from fractions import Fraction as F
from pathlib import Path
import zlib
from ergodlab.discount_demo import discount_demo


def render(data):
    if data!=discount_demo():raise ValueError('requires complete unchanged discount demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="760" viewBox="0 0 1120 760">','<rect width="1120" height="760" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata data-encoding="zlib+base64">'+base64.b64encode(zlib.compress(json.dumps(data,sort_keys=True,separators=(',',':')).encode())).decode()+'</metadata>']
    def text(x,y,s,size=17,color='#e7edf8',extra=''):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}" {extra}>'+html.escape(s)+'</text>')
    def frac(v):return str(F(*v))
    def vector(v):return '['+', '.join(frac(x) for x in v)+']'
    text(30,44,'ErgodLab | a discounted gradient proxy need not be a gradient',26)
    text(30,82,'Exact two-state, two-parameter counterexample. Both policies and transitions are known.',18)
    text(30,126,'Average-reward gradient: [1/18, -1/18]. Fixed starting state: 0.',21,'#63d8bc')
    text(30,163,'The stationary proxy and true normalized discounted-objective gradient are different quantities.',17,'#b7c4d9')
    columns=[42,170,465,820]
    for x,label in zip(columns,['Discount beta','Stationary proxy vector','Fixed-start gradient vector','Proxy curl[0][1]']):text(x,220,label,18)
    for i,row in enumerate(data['discounts']):
        y=269+56*i;out.append(f'<path d="M 30 {y-31} H 1090" stroke="#33445e"/>');vals=[frac(row['beta'])+(' (limit)' if row['is_average_reward_limit'] else ''),vector(row['stationary_gradient_proxy']),vector(row['fixed_start_objective_gradient']),frac(row['proxy_curl'][0][1])]
        for col,(x,value) in enumerate(zip(columns,vals)):text(x,y,value,19,'#63d8bc' if i==4 else '#e9bb79' if col==3 else '#e7edf8',f'data-row="{i}" data-col="{col}"')
    text(30,555,'At beta = 1/2, curl[0][1] = -1/147. A nonzero curl rules out a local scalar potential.',20,'#e9bb79')
    text(30,592,'Curl[i][j] = derivative_i(proxy_j) - derivative_j(proxy_i); it is not a full Jacobian.',17)
    text(30,628,'True objective: (1-beta) E_start[sum beta^t R_t]. The initial state is held fixed.',18)
    text(30,664,'At beta = 1, the row is the average-reward / centered-Poisson limit, not a discounted inverse.',17)
    text(30,702,'Zero curl alone does not prove a field is a gradient. No sampled estimator, optimizer or training claim.',16,'#b7c4d9')
    text(30,740,'Every table entry and the full exact resolvent/derivative/residual audit are embedded in SVG metadata.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();value=render(discount_demo());path=Path(__file__).resolve().parents[1]/'assets'/'discount-audit.svg'
    if args.check:
        if path.read_text()!=value:raise SystemExit('SVG differs')
    else:path.write_text(value)
