"""Exact-demo SVG with full audit metadata and checked axis geometry."""
import argparse
import base64
import html
import json
from fractions import Fraction as F
from pathlib import Path
import zlib
from ergodlab.demo import demo


def render(data):
    if data!=demo():raise ValueError('requires complete unchanged demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="760" viewBox="0 0 1120 760">','<rect width="1120" height="760" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata data-encoding="zlib+base64">'+base64.b64encode(zlib.compress(json.dumps(data,sort_keys=True,separators=(',',':')).encode())).decode()+'</metadata>']
    def text(x,y,s,size=17,color='#e7edf8'):
        out.append(f'<text x="{float(x):.8f}" y="{float(y):.8f}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(30,44,'ErgodLab | the one-step reward misses the long run',27)
    text(30,82,'Exact stationary policy gradients and Markov-additive reward variance. No sampling.',18)
    text(40,131,'Equal one-step variance, different temporal dependence',21)
    text(40,163,'Stationary Var(reward) = 1/4 in both two-state chains. State rewards: 0 and 1.',17,'#b7c4d9')
    text(40,197,'Vertical: Var(sum of N rewards) / N. Horizontal: block length N.',16,'#b7c4d9')
    for v in [F(0),F(1,2),F(1),F(3,2),F(7,4)]:
        y=495-160*v;out.append(f'<path d="M 75 {float(y):.8f} H 680" stroke="#33445e"/>');text(26,y+5,f'{float(v):g}',14,'#b7c4d9')
    for length in [1,4,8,12,16]:text(75+39*(length-1)-4,522,str(length),14,'#b7c4d9')
    for name,color in [('sticky','#e9bb79'),('periodic','#63d8bc')]:
        pts=[]
        for b in data[name]['variance']['stationary_blocks']:
            x=75+39*(b['length']-1);y=495-160*F(*b['variance_per_step']);pts.append(f'{x},{float(y):.8f}')
            out.append(f'<circle data-fixture="{name}" data-length="{b["length"]}" cx="{x}" cy="{float(y):.8f}" r="3.5" fill="{color}"/>')
        out.append(f'<polyline points="'+ ' '.join(pts)+f'" fill="none" stroke="{color}" stroke-width="2"/>')
        val=F(*data[name]['variance']['asymptotic_per_step']);y=495-160*val
        out.append(f'<path data-limit="{name}" d="M 75 {float(y):.8f} H 680" stroke="{color}" stroke-dasharray="6 5"/>')
    text(725,244,'Sticky chain',21,'#e9bb79');text(725,277,'P(stay) = 7/8',18);text(725,310,'Asymptotic variance = 7/4',18)
    text(725,365,'Alternating chain',21,'#63d8bc');text(725,398,'P(switch) = 1; period = 2',18);text(725,431,'Asymptotic variance = 0',18)
    text(725,484,'Dashed lines are exact limits.',16,'#b7c4d9');text(725,514,'Not mixing-rate or CLT claims.',16,'#b7c4d9')
    text(30,571,'A policy changes which states it visits, even when immediate rewards do not change',21)
    text(30,608,'Separate occupancy fixture: stationary distribution = [1/3, 2/3]; mean reward = 2/3.',17)
    text(30,641,'Frozen-occupancy gradient: 0. True average-reward gradient: 1/18.',21,'#63d8bc')
    text(30,676,'Stationary linear-system derivatives = directed spanning-tree derivatives = Poisson-bias gradient.',17)
    text(30,710,'Supplied finite irreducible chains only. Reward-sum variance is not policy-gradient estimator variance.',16,'#b7c4d9')
    text(30,744,'Complete exact audit JSON is embedded in compressed SVG metadata; displayed lines are not simulations.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();value=render(demo());path=Path(__file__).resolve().parents[1]/'assets'/'stationary-audit.svg'
    if args.check:
        if path.read_text()!=value:raise SystemExit('SVG differs')
    else:path.write_text(value)
