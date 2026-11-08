import argparse
import json
import sys
from .core import audit
from .demo import demo
from .discount import audit_discounted
from .discount_demo import discount_demo


def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate JSON key')
        out[k]=v
    return out


def main():
    p=argparse.ArgumentParser(description='Exact stationary policy and reward audit');sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('demo');sub.add_parser('discount-demo')
    a=sub.add_parser('audit');a.add_argument('input');a.add_argument('--block-limit',type=int,default=16)
    sub.add_parser('discount').add_argument('input')
    args=p.parse_args()
    try:
        if args.command=='demo':result=demo()
        elif args.command=='discount-demo':result=discount_demo()
        else:
            with open(args.input,'rb') as f:raw=f.read(65537)
            if len(raw)>65536:raise ValueError('input exceeds 64 KiB')
            payload=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON constant')))
            keys={'version','model'} if args.command=='audit' else {'version','model','betas','initial_state'}
            if type(payload) is not dict or set(payload)!=keys or type(payload['version']) is not int or payload['version']!=1:raise ValueError('input has invalid fields or version')
            if args.command=='discount' and payload['betas'] is None:raise ValueError('betas must be an array')
            result=audit(payload['model'],block_limit=args.block_limit) if args.command=='audit' else audit_discounted(payload['model'],betas=payload['betas'],initial_state=payload['initial_state'])
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError,TypeError,KeyError,RecursionError) as e:
        print(str(e),file=sys.stderr);return 2
    return 0


if __name__=='__main__':sys.exit(main())
