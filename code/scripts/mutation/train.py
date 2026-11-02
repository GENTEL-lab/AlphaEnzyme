#!/usr/bin/env python3
"""Train one component of the fixed Fig.3 mutation recipe."""
import argparse,subprocess,sys
from pathlib import Path
from common import DEFAULT_INPUTS,SEEDS
from score_library import COMPONENTS
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component',required=True,choices=COMPONENTS)
    p.add_argument('--fold',required=True,type=int,choices=range(8))
    p.add_argument('--seed',type=int,choices=SEEDS,default=1234)
    p.add_argument('--inputs',type=Path,default=DEFAULT_INPUTS)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--rounds',type=int,default=300)
    a=p.parse_args()
    base=['--fold',str(a.fold),'--inputs',str(a.inputs.resolve()),'--output',str(a.output.resolve())]
    if a.component=='original':
        script='train_original.py';extra=['--seed',str(a.seed),'--epochs',str(a.epochs)]
    elif a.component.endswith('_ridge'):
        script='train_ridge.py';extra=['--family','global' if a.component.startswith('g_') else 'local']
    else:
        script='train_stochastic.py';extra=['--config',a.component[2:],'--seed',str(a.seed),'--epochs',str(a.epochs),'--rounds',str(a.rounds)]
    subprocess.run([sys.executable,'-B',str(HERE/script),*base,*extra],check=True)
if __name__=='__main__':main()
