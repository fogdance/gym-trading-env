#!/usr/bin/env python3
"""CPU-only Corpus entry; no formal training authorization is issued."""
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/'src'),str(root.parent/'dreamerv3')]
from gym_trading_env.research.fg_s3_e_quality_corpus_cli import main
if __name__=='__main__':raise SystemExit(main())
