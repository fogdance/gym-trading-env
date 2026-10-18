#!/usr/bin/env python3
"""CPU-only WP03 diagnostic entry; no trainer or training authorization."""
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['JAX_PLATFORMS']='cpu'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/'src'),str(root.parent/'dreamerv3')]
from gym_trading_env.research.fg_s3_e_quality_checks import main

if __name__=='__main__':raise SystemExit(main())
