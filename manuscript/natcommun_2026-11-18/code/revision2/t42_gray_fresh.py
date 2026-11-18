"""Fresh, previously unseen test geometries (new seeds) for a single evaluation of the final gray readout."""
import sys,numpy as np,pandas as pd
sys.path[:0]=['/home/claude/work']
import t36_gray_test as T
from bench_shapes import FAM2
from multiprocessing import Pool
if __name__=='__main__':
    jobs=[]; seed=30261118
    for fam in FAM2:
        for i in range(30): seed+=1; jobs.append((fam,i,seed))
    with Pool(2) as pool: res=pool.map(T.job,jobs,chunksize=3)
    d=pd.DataFrame([r for rr in res for r in rr]); d.to_csv('data/t42_gray_fresh.csv',index=False)
