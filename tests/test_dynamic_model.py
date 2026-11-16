import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from pbgia_dynamic_normal_test import field, grad, neighbors, iterate

def test_constant_gradient_zero():
    assert np.max(np.abs(grad(np.ones((8,8,8),dtype=np.float32))))==0

def test_binary_only_initial_normalization():
    f,nu,n,band=field('torus',N=40)
    assert np.all(np.isin(f,[-1,1]))
    assert np.max(np.abs(np.linalg.norm(n,axis=-1)-1))<1e-5

def test_dynamic_normalization_and_topology_smoke():
    rows=iterate('torus','dynamic',h=2.,D=1.,N=40,steps=5,record=(0,5))
    assert rows[-1]['components']==1
    assert rows[-1]['euler_26']==0
    assert rows[-1]['dice']>.9
