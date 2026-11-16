import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from fixed_iteration_checkpoints import (synthetic_mask,initialize,one_step,generate,gradient)

def test_shapes_and_unit_norm():
    for dim in (2,3):
        f=synthetic_mask(dim,20)
        n=initialize(f)
        assert n.shape==f.shape+(dim+1,)
        out=one_step(n,f)
        assert np.max(abs(np.linalg.norm(out,axis=-1)-1))<1e-5
        assert np.all(np.isin(f,[-1.,1.]))

def test_checkpoints_equal_individual_continuation():
    f=synthetic_mask(2,28)
    snaps,rows=generate(f,checkpoints=(5,10,20))
    n=initialize(f)
    for t in range(1,21):
        n=one_step(n,f)
        if t in snaps: assert np.array_equal(n,snaps[t])
    assert len(rows)==3

def test_derivative_moments_interior():
    xy=np.indices((25,25),dtype=np.float32)
    assert np.max(np.abs(gradient(np.ones((25,25),dtype=np.float32))[2:-2,2:-2]))==0
    grad=gradient(xy[0]);assert np.allclose(grad[2:-2,2:-2,0],1)

def test_zero_field_retains_spin():
    f=synthetic_mask(2,12);n=initialize(f)
    assert np.array_equal(one_step(n,f,J=0,h=0,D=0),n)
