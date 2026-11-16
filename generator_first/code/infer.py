"""Frozen exact-N inference, no input-quality or test-dependent model switching."""
from diagnostics import infer
from study import *
def predict(mask,preset='default',spacing=1.):
 frozen=json.load(open(OUT/'pretest_selection.json'));assert hashlib.sha256((OUT/'pretest_selection.json').read_bytes()).hexdigest()==(OUT/'pretest_selection.sha256').read_text();d=mask.ndim
 if preset not in frozen['configurations'][str(d)]:raise ValueError('No joint default met the validation rule. Choose geometry_priority or orientation_priority explicitly.')
 c=frozen['configurations'][str(d)][preset];f=np.where(mask>0,1.,-1.).astype('float32');r,axis,t=infer(f,c['generator'],c['N'],c['heads'],c['coefficient'],spacing)
 return dict(dimension=d,preset=preset,source_model=c['model'],fixed_N=c['N'],geometry=r['prediction'],axis=axis.tolist(),eigengap=r['eigengap'],ambiguous=r['predicted_ambiguous'],spacing=spacing,timing=t,selection_sha256=(OUT/'pretest_selection.sha256').read_text())
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mask');p.add_argument('--preset',default='default');p.add_argument('--spacing',type=float,default=1.);p.add_argument('--out',default='prediction.json');a=p.parse_args();Path(a.out).write_text(json.dumps(predict(np.load(a.mask),a.preset,a.spacing),indent=2))
