"""Frozen model inference: output integral I and axis; q needs known P externally.
No hidden ruler, mask-quality gate, or model fallback.
"""
from evaluation import *
def predict(mask,preset='selected_w1',spacing=1.,physical_width=False):
 frozen=check_lock();c=frozen['configurations'][preset];f=np.where(mask>0,1.,-1.);a=forward(f[None],init(f,spacing=spacing,physical=physical_width)[None],c['p'],c['N'],spacing=spacing);theta=float(a['theta'][0]);return dict(I=float(a['I'][0]),axis_row_col=[float(np.cos(theta)),float(np.sin(theta))],eigengap=float(a['gap'][0]),N=c['N'],preset=preset,source_model=c['model'],spacing=spacing,unit_error=a['unit_error'],selection_sha256=sha(OUT/'model_freeze_2d.json'),note='No q/perimeter is inferred without an explicit external geometry definition/calibration.')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mask');p.add_argument('--preset',default='selected_w1');p.add_argument('--spacing',type=float,default=1.);p.add_argument('--physical-width',action='store_true');p.add_argument('--out',default='prediction.json');a=p.parse_args();Path(a.out).write_text(json.dumps(predict(np.load(a.mask),a.preset,a.spacing,a.physical_width),indent=2))
