"""Self-consistent learning of stable D4-symmetric 5x5 exchange stencils on binary stadiums.

Kernel weights by bond orbit p=(a,2b,4c,10d,8e), p>=0, sum p=1.
Each objective evaluation reoptimizes all magnetization fields for the new stencil.
The only gradient approximation in outer optimization is finite residual inner L-BFGS;
we use the envelope theorem to differentiate minimized energy.
"""
import os,time,json,sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from stadium_wall_experiment import N,R,A,H,D,make_mask,init_from_binary
from compare_exchange_kernels import OLD,MASKS
ROOT=Path('/mnt/data')
ANGLES=np.arange(0,50,5)
NW=np.array([2]+[4]*8+[2],dtype=float)
RVAL=(2*np.pi+4)*R
SHELLS=[[(1,0),(0,1)],[(1,1),(1,-1)],[(2,0),(0,2)],[(2,1),(2,-1),(1,2),(1,-2)],[(2,2),(2,-2)]]
FACT=np.array([1.,2.,4.,10.,8.]); LABELS=['(1,0)','(1,1)','(2,0)','(2,1)','(2,2)']

def get_shifted(m,dy,dx):return np.roll(m,(-dy,-dx),axis=(-2,-1))

def calc(z,s,p,jac=True,extra=False):
    sh=s.shape
    v=z.reshape((3,)+sh)
    norm=np.linalg.norm(v,axis=0)
    m=v/norm
    coeff=p/FACT
    feats=np.zeros(5)
    g=np.zeros_like(m) if jac else None
    # E shell and gradient explicitly paired so variations consistent.
    for i, pairs in enumerate(SHELLS):
        w=coeff[i]
        if abs(w)<1e-15: continue
        u=0.
        for dy,dx in pairs:
            v=get_shifted(m,dy,dx)
            delta=m-v
            u+=np.sum(delta*delta)
            if jac:
                g+=2*A*w*(2*m-v-np.roll(m,(dy,dx),axis=(-2,-1)))
        feats[i]=A*u/FACT[i]
    Eex=np.dot(p,feats)
    mx,my,mz=m
    fld=H*np.sum(1-s*mz)
    dx=lambda x:(np.roll(x,-1,axis=-1)-np.roll(x,1,axis=-1))*.5
    dy=lambda x:(np.roll(x,-1,axis=-2)-np.roll(x,1,axis=-2))*.5
    dxmz=dx(mz);dymz=dy(mz);div=dx(mx)+dy(my)
    edm=D*np.sum(mz*div-mx*dxmz-my*dymz)
    E=Eex+fld+edm
    if extra:
        # Compute every shell feature even if current coefficient is zero.
        for i,pairs in enumerate(SHELLS):
            if p[i]>1e-14: continue
            val=0.
            for ddy,ddx in pairs:
                delta=m-get_shifted(m,ddy,ddx)
                val+=np.sum(delta*delta)
            feats[i]=A*val/FACT[i]
        return float(E),np.array([Eex,fld,edm]),feats
    if not jac:return float(E)
    g[2]-=H*s
    g[0]-=2*D*dxmz;g[1]-=2*D*dymz;g[2]+=2*D*div
    g=(g-m*np.sum(m*g,axis=0,keepdims=True))/norm
    return float(E),g.ravel()

def row_eval(p,itermax=85,Rr=R,angs=ANGLES,init='old',store=False):
    ee=[];gg=[];nn=[];ff=[];comps=[];states=[]
    for ang in angs:
        if Rr==R:
            s=np.where(MASKS[int(ang/5)],1.,-1.)
            m0=OLD[int(ang/5)] if init=='old' else init_from_binary(MASKS[int(ang/5)])
        else:
            # Stadium digitization at a different radius with the same pixel-centred coordinates
            xx=np.arange(N)-N/2+.5
            XX,YY=np.meshgrid(xx,xx)
            ph=np.deg2rad(ang)
            u=XX*np.cos(ph)+YY*np.sin(ph)
            v=-XX*np.sin(ph)+YY*np.cos(ph)
            mask=(np.maximum(np.abs(u)-Rr,0)**2+v*v<=Rr*Rr)
            s=np.where(mask,1.,-1.)
            m0=init_from_binary(mask)
        opt=minimize(calc,m0.ravel(),args=(s,p),jac=True,method='L-BFGS-B',
              options={'maxiter':itermax,'gtol':2e-6,'ftol':1e-12,'maxls':35,'maxcor':7})
        E,cmp,feat=calc(opt.x,s,p,extra=True)
        g=np.linalg.norm(calc(opt.x,s,p)[1])/np.sqrt(s.size)
        ee.append(E);gg.append(g);nn.append(opt.nit);ff.append(feat);comps.append(cmp)
        if store:states.append(opt.x.reshape((3,)+s.shape).astype(np.float32))
    E=np.array(ee);feat=np.array(ff)
    ww=NW if len(angs)==len(NW) and np.all(angs==ANGLES) else np.ones(len(angs))
    mean=np.average(E,weights=ww);std=np.sqrt(np.average((E-mean)**2,weights=ww))
    cv=100*std/mean;pp=100*np.ptp(E)/mean
    df=pd.DataFrame({'angle_deg':angs,'energy':E,'gradient_rms':gg,'iterations':nn,'E_exchange':np.array(comps)[:,0], 'E_field':np.array(comps)[:,1], 'E_DMI':np.array(comps)[:,2]})
    result={'cv_pct':float(cv),'range_pct':float(pp),'mean_energy':float(mean),'grad_rms_max':float(max(gg)),'iter_mean':float(np.mean(nn)), 'energy':E,'features':feat,'df':df}
    if store: result['states']=np.array(states)
    return result

def envelope_grad(res):
    E=res['energy']; f=res['features']; w=NW
    u=np.average(E,weights=w);d=np.average(f,axis=0,weights=w)
    v=np.average((E-u)**2,weights=w);std=np.sqrt(v)
    if std==0:return np.zeros(5)
    dv=2*np.average((E-u)[:,None]*(f-d[None,:]),axis=0,weights=w)
    return 100*(dv/(2*std*u)-std*d/(u*u))

if __name__=='__main__':
    t0=time.monotonic()
    p4=np.array([1.,0,0,0,0]);p9=np.array([2/3,1/3,0,0,0]);p5=np.array([.50,.20,.10,.15,.05])
    for label,p in [('NN4',p4),('ISO9',p9),('5x5 initial',p5)]:
        r=row_eval(p,itermax=85)
        print('INITIAL',label,'CV',r['cv_pct'],'range',r['range_pct'],'meanE',r['mean_energy'],'maxgrad',r['grad_rms_max'],'sec',time.monotonic()-t0,flush=True)
    # Train all 10 crystallographically unique angles with full magnetization relaxation every evaluation.
    cnt=[0]
    last_cache={}
    def objective(p):
        key=tuple(np.round(p,9))
        if key not in last_cache:
            r=row_eval(p,itermax=85)
            last_cache[key]=(r['cv_pct'],envelope_grad(r),r)
            cnt[0]+=1
            print(f'outer={cnt[0]:02d} t={time.monotonic()-t0:.1f} CV={r["cv_pct"]:.7f}% range={r["range_pct"]:.6f}% gradRMS={r["grad_rms_max"]:.2g} p={np.round(p,5)}',flush=True)
        return last_cache[key][0]
    def grad(p):
        objective(p);return last_cache[tuple(np.round(p,9))][1]
    # Lower bound for nearest neighbour bonds to prevent isolated sublattice/alias artifacts.
    cons=[{'type':'eq','fun':lambda p:sum(p)-1.,'jac':lambda p:np.ones_like(p)}]
    options={'maxiter':22,'ftol':1e-8,'disp':False}
    bounds=[(.2,1.)]+[(0.,1.)]*4
    opt=minimize(objective,p5,jac=grad,bounds=bounds,constraints=cons,method='SLSQP',options=options)
    print('OPT 1:',opt.x,opt.fun,opt.message,'time',time.monotonic()-t0,flush=True)
    # Additional start from 9pt. Cached exact evaluation points reused.
    opt2=minimize(objective,p9,jac=grad,bounds=bounds,constraints=cons,method='SLSQP',options={'maxiter':18,'ftol':1e-8})
    print('OPT 2:',opt2.x,opt2.fun,opt2.message,'time',time.monotonic()-t0,flush=True)
    best=min([opt,opt2],key=lambda z:z.fun)
    # Re-evaluate all optimized kernels with significantly longer relaxation, including old NN4.
    tests={'NN4':p4,'ISO9':p9,'5x5_initial':p5,'5x5_learned':best.x}
    summaries=[];fulls=[]
    for label,p in tests.items():
        for steps in (85,250):
            r=row_eval(p,itermax=steps,store=(label=='5x5_learned' and steps==250))
            df=r['df'].copy();df['method']=label;df['iterations_cap']=steps
            for j in range(5): df[f'p{j}']=p[j]
            fulls.append(df)
            ss={'method':label,'iterations_cap':steps,'cv_pct':r['cv_pct'],'range_pct':r['range_pct'],'mean_energy':r['mean_energy'],'max_grad_rms':r['grad_rms_max'],'average_iters':r['iter_mean'],'p':p.tolist()}
            summaries.append(ss)
            print('CHECK',ss,'sec',time.monotonic()-t0,flush=True)
            if 'states' in r:
                np.savez_compressed(ROOT/'stadium_learned5x5_magnetization.npz',angle_deg=ANGLES,m=r['states'])
    # Holdout shape size test: these radii were never used during training.
    for radius in [14,28]:
        for label,p in [('ISO9',p9),('5x5_learned',best.x)]:
            r=row_eval(p,itermax=210,Rr=radius,angs=ANGLES,init='distance')
            df=r['df'].copy();df['method']=label;df['R']=radius;df['iterations_cap']=210
            fulls.append(df)
            ss={'method':label,'R':radius,'iterations_cap':210,'cv_pct':r['cv_pct'],'range_pct':r['range_pct'],'mean_energy':r['mean_energy'],'max_grad_rms':r['grad_rms_max'],'average_iters':r['iter_mean'],'p':p.tolist()}
            summaries.append(ss)
            print('HOLDOUT',ss,'sec',time.monotonic()-t0,flush=True)
    pd.concat(fulls,ignore_index=True).to_csv(ROOT/'stadium_5x5_all_angle_energies.csv',index=False)
    pd.DataFrame([{k:v for k,v in s.items() if k!='p'} for s in summaries]).to_csv(ROOT/'stadium_5x5_summary.csv',index=False)
    data={'model':{'N':N,'R':R,'A':A,'H':H,'D':D,'angles':ANGLES.tolist(),'multiplicities':NW.tolist(),'factors':FACT.tolist(), 'shells':SHELLS,'NN4':p4.tolist(),'ISO9':p9.tolist()},'best_p':best.x.tolist(),'best_weights':(best.x/FACT).tolist(),'best_objective_pct':float(best.fun),'success':bool(best.success),'message':str(best.message),'summaries':summaries,'elapsed_s':time.monotonic()-t0,'outer_evaluations':cnt[0]}
    (ROOT/'stadium_5x5_fit_results.json').write_text(json.dumps(data,indent=2))
    print('COMPLETE',data['elapsed_s'],flush=True)
