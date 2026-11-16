from study_core import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_fixed_study import manifest

def table(df,cols=None):return (df[cols] if cols else df).to_markdown(index=False,floatfmt='.4g')
def run():
 cfg=json.load(open(OUT/'selected_configuration.json'));N=cfg['selected_N'];kern=json.load(open(OUT/'kernel_by_iteration.json'));vs=pd.read_csv(OUT/'validation_summary.csv');ts=pd.read_csv(OUT/'test_summary.csv');test=pd.read_csv(OUT/'test_results.csv');cp=pd.read_csv(OUT/'checkpoint_metrics.csv');tim=pd.read_csv(OUT/'timing.csv');families=pd.read_csv(OUT/'test_family_summary.csv');degree=pd.read_csv(OUT/'degree_results.csv');manifest_df=manifest()
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 for dim in [2,3]:
  fig,axs=plt.subplots(1,2,figsize=(12,4))
  for noise in [0,.002,.01]:
   g=vs[(vs.dim==dim)&(vs.method=='learned')&(vs.noise==noise)].sort_values('N');axs[0].plot(g.N,g.geometry_MARE,'o-',label=f'noise {noise*100:g}%');axs[1].plot(g.N,g.orientation_MAE,'o-',label=f'noise {noise*100:g}%')
  axs[0].set(xlabel='fixed iterations N',ylabel='Geometry MARE (%)');axs[1].set(xlabel='fixed iterations N',ylabel='Orientation MAE (deg)')
  for a in axs:a.axvline(N,color='black',ls=':',label=f'selected N={N}');a.legend(fontsize=8);a.grid(alpha=.2)
  fig.suptitle(f'{dim}D validation only; independently optimized kernel at each N');fig.tight_layout();fig.savefig(figdir/f'accuracy_iterations_{dim}d.png',dpi=170);plt.close(fig)
  fig,ax=plt.subplots(figsize=(6,4))
  for noise in [0,.002,.01]:
   g=vs[(vs.dim==dim)&(vs.method=='learned')&(vs.noise==noise)].sort_values('N');ax.plot(1000*g.median_seconds,g.geometry_MARE,'o-',label=f'noise {noise*100:g}%')
   for _,r in g.iterrows():ax.annotate(str(int(r.N)),(1000*r.median_seconds,r.geometry_MARE),fontsize=8)
  ax.set(xlabel='Estimated single-N pipeline ms (checkpoint cumulative timings)',ylabel='Validation geometry MARE (%)',title=f'{dim}D time–accuracy Pareto candidates');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/f'pareto_{dim}d.png',dpi=170);plt.close(fig)
  fig,axs=plt.subplots(1,5,figsize=(13,3))
  for ax,it in zip(axs,CHECKPOINTS):
   b,t,u=kern['kernels'][str(dim)][str(it)]['p'];K=np.outer([u,t,1-2*t-2*u,t,u],[-b,-(.5-2*b),0,.5-2*b,b]);ax.imshow(K,cmap='coolwarm',vmin=-.7,vmax=.7);ax.set_title(f'N={it}');ax.set_xticks([]);ax.set_yticks([])
  fig.suptitle(f'{dim}D learned derivative: axial × one transverse plane (3D uses two)');fig.tight_layout();fig.savefig(figdir/f'kernels_{dim}d.png',dpi=170);plt.close(fig)
 # Pareto flags over validation criteria (geometry/angle/runtime averaged across noise strata).
 pareto=[]
 for dim in [2,3]:
  g=vs[(vs.dim==dim)&(vs.method=='learned')].groupby('N')[['geometry_MARE','orientation_MAE','rotation_shift_CV','median_seconds']].mean()
  for it,row in g.iterrows():
   dominated=any((other<=row).all() and (other<row).any() for _,other in g.iterrows());pareto.append(dict(dim=dim,N=it,pareto_nondominated=not dominated,**row.to_dict()))
 pd.DataFrame(pareto).to_csv(OUT/'pareto_front.csv',index=False)
 profiles=pd.read_csv(OUT/'wall_profiles.csv')
 for dim in [2,3]:
  familylist=profiles[profiles.dim==dim].family.unique();fig,axs=plt.subplots(1,len(familylist),figsize=(5*len(familylist),3.7));axs=np.atleast_1d(axs)
  for ax,family in zip(axs,familylist):
   for it,g in profiles[(profiles.dim==dim)&(profiles.family==family)].groupby('N'):ax.plot(g.distance,g.mean_s,label=f'N={it}')
   ax.set(title=family,xlabel='Analytic signed distance: evaluation only',ylabel='mean s');ax.legend(fontsize=8);ax.grid(alpha=.2)
  fig.tight_layout();fig.savefig(figdir/f'wall_profiles_{dim}d.png',dpi=170);plt.close(fig)
 # Quantitative frequency and angular response.
 fr=[]
 for dim in [2,3]:
  for it in CHECKPOINTS:
   b,t,u=kern['kernels'][str(dim)][str(it)]['p']
   for mag in [.2,.5,1.,1.5,2.,2.8]:
    for theta in np.linspace(0,np.pi/2,31):
     direction=np.array([np.cos(theta),np.sin(theta)]+([.4] if dim==3 else []));direction/=np.linalg.norm(direction);w=mag*direction;response=[]
     for k in range(dim):
      a=2*(.5-2*b)*np.sin(w[k])+2*b*np.sin(2*w[k])
      for j in range(dim):
       if j!=k:a*=1-2*t-2*u+2*t*np.cos(w[j])+2*u*np.cos(2*w[j])
      response.append(a)
     response=np.array(response);fr.append(dict(dim=dim,N=it,frequency=mag,theta=theta,gain=np.linalg.norm(response)/mag,direction_error_deg=np.degrees(np.arccos(np.clip(response@direction/max(np.linalg.norm(response),1e-12),-1,1)))))
 pd.DataFrame(fr).to_csv(OUT/'frequency_response.csv',index=False)
 # Two worst unique geometry units per dimension, chosen only for post-test illustration.
 worst=test[(test.N==N)&(test.method=='learned')].assign(abs_error=lambda a:abs(a.error_pct)).sort_values('abs_error',ascending=False).drop_duplicates('unit').groupby('dim').head(2)
 fig,axs=plt.subplots(len(worst),3,figsize=(10,3*len(worst)))
 for axrow,(_,row) in zip(axs,worst.iterrows()):
  z=np.load(CACHE/(row.cache_key+'.npz'));n=z['n'+str(N)];f=z['f'];entry=manifest_df[manifest_df.case==row['case']].iloc[0].to_dict();_,clean,_,_,_,_=geometry(entry)
  arrays=[clean,f>0,n[...,-1]>0]
  if row.dim==3:
   zid=int(clean.sum(axis=(0,1)).argmax());arrays=[a[:,:,zid] for a in arrays]
  for ax,a,title in zip(axrow,arrays,['Clean truth (slice)','Noisy binary input',f'Fixed N={N} segmentation']):ax.imshow(a,cmap='gray');ax.set_title(title);ax.set_axis_off()
  axrow[0].set_title(f'{row.family}: GT; error {row.error_pct:.2f}%'+(f' (z={zid})' if row.dim==3 else ''))
 fig.tight_layout();fig.savefig(figdir/'failure_cases.png',dpi=170);plt.close(fig)
 # Baseline summary with independent geometry weighting and bootstrap.
 bl=pd.read_csv(OUT/'baseline_results.csv');bs=[]
 for keys,g in bl.groupby(['dim','noise','method','calibrated']):
  units=g.groupby('unit').agg(error=('error_pct',lambda a:np.nanmean(abs(a)) if a.notna().any() else np.nan),angle=('angle_error','mean'),seconds=('measurement_seconds','mean'));vals=units.error.dropna().to_numpy();rng=np.random.default_rng(61116);bt=rng.choice(vals,(2000,len(vals)),replace=True).mean(1) if len(vals) else [np.nan]
  bs.append(dict(zip(['dim','noise','method','calibrated'],keys),units=len(units),geometry_MARE=np.mean(vals) if len(vals) else np.nan,CI95_low=np.quantile(bt,.025),CI95_high=np.quantile(bt,.975),orientation_MAE=units.angle.mean(),median_seconds=g.measurement_seconds.median()))
 bs=pd.DataFrame(bs);bs.to_csv(OUT/'baseline_summary.csv',index=False)
 # Runtime paired by fixture, excluding conventional rows.
 dep=tim[tim.kind=='single_N_deployment'];rt=dep.groupby(['dim','N'])[['initialization_seconds','field_seconds','measurement_seconds','total_seconds']].median().reset_index();rt.to_csv(OUT/'timing_summary.csv',index=False)
 sensitivity=pd.read_csv(OUT/'selection_sensitivity.csv');sv=sensitivity[sensitivity.tolerance==.1];selv=vs[(vs.N==N)&(vs.method=='learned')];selt=ts[(ts.N==N)&(ts.method=='learned')];ttest=test[(test.N==N)&(test.method=='learned')];transfer=pd.read_csv(OUT/'transfer_results.csv');trsum=transfer[transfer.N==N].groupby(['dim','h','D']).agg(MARE=('error_pct',lambda a:np.mean(abs(a))),orientation_MAE=('angle_error','mean'),max_residual=('one_step_rms','max'),unresolved=('unresolved_wall','sum')).reset_index();trsum.to_csv(OUT/'transfer_summary.csv',index=False)
 maskstats=ttest.groupby(['dim','noise']).agg(min_clean_Dice=('clean_dice','min'),max_residual=('one_step_rms','max'),max_local_change=('one_step_max','max'),unresolved=('unresolved_wall','sum')).reset_index()
 kernrows=[]
 for dim in ['2','3']:
  for it in CHECKPOINTS:
   r=kern['kernels'][dim][str(it)];kernrows.append(dict(dim=dim,N=it,b=r['p'][0],t1=r['p'][1],t2=r['p'][2],evaluations=r['evaluations'],objective=r['objective']))
 splits=manifest_df.groupby(['split','dim']).agg(independent_units=('unit','nunique'),images=('case','count')).reset_index()
 paired=pd.read_csv(OUT/'paired_bootstrap_ci.csv');pairedsel=paired[(paired.split=='test')&(paired.N==N)&(paired.metric=='geometry_abs_error_delta_pp')]
 text=f'''# PBGIA 고정 반복횟수–측정 커널 최적화: 실행 결과 보고서

2026-11-16 · 기준 문서: `CODEX_TASK_KO.md` · 새로운 실행 데이터만 집계.

## 결론: 고정 N*={N}

사전 지정 validation 기준을 적용한 선택은 **N*={N}**이다. 2D/3D는 동일 반복횟수를 사용하되, 측정용 커널은 차원별·N별로 독립 학습했다. 주 조건은 J=1,h=2,D=1, 생성 구배는 중앙차분, float32, alpha=1이다. 선택은 GT 정확도와 비용의 절충이며 고정점 수렴을 뜻하지 않는다.

선택 configuration의 clean test MARE는 2D {selt[(selt.dim==2)&(selt.noise==0)].geometry_MARE.iloc[0]:.3f}%, 3D {selt[(selt.dim==3)&(selt.noise==0)].geometry_MARE.iloc[0]:.3f}%이다. 별도 배포 계측 중앙값은 2D {1000*rt[(rt.dim==2)&(rt.N==N)].total_seconds.iloc[0]:.2f} ms, 3D {1000*rt[(rt.dim==3)&(rt.N==N)].total_seconds.iloc[0]:.2f} ms이다.

2D clean raw Crofton의 MARE {bs[(bs.dim==2)&(bs.noise==0)&(bs.method=='Crofton')&(~bs.calibrated)].geometry_MARE.iloc[0]:.3f}%는 선택 PBGIA보다 작다. 3D clean reference-calibrated marching cubes는 {bs[(bs.dim==3)&(bs.noise==0)&(bs.method=='marching_cubes_surface_tensor')&(bs.calibrated)].geometry_MARE.iloc[0]:.3f}%이다. 이번 최적화는 40회 대비 비용 절감 조건을 만족하는 구성을 찾았지만 기존 기법에 대한 보편적 우월성을 입증하지 않는다.

2D 선택 커널 p=(b,t1,t2): `{cfg['kernel_by_dimension']['2']}`.
3D 선택 커널 p=(b,t1,t2): `{cfg['kernel_by_dimension']['3']}`.
환산계수 c: 2D `{cfg['calibration_by_dimension']['2']:.9g}`, 3D `{cfg['calibration_by_dimension']['3']:.9g}`. 단위 격자의 둘레/면적용이며 다른 해상도에 임의 전용하지 않는다.

## 1. 모델 및 선행 결과 구분

`Hv=J Σ_axial v(t)+D grad(s(t))`, `Hs=J Σ_axial s(t)+hf`, `n(t+1)=H/|H|`의 동기식 갱신이다. f는 고정하고 div(v), grad(f) 방향장, Laplacian 교체, 적응형 종료를 사용하지 않았다. |H|<=1e-12에서는 이전 스핀을 유지한다. 비변분 정렬 알고리즘이므로 단일 에너지 감소나 tr(T)의 에너지 해석을 하지 않는다.

포함된 smoke 2D 40²/3D 24³ 데이터를 직접 다시 계산해 저장된 체크포인트와 비트 단위 일치를 검사했다. 이전 실행의 원시 코드/CSV는 현재 workspace의 `/workspace/research/PBGIA_Codex_handoff_20261116`에서 실제로 접근 가능했다. 이전 보고서와 이번 ZIP의 선행 자료를 확인했지만 이전 projected-gradient 결과와 이번 결과는 합치지 않았다. `prior_learned` 대조군은 이전 동적 모델 학습 커널의 고정 계수만 가져온 것이며 새 필드에서 새로 측정했다.

## 2. 분할과 데이터

split manifest를 학습 전에 저장했다. 한 shape family/size prototype의 모든 회전·이동·잡음 파생 영상은 같은 split이다. 같은 차원/도형에서 반지름도 split 사이에 중복하지 않는다. Train 전용 reference 원 R=5.25, 구 R=4.25는 다른 geometry와 분리한다.

{table(splits)}

2D: 원, stadium, 타원, 링, 분리 원 2개. 3D: 구, capsule, torus, 타원체, 분리 구 2개, 작은 구. **2D 링과 3D 타원체는 shape-family held-out test**로만 두었다. 다른 family도 test에서는 새 반지름/원형을 사용한다. 원형별 두 개의 회전·subvoxel 이동, 세 잡음률 0%,0.2%,1%를 고정했다. 3D는 기울어진 축을 포함한다. 크기는 2D 80², 3D 56³이다. 독립 geometry unit은 총 51개이고 calibration 2개를 제외하면 49개이다.

Noise는 외곽 7 voxel을 제외한 내부에서 독립 salt-and-pepper flip을 가한다. 회전/이동은 같은 geometry 파생 영상에서 유지하고 잡음 seed만 바꾼다. 초기화는 supplied binary EDT 함수뿐이며 해석적 거리장/법선은 입력 마스크 생성과 진단·평가에만 사용했다. Capsule은 원통 길이 2R, 전체 길이 4R, 면적 8πR²이다. Torus는 minor radius=.3R. 타원체 GT는 연속 면적 요소 120×240 quadrature로 계산했다.

## 3. 체크포인트와 학습의 공정성

각 고유 입력·J/h/D 조건에서 40 step을 한 번 수행하여 N=5,10,20,30,40을 저장했다. 키는 입력 bytes, shape, 생성 규칙, dtype, 초기화 코드 hash와 파라미터를 포함한다. 개별 N에 대해 과학 실험의 이완을 다섯 번 중복하지 않았다. 학습 계산은 tensor derivative의 정확한 선형 basis Gram으로 가속했으며 full-array tensor 계산과 일치를 자동 검사했다. 이 가속은 학습용이고 CLI inference는 실제 convolution 측정이다.

각 차원·N에 동일한 3차원 parameter range, 같은 초기 탐색 seed, differential evolution **270회 objective evaluation**을 제공했다. 고정 예산 종료이며 전역 최적성/optimizer 수렴을 주장하지 않는다. 측정 커널 p의 axial weights는 offset ±1에 ±(.5−2b), ±2에 ±b이고 transverse smoothing은 [t2,t1,1−2t1−2t2,t1,t2]이다. 3D에서는 두 transverse 축에 적용한다. 상수장 도함수 0, 선형장 응답 1을 만족한다. 축치환 공변성, 모든 축 response, 주파수 response를 검사했다. 연속 회전 불변성은 가정하지 않는다.

학습 목적은 geometry relative MSE + .05*(orientation_deg/10)² + 회전/이동 CV²이다. Train geometry의 noise rates도 포함한다. 각 (N,kernel,dimension,h,D)에 train reference 원/구에서만 c를 정한다. `M=trace(T)`, `T_ij=Σ_x,a D_i n_a D_j n_a`, `geometry=cM`이다. c는 test GT에서 적합하지 않는다.

{table(pd.DataFrame(kernrows))}

![2D 커널](figures/kernels_2d.png)
![3D 커널](figures/kernels_3d.png)

## 4. Validation 선택 규칙과 결과

h=2는 사전 고정한 주 조건이다. h를 test 성능으로 조정하지 않았다. 각 dimension/noise stratum에서 독립 geometry를 동일 가중한 MARE 증가가 N=40 전용 학습 커널 대비 <=.1 percentage point, 연결 elongated object orientation MAE 증가 <=.1 degree, clean Dice<.95 또는 Euler/components mismatch 실패 수 증가 없음, 대표 degree 실패 증가 없음, 실행 비용 개선을 모두 만족하는 가장 작은 N을 선택했다. 이 기준에 적합하지 않으면 40을 유지한다. p90/p95, CV는 함께 공개하지만 추가 임의 threshold로 test 후 규칙을 바꾸지 않았다. 이번 validation에서는 N=20의 3D geometry 증가가 noise별 약 .13–.20 pp, orientation 증가가 약 .23–.28도여서 탈락했다. 이 차이는 field step뿐 아니라 N별로 독립 학습한 커널의 finite-budget 차이도 포함하므로, 30회가 모델의 본질적 최소 반복횟수라고 해석하면 안 된다.

{table(sv,['N','passed','degree_ok'])}

선택 N의 validation:

{table(selv,['dim','noise','n_units','geometry_MARE','error_p90','error_p95','orientation_MAE','rotation_shift_CV','mask_failures','median_seconds'])}

![2D accuracy](figures/accuracy_iterations_2d.png)
![3D accuracy](figures/accuracy_iterations_3d.png)
![2D Pareto](figures/pareto_2d.png)
![3D Pareto](figures/pareto_3d.png)

Tolerance .05/.1/.2/.5의 sensitivity는 `selection_sensitivity.csv`에 모든 stratum의 paired 증가와 함께 기록했다. 그림은 checkpoint cumulative update time+초기화+단일 측정으로 추정한 단일 N 비용이다. 모든 checkpoint를 저장·진단하는 연구 실행 전체 비용과 혼동하지 않는다. Pareto non-dominated 표에는 geometry/angle/CV/runtime 평균 4개 지표를 사용했고 최종 선택 규칙은 위의 더 엄격한 stratum 조건이다.

`selected_configuration.json`을 test 이전에 저장하고 SHA256 lock을 남겼다. test 단계는 이 파일이 바뀌면 중단하고, 이미 test 결과가 있으면 조용히 재평가하지 않는다. `test_audit.json`에 선택 파일 hash와 실행 완료 시각을 남겼다. Test에는 선택 N 및 사전 지정 N=40 참조와 고정 커널/기존 방법 대조만 적용했다. N=40도 GT가 아니며 실제 GT 오차를 따로 계산한다.

## 5. 최종 test 정확도와 baseline

선택 configuration test 결과(geometry 원형별 평균을 다시 평균; clean/noisy 별도):

{table(selt,['dim','noise','n_units','geometry_MARE','CI95_low','CI95_high','error_p90','error_p95','orientation_MAE','rotation_shift_CV','mask_failures'])}

학습 원형과 다른 test family/size의 개별 오차는 `test_results.csv`, family-held-out 결과는 `test_family_summary.csv`에 분리했다. 아래는 clean의 family-held-out 결과이다.

{table(families[(families.N==N)&(families.method=='learned')&(families.noise==0)&families.heldout_family],['dim','family','n_units','MARE','p95','orientation_MAE'])}

N=40 대비 test paired absolute-error 변화(percentage point), independent geometry unit bootstrap:

{table(pairedsel,['dim','noise','heldout_family','n_units','mean','CI95_low','CI95_high'])}

Bootstrap은 원형 단위로 2,000회(summary) 또는 3,000회(paired) 재표본화하고 그 원형의 회전·위치·noise 파생 샘플을 함께 유지한다. 독립 표본이 family당 단 두 개 test 원형에 불과하므로 CI는 작은 합성 집합의 기술적 불확실성이고 모집단 우월성 보증이 아니다.

기존 방법 clean baseline(원시값과 train reference 보정값 모두 표시):

{table(bs[bs.noise==0],['dim','method','calibrated','geometry_MARE','CI95_low','CI95_high','orientation_MAE','median_seconds'])}

Gaussian sigma=1은 사전 고정하며 test에서 튜닝하지 않았다. Crofton/marching squares의 각도는 binary PCA로 별도 측정한 값이며 그 둘레 추정 자체의 방향값이 아니다. Marching cubes 행은 surface-normal tensor 각도이다. Gaussian gradient integral은 structure tensor를 사용하고 threshold+marching은 PCA를 사용한다. PCA 단독 행에는 길이/면적을 억지로 부여하지 않았다.

## 6. 실행 시간과 배포

독립 배포 microbenchmark: 대표 test fixture별 warmup 후 5회, 선택 N 및 N=40만 따로 실행했다. 이는 배포 비용 계측용 반복이며 B1의 중복 없는 checkpoint 실험과 구분한다. 아래는 표본과 반복을 합친 중앙값 초이다. 초기화와 measurement와 field loop를 각각 실측했다.

{table(rt)}

세부 timing과 baseline 측정 시간은 `timing.csv`에 있다. 메인 checkpoint iteration curve의 비용은 기계 부하에 민감한 단일 측정이고, 위 warmed 결과도 shared CPU의 소규모 microbenchmark다. 파일 I/O, 합성 GT 생성, 학습·kernel 탐색·degree 진단은 배포 pipeline에서 제외했다. 개발/학습 비용을 배포 속도 개선에 포함하지 않는다. GPU는 사용하지 않았다. CPU·Python·패키지·thread 설정은 `environment.json`, 코드 commit은 `code_version.json`에 보관한다.

단일 이미지 실행:

```bash
python code/predict_fixed.py mask.npy --config output/selected_configuration.json --out prediction.json
```

mask는 2D/3D boolean 또는 0/1, ±1 `.npy`; `>0`를 내부로 본다. 출력은 geometry, tensor, eigengap, 축 경고와 timing JSON 및 NPZ다. 주어진 차원에서 고정 N만 정확히 실행한다. 단위 spacing=1, 배경 여백과 calibration 적용 범위가 전제이다.

## 7. 잔차·위상·형태 보존과 실제 wall

선택 N의 test 진단:

{table(maskstats)}

One-step residual은 **현재 checkpoint에서 다음 map을 한 번 더 평가한** RMS이다. 최대 국소 변화, 지난 update와 n(t+1)−n(t−1) two-step residual도 저장한다. 작은 two-step 변화만으로 수렴으로 오인하지 않도록 checkerboard 2-cycle smoke test를 포함했다. 짧은 고정 반복에서 geometry가 정확해도 field가 평형이라고 하지 않는다.

대표 3D sphere/capsule/torus에 모든 N에서 three-target integer degree를 측정했다. 외부 negative collar를 남극으로 연속 정규화 compactification하고, 모든 Freudenthal tetrahedron vertex가 공통 열린 hemisphere에 드는 sufficient certificate를 검사했다. 실패하면 비특이성이 인증되지 않았다고 표시하며 단순 target 일치만으로 정수 topology를 보증하지 않는다. Euler는 이진 segmentation, degree는 compactified spin map에 대한 서로 다른 관측량이다. 대표 소수의 성공을 전체 데이터의 integer topology 보증으로 일반화하지 않는다.

{table(degree[(degree.stage=='test')&(degree.N==N)],['family','N','degree','degree_target_agreement','hemisphere_min_dot','negative_collar_max_s','degree_certified'])}

Wall voxel은 |s|<.7 & |v|>.04이고 width proxy는 1.4*count/Σ|grad(s)|이다. 이것은 grid-dependent coarea proxy이며 해석적 sqrt(2A/h)가 아니다. wall_count < max(8, .02*binary-morphology-surface-voxel-count)인 heuristic unresolved flag와 normal alignment를 별도 저장했다. 이것은 실제 wall 해상도에 대한 완전한 인증은 아니다. 약한 해상도의 작은 물체와 강한 h transfer는 근거가 약하므로 geometry 평균의 개선으로 숨기지 않는다.

![2D wall profiles](figures/wall_profiles_2d.png)
![3D wall profiles](figures/wall_profiles_3d.png)
![실패 사례](figures/failure_cases.png)

## 8. 전이·격자 및 여백 진단

학습한 K_N은 고정하고 h=4,D=1 및 h=2,D=.5에서 별도 train reference로 c만 다시 측정했다. 각 test 원형의 clean rep0만 사용했고 이 결과로 선택 규칙을 바꾸지 않았다.

{table(trsum)}

`refinement_margin_checks.csv`: 같은 물리 sphere를 spacing=1,.75,.5로 voxelize해 grid refinement하고 각 spacing의 별도 train reference sphere로 c를 측정했다. J/h/D는 lattice 단위로 고정했으므로 물리적 wall 두께가 같지는 않다. 이는 geometry discretization 진단이며 동일 continuum dynamics 수렴을 입증하지 않는다. Torus domain 48³/64³/80³는 spacing=1을 유지한 margin check로 분리했다. 이미지 크기 확장만을 해상도 향상이라고 하지 않는다.

## 9. 한계 및 재현

- N={N} 선택은 validation의 정해진 범위에 한정한다. 더 엄격한 p95 한계나 자연영상 조건은 선택을 바꿀 수 있다. test 결과를 보고 재학습하거나 N을 바꾸지 않았다.
- 작은 물체가 줄어드는 경우, unresolved wall, noise-induced mask/topology 오류는 남는다. Normal alignment, segmentation fidelity, integer topology, geometric accuracy, fixed-point convergence는 별개다.
- 전체 local-gradient tensor는 분리 구형 객체의 global center axis를 복원한다는 보장이 없다. 그런 multi fixture의 tensor orientation은 사전 undefined로 처리했다. 연결 elongated fixture만 minimum-eigenvalue 장축으로 평가하고 eigengap을 함께 기록한다. PCA는 별도 대조다.
- 2D와 3D 커널을 따로 학습했으므로 하나의 universal kernel transfer를 입증한 것이 아니다. 생성 커널은 학습하지 않았으며 measurement-only 최적화이다.
- 합성 도형, 제한된 prototype/rotation/noise율, 단일 주 h/D/J, finite optimizer budget이다. train 기준 보정과 scalar-field 주변의 nonlocal residual 효과 때문에 다른 해상도/형상에서 재보정이 필요할 수 있다.
- prior 2D learned 및 3D fourth-order 우위는 이번 분할에서 자동 보장되지 않는다. 모든 기존 커널을 새 필드에서 재평가했고 우월성을 전제하지 않았다.

`README.md`의 한 명령 실행으로 smoke, train/validation, freeze, test, transfer, timing, report를 재현한다. Train/validation/test 결과, 주요 raw CSV, 독립 geometry split, kernel JSON, 선택 lock, baseline, checkpoint metrics, paired bootstrap, figures와 cache가 저장된다. 배포 CLI는 선택된 고정 map만 사용한다. 자동 테스트에는 f 불변, 배열 크기, normalization, direct-map 일치, checkpoint 복원, split/calibration leakage, kernel moments와 정확 tensor feature 일치, 2-cycle, axis permutation, smoke bitwise 재현을 포함한다.
'''
 (OUT/'REPORT_KO.md').write_text(text);print('REPORT AND FIGURES DONE',flush=True)
if __name__=='__main__':run()
