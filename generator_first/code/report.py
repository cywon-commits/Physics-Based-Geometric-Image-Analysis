"""Build figures and Korean report from frozen, already scored results only."""
from evaluate import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def table(df,cols=None):return (df[cols] if cols else df).to_markdown(index=False,floatfmt='.4g')
def save(name):plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=170);plt.close()

def report():
 assert (OUT/'test_audit.json').exists();frozen=json.load(open(OUT/'pretest_selection.json'));models=json.load(open(OUT/'models.json'));test=pd.read_csv(OUT/'test_results.csv');ts=pd.read_csv(OUT/'test_summary.csv');val=pd.read_csv(OUT/'validation_summary.csv');train=pd.read_csv(OUT/'train_summary.csv');cl=pd.read_csv(OUT/'test_conventional_summary.csv');pairs=pd.read_csv(OUT/'paired_bootstrap.csv');tr=pd.read_csv(OUT/'generator_trials.csv');manifest=make_manifest();noise=pd.read_csv(OUT/'test_noise_summary.csv');res=pd.read_csv(OUT/'resolution_results.csv');small=pd.read_csv(OUT/'small_object_stress.csv');timing=pd.read_csv(OUT/'timings.csv');symbol=pd.read_csv(OUT/'fourier_symbols.csv');budget=pd.read_csv(OUT/'budget_sensitivity.csv')
 primary=['A','A_train_c','B','D','E','F','G','geometry_priority','orientation_priority','default'];cols=['dim','model','N','geometry_MARE','angle_MAE','CI_low','CI_high','median_seconds'];main=ts[ts.model.isin(primary)]
 cn=summary(pd.read_csv(OUT/'test_conventional.csv'),['dim','model','mode','N','noise']);cn.to_csv(OUT/'test_conventional_noise_summary.csv',index=False)
 # Pareto and N curves use validation, never test-based selection.
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  v=val[(val.dim==d)&(val['mode']=='retuned')&val.model.isin(['B','D','E','F','G'])]
  for name,g in v.groupby('model'):ax.plot(g.geometry_MARE,g.angle_MAE,'o-',label=name)
  ax.set(xlabel='Geometry MARE (%)',ylabel='Axis MAE (degree)',title=f'{d}D validation trade-off');ax.legend(ncol=2,fontsize=8)
 save('validation_pareto')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  for name,g in val[(val.dim==d)&(val['mode']=='retuned')&val.model.isin(['B','D','E','F','G'])].groupby('model'):ax.plot(g.N,g.geometry_MARE,'o-',label=name)
  ax.set(xlabel='Fixed iterations N',ylabel='Geometry MARE (%)',title=f'{d}D validation, independent heads');ax.legend(fontsize=8)
 save('fixed_N')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  for name,g in noise[(noise.dim==d)&noise.model.isin(['A','B','D','E','F'])].groupby('model'):ax.plot(100*g.noise,g.geometry_MARE,'o-',label=name)
  ax.set(xlabel='Flip probability (%)',ylabel='Geometry MARE (%)',title=f'{d}D independent test');ax.legend(fontsize=8)
 save('noise')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  for name,g in res[(res.dim==d)&(res.method=='PBGIA')&(res['mode']=='frozen_c')].groupby('model'):ax.plot(g.spacing,abs(g.error_pct),'o-',label=name)
  conv=res[(res.dim==d)&(res.model=='B')&(res['mode']=='raw')]
  for name,g in conv.groupby('method'):ax.plot(g.spacing,abs(g.error_pct),'--',label=name)
  ax.set(xlabel='Physical spacing a',ylabel='Absolute geometry error (%)',title=f'{d}D same physical object, rerasterized');ax.legend(fontsize=7)
 save('resolution')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  for name,g in small[(small.dim==d)&small.model.isin(['A','B','D','E','F'])].groupby('model'):ax.plot(g.R,abs(g.error_pct),'o-',label=name)
  ax.set(xlabel='Physical radius',ylabel='Absolute geometry error (%)',title=f'{d}D small-object stress (known size regime)');ax.legend(fontsize=8)
 save('small_objects')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  for name,g in symbol[(symbol.dim==d)&symbol.model.isin(['B','D','E','F','G'])].groupby('model'):ax.plot(g.k,100*g.angular_CV,'o-',label=name)
  ax.set(xlabel='Wave number magnitude',ylabel='Symbol angular CV (%)',title=f'{d}D generator derivative');ax.legend(fontsize=8)
 save('fourier_anisotropy')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  tt=timing[(timing.dim==d)&(timing.kind=='warm')].groupby('model')[['initialization_seconds','field_seconds','observation_seconds']].median();tt.plot.bar(stacked=True,ax=ax);ax.set(title=f'{d}D warm latency',ylabel='Seconds');ax.tick_params(axis='x',rotation=45);ax.legend(fontsize=7)
 save('timing')
 # Rotation error is separate from rotation dispersion.
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for d,ax in zip([2,3],axs):
  for name,g in test[(test.dim==d)&(test.noise==0)&test.model.isin(['B','D','E','F'])].groupby('model'):
   h=g[g.angle_error.notna()];ax.scatter(h.rep,h.angle_error,label=name,alpha=.65)
  ax.set(title=f'{d}D clean test axis error',xlabel='Independent rotation/shift index',ylabel='Axial error (degree)');ax.legend(fontsize=8)
 save('rotation')
 coeff=[];chosen=[]
 for d,conf in frozen['configurations'].items():
  for preset in ['default','geometry_priority','orientation_priority','B','D','E','F','G']:
   if preset not in conf:continue
   c=conf[preset];chosen.append(dict(dim=d,preset=preset,model=c['model'],N=c['N'],geometry_val=c.get('validation_geometry_MARE'),angle_val=c.get('validation_angle_MAE')));coeff.append(dict(dim=d,preset=preset,N=c['N'],G=c['generator']['gp'],exchange=c['generator']['family'],E_weights=c['generator']['ep'],Kgeom=c['heads']['geometry']['p'],Korient=c['heads']['orientation']['p'],orientation=c['heads']['orientation']['type'],axis=c['heads']['orientation']['which'],mix=c['heads']['orientation']['mix'],c=c['coefficient']))
 cf=pd.DataFrame(coeff);cf.to_csv(OUT/'selected_coefficients.csv',index=False)
 # Per-N retraining comparison at matching N only.
 ns=[]
 for d in [2,3]:
  for N in [5,10,30]:
   for name in ['D',f'D_N{N}']:
    h=val[(val.dim==d)&(val.model==name)&(val.N==N)&(val['mode']=='fixed_B_train_c')]
    if len(h):ns.append(h.iloc[0])
 ns=pd.DataFrame(ns)
 # Equal-N fixed-observer comparisons, primary scientific attribution.
 fixed=ts[ts.model.str.contains('fixed_B')|ts.model.eq('B')]
 rawval=val[(val.model.isin(['raw_tied','raw_scale_matched','binary_init','E_fixedG','F_fixedG','B','D']))&(val['mode']=='fixed_B_train_c')]
 selected_ctrl=rawval.sort_values('geometry_MARE').groupby(['dim','model']).head(1)
 warm=timing[timing.kind=='warm'].groupby(['dim','model','N']).agg(mean_total=('wall_seconds','mean'),median_total=('wall_seconds','median'),mean_field=('field_seconds','mean'),mean_observer=('observation_seconds','mean')).reset_index()
 hist=summary(pd.read_csv(OUT/'test_legacy.csv'))
 bootstrap=pairs[pairs.model.str.contains('fixed_B')|pairs.model.isin(['D','E','F'])]
 trainingtime=pd.read_csv(OUT/'training_timings.csv');cpu=json.load(open(OUT/'training_process_cpu.json')) if (OUT/'training_process_cpu.json').exists() else {};stages=json.load(open(OUT/'stage_timings.json')) if (OUT/'stage_timings.json').exists() else []
 gains=[]
 for d in [2,3]:
  b=ts[(ts.dim==d)&(ts.model=='B')].iloc[0]
  for name in ['D_fixed_B_frozen_c','D_fixed_B_train_c','E_fixed_B_train_c','F_fixed_B_train_c','D','E','F']:
   h=ts[(ts.dim==d)&(ts.model==name)].iloc[0];gains.append(f"- {d}D {name}: B 대비 기하 MARE {h.geometry_MARE-b.geometry_MARE:+.4f} pp, 각도 MAE {h.angle_MAE-b.angle_MAE:+.4f}°. 음수는 개선이다.")
 text=f'''# 자화장 생성 연산자 우선 최적화: 2D·3D 재현 연구

## 1. 핵심 결론과 판정 범위

이 보고서는 신규 generator-first 의뢰서를 실제 실행한 결과다. **이번 유한 예산에서는 학습형 생성 미분의 일관된 기하 정확도 우월성이 입증되지 않았다.** 기본 선택은 두 차원 모두 학습형 G2가 아니라 고정 4차 중심차분 G1(실험 ID G), N=10이다. 이 선택에서도 관측 head는 새로 독립 학습했다. E1/E2 교환 학습의 일관된 추가 기하 이득도 확인되지 않았다.

선택된 default의 신규 test MARE는2D2.3261%,3D1.1113%이고, 원래 frozen PBGIA-30은 각각2.0958%,1.3356%다. default 각도 MAE는2D0.4350°,3D0.8128°, PBGIA-30은0.5599°,0.4794°다. 따라서 2D에서는 기하가 나빠지고 방향이 좋아졌으며,3D에서는 기하가 좋아지고 방향이 나빠지는 trade-off가 있다. '모든 측정이 개선된 새 학습 커널'이라는 결론을 지지하지 않는다.

 **동일 관측 커널·동일 N에서 생성기만 바꾼 결과**를 먼저 판단하며, 관측자 재학습과 반복횟수 변경의 효과는 분리했다. 신규 모델이 반드시 이겨야 성공이라는 전제는 두지 않았다. 아래 차이는 같은 신규 test 원형에서 계산했으며, test를 보고 계수나 N을 바꾸지 않았다.

{chr(10).join(gains)}

같은 N·동일 관측자 결과가 생성장 자체의 증거다. D/E/F의 retuned 결과는 생성기·관측자·선택 N을 모두 포함하는 최종 시스템 비교이므로 생성기만의 우월성으로 해석하지 않는다. 작은 차이나 일부 family의 개선을 보편적 우월성으로 확대하지 않는다.

최종 validation 선택:

{table(pd.DataFrame(chosen))}

공동 default는 validation 최저 기하 MARE의 +0.25 pp, 최저 각도 MAE의 +0.25° 안에서 가장 빠른 후보다. 두 조건의 교집합이 없으면 default를 만들지 않고 geometry_priority/orientation_priority만 제공한다. 이 조건은 reconstruction/위상/잔차 gate가 아니며 test 뒤 fallback도 없다. 실측 단일 validation rollout의 component latency를 선택에 사용했고, 별도의 warm timing은 나중에 측정했다. `validation_pareto.csv`는 기하·방향·latency Pareto를 모두 보존한다. 유한한 후보 집합에서의 선택이며 전역 최적 N이라는 뜻은 아니다.

## 2. 코드 검토·수치 모델

요청된 GitHub 참조 브랜치의 fixed_iteration checkpoint, joint core, train_joint, design 및 보고서를 직접 읽었다. 원본 SHA/URL은 `reference/github_source_review.json`에 있다. 기존 G0는 실제로 0.5*(roll(s,-1)-roll(s,+1)) 중심차분이다. 기존 screening 함수나 train_joint의 candidate_score는 가져오지 않았다. 검증된 순수 수치 함수만 복사했고 provenance를 남겼다. 과거 numerical coefficients는 A와 H 역사적 통제에만 사용하며 새로운 G/K/head의 초기값으로 복사하지 않았다. 이전 geometry-only 실행도 새 독립 study와 분리되어 있다.

H_v=J KJ*v+D Ggen*s, H_s=J KJ*s+h f, n_next=H/||H||. f는 불변, grad의 대상은 매 단계의 s, 교환도 매 단계 갱신된다. zero-H(1e-12 이하)는 이전 상태를 유지한다. div(v), grad(f), Laplacian 교환은 없다. 2D의 n은3성분, 3D는4성분이며 각 격자점에서 단위 벡터다. 기본 J=1,h=2,D=1, periodic, pinned binary EDT/tanh(width2.3) 초기화다. analytic GT distance는 초기화에 쓰지 않는다. N은 정확히5/10/20/30/40회다. 선택이나 early stopping에 잔차를 쓰지 않는다.

G의 support는 최대 5×5 및5×5×5다. separable d=[-b,-(.5-2b),0,.5-2b,b], q=[u,t,1-2t-2u,t,u]. b=t=u=0은 G0, b=-1/12,t=u=0은 G1이다. 상수 응답0, 1차 moment1, 축순열/반사 공변성을 테스트했다. E0=2d NN sum, E1=sum_i G_i**2 원소 제곱, E2=self/nearest/second/diagonal positive shell mass. E1/E2 기본합2d이며 -G†G가 아니다. raw E1은 별도로 비교했다. raw합이 alpha·2d면 normalized E1의 h/alpha,D/alpha로 모든 H의 상수배 동치가 된다. 이 동치는 수치 테스트로 확인했다.

기하 head Kgeom은 train 원형 단위 MARE만, Korient는 축 각도만 최적화했다. bond/current HJ, direct v⊗v, gradient tensor, 정규화 traceless tensor simplex ensemble과 min/max eigenaxis를 train에서 비교했다. geometry와 orientation p는 별도다. maximum eigenaxis는 같은 orientation DE 해의 대조이며 별도 DE를 배정하지 않았다. 2D half-angle/90° convention 및3D 축교환을 검사했다. 원·구·ring·torus의 장축이나 복합체에 임의 GT를 부여하지 않았다. true axis가 있는 stadium/ellipse/capsule/ellipsoid/peanut만 각도를 계산했다. 작은 eigengap의 예측도 각도 분모에서 제거하지 않았다.

## 3. 데이터·GT·누출 방지

{table(manifest.groupby(['dim','split']).agg(images=('case','count'),units=('unit','nunique')).reset_index())}

각 원형에4개의 독립 회전/shift와 clean/0.2%/1% salt-pepper 변형을 만들었다. 동일 원형의 모든 파생 영상은 같은 split이다. Train과 validation은 차원별4원형, test는 차원별8원형이며 전체 test 독립 단위는16개다. calibration 원/구2개는 train-only다. Test에는 새 크기의 seen family와 ring/ellipsoid/peanut family held-out이 포함된다. seed291116이며3D 회전은 Gaussian QR의 Haar SO(3) 표본,2D는[0,π) 균등 회전이다. 반사/축순열은 수치 테스트에서 따로 확인했다. 각4회전은 SO(3) 완전 불변성의 증명이 아니다.

과거 fixed_iteration·joint_operator·geometry_only manifest의 (차원,family,R)와 겹치지 않음을 검사했다. 종횡비·복합체의 unequal radii 및 peanut separation도 변경했다. 비슷한 형태 계열에서의 새 파라미터 표본이며 모든 family가 미지인 대규모 외부 데이터라는 주장은 하지 않는다. 작은 구 R≈1.6–3.7은 과거에 관찰한 크기 범위이므로 post-freeze stress로만 표시하고 신규 독립 test에 합치지 않았다.

둘레/면적 GT는 연속 원형의 analytic 값(원,stadium,구,capsule,torus,분리된 다중체,ring,두 원/구 합집합)이다. 타원은 완전타원적분, triaxial ellipsoid는 Gauss-Legendre120×periodic240 quadrature다. 합집합 peanut은 노출된 원호/구면 cap의 정확식이며 ellipse 근사로 바꾸지 않았다. 입력은 그 연속체를 중심에서 binary sampling한 f다. noise GT는 **잡음을 넣기 전의 원형**이므로 이 실험은 원형 측정 robustness 질문을 다룬다. salt noise가 실제 신호인지 일반적으로 판정하는 실험은 아니다.

보정은 모든 후보의 같은 train-only 참조 원/구로 정한다. A는 pinned PBGIA-30의 원래 c와 새 train reference c를 나란히 제시한다. Conventional도 raw와 같은 참조 보정 결과가 모두 있다. Test GT로 c를 정하지 않았다. `pretest_selection.json`/sha256에 N, 생성기, 각 head, convention,c,protocol/manifest/model/calibration hash를 test 전에 잠갔다. test는 한 번만 실행했고 모든 사례를 유지했다. 중복 preset이 같은 모델을 가리켜도 새로운 독립 표본으로 세지 않았다.

## 4. 실제 학습 예산과 한계

차원/주 family(axial,tied,free)당12후보,2개 시작점이며 후반6개는 앞부분 best 주위로 범위를 줄이는 탐색이다. Train proxy는4원형×2회전×clean/1% 및 참조=17영상/차원이다. 공유 G는5개 N의 평균 L=relative-MARE+0.05*(angle/90)로 학습했다. 손실 가중치0,0.2도 같은 방문 후보에서 기록했으나 각각 독립 재최적화한 연구로 주장하지 않는다. N5/10/30의 axial generator를 각각6후보로 재학습했고 고정된 D의 G에 대해 E1 및 free E2(6후보)를 비교했다. 원래 baseline 또는 fourth-order 자체가 최선이면 이를 숨기지 않는다.

{table(tr.groupby(['dim','family','scope','fixed_gp']).agg(evaluations=('loss','count'),best_loss=('loss','min'),wall_seconds=('seconds','sum')).reset_index())}

각 최종 generator마다 동일 DE120평가/N/head로 관측자를 독립 재학습했다. 기하와 방향 DE는 각각 popsize4,maxiter9,polishFalse이며 대개 max-iteration 종료다. 이는 optimizer success/수렴 인증이 아니다. bounds는 G b[-.14,.04],t[0,.27],u[0,.08], E2 self[0,.6],second/diag[0,.17]이고 axial mass는 나머지다. 최종계수의 bounds 접근은 `selected_coefficients.csv`와 원 trial에서 확인 가능하다. 고차원 free에 같은 evaluations를 주는 편향과 proxy의 한 원형/family 제한이 있다. 특히3D E의 b=.04,u=.08은 상한에 도달해 parameter-space truncation 가능성이 분명하다. 이를 수렴한 최적해라고 부르지 않는다.

실제 전체 training 단계:

{table(trainingtime)}

전체 훈련 process CPU 기록: `{json.dumps(cpu,ensure_ascii=False)}`. 최초 실행은 process timer wrapper 전에 시작했으므로 CPU는 /proc의 user+system 누적값을0.1초 간격으로 샘플한 마지막 값이다. 정확한 종료 시점의 CPU보다 소량 작을 수 있다. walltime은 전체 최적화 루프를 직접 측정했다. generator_trials의 seconds는 후보 rollout+읽기/점수 비용이며 observer 학습은 별도 전체 훈련 시간에 포함된다. stage runner는 이후 단계 및 재현 실행의 child CPU/RSS도 기록한다. GPU는 사용하지 않았다.

같은 측정 walltime prefix의 민감도:

{table(budget)}

이 표는 동일시간 예산에 들어온 실제 탐색 prefix의 best train loss다. 별도 optimizer 재시작이나 walltime 강제 종료로 훈련한 모델이 아니며 validation/test 개선을 보장하지 않는다. 자원은 stage1–3의 분리에 우선 배정했다. G3의7-point/비분리형 확장, E3, autodiff, 새로운 loss-weight별 전체 재훈련은 수행하지 않았다. 이 범위 밖 작업을 완료한 연구처럼 표시하지 않는다.

## 5. Train / validation / N별 통제

Train 요약(주 family, retuned):

{table(train[(train['mode']=='retuned')&train.model.isin(['B','D','E','F','G'])],cols)}

Validation 요약(주 family, retuned):

{table(val[(val['mode']=='retuned')&val.model.isin(['B','D','E','F','G'])],cols)}

N별 재학습의 **동일 N, B 관측자, train 재보정 c** 비교:

{table(ns,cols)}

N별 G 재학습은 공유 G의 한계를 직접 확인하는 대조다. 한 N에서 좋은 계수가 다른 N에서도 최적이라는 가정은 하지 않았다. 최종 validation pool에는 공유 및 N별 학습 generator가 포함되므로 추가 후보 선택의 자유도도 존재한다.

Exchange와 initialization의 validation 통제(각 통제의 geometry-best N; 이 표 자체는 matched-N 인과 비교가 아니며 원본CSV에 모든N이 있다):

{table(selected_ctrl,cols)}

Raw tied와 h,D 재스케일 normalized tied의 수치 동치는 위 테스트가 검증한다. 이 두 성능이 비슷하면 raw 개선을 새로운 exchange 공간형태만의 효과라 해석할 수 없다. E_fixedG/F_fixedG는 D의 G를 고정한 exchange-only 대조다. Binary init도 EDT와 혼동하지 않는다. 공유 Kgeom=Korient simplification의 validation-only 결과는 `shared_observer_validation.csv`에 있으며 선택을 다시 하지 않았다.

![Validation Pareto](figures/validation_pareto.png)
![Fixed N](figures/fixed_N.png)

Validation 동일 관측자·동일 N에서 D의 기하 MARE는2D B0.772109%→D0.742041%(N10),3D B3.895707%→D3.892946%(N30)이었다. 작은 validation 이득이 아래 새 test에서 재현되지 않았다. 이 일반화 실패를 숨기지 않는다. 자원 우선순위에 따라 G3 확장은 이번 동결 연구에서는 수행하지 않았으며, 미실행 한계로 남긴다.

## 6. 새로운 독립 test: 생성기 효과와 최종 시스템

동일 B 관측자, **B의 validation 선택 N을 동일하게 적용**한 결과:

{table(fixed,cols)}

`frozen_c`는 B의 train c까지 동결한다. `train_c`는 같은 관측 커널을 유지하면서 해당 generator의 train 참조만 다시 보정한다. 둘의 차이는 generator가 원형별 기하성능을 개선했는지 단지 전체 gain을 바꾸었는지 해석하는 데 필요하다.

각 arm이 별도 관측자를 학습하고 validation에서 고른 N의 최종 시스템:

{table(main,cols)}

Paired unit bootstrap(B 대비, 음수 개선):

{table(bootstrap)}

CI는 원형 단위3000재표본화다. 한 원형의 모든 회전/shift/noise는 함께 묶는다. 단16원형/차원별8원형이므로 descriptive uncertainty이며 일반적 우월성 또는 유의성 검정의 대체가 아니다. 같은 generator를 가진 다른 preset을 독립 복제로 취급하지 않았다. test-family/noise별 전수 MARE,RMSE,p90,p95,worst unit,bias 및 각도 RMSE/max는 `test_family_summary.csv`, `test_noise_summary.csv`에 있다. worst image도 공개하므로 평균이 큰 국소 오차를 가리지 않게 했다. angle GT가 없는 사례 수는 각 summary의 undefined_orientation이다.

Noise별 주 arm:

{table(noise[noise.model.isin(['A','B','D','E','F'])],['dim','model','N','noise','geometry_MARE','angle_MAE','p95','worst_image'])}

![Noise](figures/noise.png)
![Rotation](figures/rotation.png)

## 7. Conventional 및 원본 Colab

{table(cl,['dim','model','mode','geometry_MARE','angle_MAE','median_seconds'])}

Clean-only conventional 대조(동일 새 test의 noise=0):

{table(cn[cn.noise==0],['dim','model','mode','geometry_MARE','angle_MAE','median_seconds'])}

2D에서는 train-calibrated Gaussian threshold/marching/PCA가 전체 noise 조건 합산에서도 default보다 기하·방향 오차가 작고 훨씬 빠르다.3D에서는 이 Gaussian 대조의 방향은 더 정확하지만 기하 오차는 더 크다. Raw Crofton/MC의 잡음 민감성만을 골라 PBGIA의 일반적 우월성을 주장하면 안 된다.

Crofton/marching-squares/voxel의 각도는 binary PCA이며 해당 둘레 계산 자체의 출력 방향이 아니다. marching-cubes는 surface-normal tensor, Gaussian gradient는 gradient tensor다. Gaussian sigma1은 test 전에 고정했다. binary_PCA 단독에는 임의 기하값을 붙이지 않았다. clean/noisy 합산 숫자는 위 표, 원시 noise분리는 `test_conventional.csv`에서 가능하다. 모델의 참조 보정과 raw estimator의 픽셀화 편향을 혼동하지 않는다. PBGIA가 더 느리거나 더 부정확한 조건도 그대로 해석한다.

원본 Colab의 **전체 함수**(후처리 totalAmount/tiltingAngle 포함),2D만:

{table(hist,['dim','model','mode','N','geometry_MARE','angle_MAE','median_seconds'])}

H_texture는 binary init,wrap,h1,D1,N5이며 원본 length의5×5 numerical kernel을 역사적 통제에 사용했다. H_length는 원본 default kernel,h1,D.1,N10,symm,conversion2/37이다. raw와 train-calibrated totalAmount를 모두 보고했다. 원본 tiltingAngle의 column-x/row-y 좌표를 배열 좌표로 변환했다. 이 과정은 신규 커널 학습이 아니며 기존 numerical coefficients를 복사한 새 방법으로 제시하지 않는다. 원본3D entire algorithm은 제공되지 않아 존재하지 않는 H3D 수치를 만들지 않았다.

## 8. 해상도·작은 물체·Fourier 해석·비용

동일한 물리적 물체를 a=1.5,1,.75에서 다시 rasterize했다. domain padding만 바꾼 실험은 별도 `padding_probe`로 표시했다. 생성 미분은1/a, geometry tensor는 물리적 미분과 volume 요소를 합쳐a^(d−2)를 반영한다. EDT width2.3은 lattice 단위로 고정했으므로 refinement가 물리적 domain wall width를 동일하게 유지하는 연속체 PDE 수렴 검증은 아니다. frozen c와 같은 train 참조를 새 a에서 다시 그린 c를 나란히 측정했다. 본문의 기하 reference는 연속체 측정량이지 spin 해의 analytic 해가 아니다. 오차가 단조 감소하지 않으면 그것도 결과다. conventional/PBGIA의 연속 GT 대비 차이는 pixelization이 유력하다는 가설을 탐색하지만 pixelization만을 정확히 분리해 정량화한 증명은 아니다.

{table(res[(res.method=='PBGIA')|(res.method=='padding_probe')])}

![Resolution](figures/resolution.png)
![Small objects](figures/small_objects.png)

작은 원/구 stress는 학습하지 않은 frozen 모델을 적용한 설명용 결과다. 과거에 본 크기 범위이며 새로운 독립 test라고 부르지 않는다. 작은 물체의 큰 기하 오차는 실제 한계이며 shape/topology 검사로 숨기거나 탈락시키지 않았다.

Fourier symbol은 i[(1−4b)sin(k_i)+2b sin(2k_i)] × product_j qhat(k_j), qhat=1−2t−2u+2t cos(k)+2u cos(2k)이다. 저파수 전개는 ik_i[1−(1/6+2b)k_i²−(t+4u)sum_j!=i k_j²+O(k⁴)]다. fourth order는 종방향 cubic을 지우지만 실제 geometry 최적해일 필요는 없다. `fourier_symbols.csv`는 고정 seed256방향의 magnitude bias/CV와 vector error를 기록한다. lattice symmetry 공변성과 SO(3) 정확 불변성은 다르다.

![Fourier anisotropy](figures/fourier_anisotropy.png)

실제 frozen CLI와 동일한 경로의 warm3회 timing(대표 family별):

{table(warm)}

첫 호출도 `timings.csv`에 따로 있다. 첫 호출은 같은 process에서 해당 mask/config의 첫 inference이며 OS cache가 완전히 비어 있는 독립 process cold-start 측정은 아니다. field와 observation 비용을 분리했고 synthetic GT 생성/디스크쓰기/학습은 추론 시간에서 제외했다. warm 총 walltime과 component 합의 차이는 Python/타이머 overhead다. peak RSS는 training 및 stage timings, 실행환경은 environment.json에 있다. 비교는 CPU 단일 BLAS thread이며 연산 family별 비용이 다르다.

![Timing](figures/timing.png)

## 9. 최종 계수·재현·제약

{table(cf)}

전체 E2 shell계수,5^d 커널 배열은 `kernel_coefficients.csv`, 모든 N별 head와 generator는 `models.json`에 있다. N에 대한 적응 정지나 test-dependent fallback 없이 추론한다. CLI의 default가 없을 경우 우선순위 preset을 명시해야 하며 PBGIA-30으로 몰래 교체하지 않는다. inference의 ambiguous flag는 설명용이며 결과 탈락이나 모델 교체를 하지 않는다.

`python run_all.py --all`로 새로운 빈 output에서 smoke/tests→train→validation/freeze→single test→diagnostics→report를 실행한다. 단계별 명령과 pinned requirements는 README에 있다. 이미 test를 계산한 output은 훈련/선택/시험을 덮어쓰지 못하게 한다. 큰 field checkpoint는 게시하지 않으며 manifest/config/seed로 재생성한다. `run_logs`, calibration,trial,freeze hashes,rawCSV,figures와 source provenance를 함께 제공했다. 추가 train-only 재현에서2D/3D B모델의 N5/10/20/30/40 모든 기하·방향 head 계수와 tensor convention이 정확히 재현되었으며 `reproduction_check.json`에 실측 비용을 기록했다. 이 확인은 test를 다시 평가하거나 선택을 바꾸지 않는다.

제약: synthetic binary/isotropic spacing만, family당 train 원형1개와 매우 적은 rotation,3D orientation train은 capsule에 의존, 작은 noise 확률의 특정 주입 모형,좁은 separable family와 유한 탐색 budget,관측 head120평가의 max-iteration 종료,후보 수 증가에 따른 validation 편향이 있다. 두 weight의 추가 기록은 완전한 독립 재학습이 아니다. 별도 equal-walltime 최적화/G3/실제 영상/anisotropic spacing/GPU autodiff는 미실행이다. 모든 failure는 기하·각도·기본 수치 정합성의 의미로만 보고하며 원본 마스크 복원이나 위상 보존을 성공 기준으로 사용하지 않는다. 위상 현상은 이 공통장의 중요한 향후 활용이지만 이번 손실 제약은 아니다.
'''
 (OUT/'RESEARCH_REPORT_KO.md').write_text(text)
 print('REPORT COMPLETE',flush=True)
if __name__=='__main__':report()
