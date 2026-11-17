from evaluation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def tbl(df,cols=None):return (df[cols] if cols else df).to_markdown(index=False,floatfmt='.5g')
def fig(name):plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=170);plt.close()
def report():
 assert (OUT/'test_audit_2d.json').exists();f=check_lock();models=json.load(open(OUT/'models.json'));test=pd.read_csv(OUT/'calibration_free_summary.csv');trace=pd.read_csv(OUT/'train_optimizer_trace.csv');trials=pd.read_csv(OUT/'train_trials.csv');val=pd.read_csv(OUT/'validation_summary.csv');sweep=pd.read_csv(OUT/'reference_ruler_sweep.csv');sensitivity=pd.read_csv(OUT/'ruler_sensitivity_summary.csv');rawtest=pd.read_csv(OUT/'test_raw_2d.csv');curv=pd.read_csv(OUT/'test_by_curvature.csv');timing=pd.read_csv(OUT/'timings.csv');cost=json.load(open(OUT/'training_cost.json'));partial=json.load(open(OUT/'status_3d.json'));ratios=pd.read_csv(OUT/'pairwise_q_ratios.csv');boot=pd.read_csv(OUT/'paired_bootstrap.csv');coeff=pd.read_csv(OUT/'kernel_coefficients.csv');fourier=pd.read_csv(OUT/'fourier_symbols.csv');transfer=pd.read_csv(OUT/'spacing_padding_small.csv');ranks=sweep.groupby(['model','noise']).MARE_rank.agg(['min','max','std']).reset_index();ranks.to_csv(OUT/'ruler_rank_variability.csv',index=False)
 selected=[]
 for name,c in f['configurations'].items():selected.append(dict(preset=name,source_model=c['model'],N=c['N'],coefficients=c['p'],m1=float(coeff[coeff.model==c['model']].first_moment.iloc[0]),validation_objective=c.get('validation_objective',np.nan)))
 pd.DataFrame(selected).to_csv(OUT/'selected_coefficients.csv',index=False)
 figu,axs=plt.subplots(1,2,figsize=(11,4))
 for name,g in trace.groupby('model'):
  axs[0].semilogy(g.evaluation,g.objective,label=name,alpha=.8);axs[1].semilogy(g.evaluation,g.gradient_norm,label=name,alpha=.8)
 axs[0].set(xlabel='Objective/gradient evaluations',ylabel='Declared training objective');axs[1].set(xlabel='Evaluations',ylabel='Gradient norm');axs[0].legend(fontsize=6,ncol=2);fig('optimization')
 figu,axs=plt.subplots(1,3,figsize=(15,4))
 for name,g in curv[(curv.noise==0)&curv.model.isin(['G4','selected_w0','selected_w1','selected_w4'])].groupby('model'):
  axs[0].scatter(g.curvature_proxy,g.q_mean,label=name);s=g.sort_values('curvature_proxy');axs[1].scatter(s.curvature_proxy,100*(s.q_mean/np.exp(np.log(s.q_mean).mean())-1),label=name);axs[2].scatter(s.curvature_proxy,s.q_CV_pct,label=name)
 axs[0].set(xlabel='1/R scale proxy (not universal minimum curvature radius)',ylabel='Uncalibrated q=I/P');axs[1].set(xlabel='1/R scale proxy',ylabel='q/geometric-mean(q)-1 (%) diagnostic only');axs[2].set(xlabel='1/R scale proxy',ylabel='Within-unit rotation/shift CV (%)');axs[0].legend(fontsize=8);fig('q_curvature')
 figu,axs=plt.subplots(1,2,figsize=(11,4))
 for name,g in sweep[(sweep.noise==0)&sweep.model.isin(['G4','legacy','pilot','selected_w0','selected_w1','selected_w4'])].groupby('model'):
  c=g[g.reference_family=='circle'].sort_values('reference_R');axs[0].plot(c.reference_R,c.c,'o-',label=name);axs[1].plot(c.reference_R,c.MARE,'o-',label=name)
 axs[0].set(xlabel='Train-only reference circle R',ylabel='Diagnostic c=P_ref/I_ref');axs[1].set(xlabel='Reference circle R',ylabel='Held-out MARE (%)');axs[0].legend(fontsize=7);fig('reference_radius')
 plt.figure(figsize=(10,4));families=['circle','ellipse','stadium','rounded_box']
 for j,name in enumerate(['G4','selected_w0','selected_w1','selected_w4']):
  a=sweep[(sweep.noise==0)&(sweep.model==name)].groupby('reference_family').MARE.agg(['mean','min','max']).reindex(families);xx=np.arange(4)+(j-1.5)*.12;plt.errorbar(xx,a['mean'],yerr=np.stack([a['mean']-a['min'],a['max']-a['mean']]),fmt='o',capsize=3,label=name)
 plt.xticks(range(4),families);plt.ylabel('Diagnostic MARE (%) across train rulers');plt.title('Circle: 5 radii; other families: 1 ruler each (ranges, not CIs)');plt.legend(fontsize=8);fig('reference_family')
 figu,axs=plt.subplots(1,2,figsize=(11,4))
 for name,g in test[test.model.isin(['G4','legacy','pilot','selected_w0','selected_w1','selected_w4'])].groupby('model'):
  axs[0].plot(100*g.noise,g.q_CV_pct,'o-',label=name);axs[1].plot(100*g.noise,g.angle_MAE,'o-',label=name)
 axs[0].set(xlabel='Salt flip probability (%)',ylabel='Between-prototype q CV (%)');axs[1].set(xlabel='Salt flip probability (%)',ylabel='Axis MAE (degree)');axs[0].legend(fontsize=7);fig('noise')
 figu,axs=plt.subplots(1,2,figsize=(11,4))
 c=f['configurations']['selected_w1']
 for ax,name,p in [(axs[0],'G4',G4),(axs[1],'selected_w1',np.array(c['p']))]:
  K=kernels(p)[1];im=ax.imshow(K,cmap='RdBu_r');ax.set_title(name+' unnormalized Kx');plt.colorbar(im,ax=ax)
 fig('kernels')
 plt.figure(figsize=(10,4));pivot=sweep[(sweep.noise==0)&sweep.model.isin(['G4','legacy','pilot','selected_w0','selected_w1','selected_w4'])].pivot(index='model',columns='reference_unit',values='MARE_rank');plt.imshow(pivot,cmap='viridis',aspect='auto');plt.xticks(range(len(pivot.columns)),pivot.columns);plt.yticks(range(len(pivot.index)),pivot.index);plt.colorbar(label='MARE rank (smaller better)');plt.xlabel('Train-only reference unit');fig('reference_rank')
 plt.figure(figsize=(9,4))
 for name,g in transfer[transfer.protocol.isin(['lattice_width_fixed','physical_width_fixed'])].groupby(['model','protocol','calibration']):plt.plot(g.spacing,abs(g.error_pct),'o-',label='/'.join(name))
 plt.xlabel('Physical grid spacing a');plt.ylabel('Absolute diagnostic error (%)');plt.legend(fontsize=6,ncol=2);fig('spacing')
 plt.figure(figsize=(9,4))
 for name,g in fourier[fourier.model.isin(['G4',c['model'],'legacy','pilot'])].groupby('model'):plt.plot(g.k,g.mean_gain,'o-',label=name)
 plt.xlabel('Wave number magnitude');plt.ylabel('Raw generator symbol magnitude');plt.legend(fontsize=7);fig('fourier')
 clean=test[test.noise==0];maincols=['model','N'] if 'N' in test else ['model'];statcols=['model','units','q_mean','q_CV_pct','log_q_std','q_CV_CI_low','q_CV_CI_high','angle_MAE','angle_RMSE','angle_p95','angle_count'];ratiosummary=ratios.groupby(['model','noise']).agg(ratio_min=('ratio','min'),ratio_max=('ratio','max'),max_abs_log_ratio=('log_ratio',lambda x:abs(x).max())).reset_index();warm=timing[timing.kind=='warm'].groupby('model').agg(median_seconds=('wall_seconds','median'),mean_seconds=('wall_seconds','mean'),field_seconds=('field_seconds','mean'),observer_seconds=('observer_seconds','mean')).reset_index();differences=[]
 for name in ['selected_w0','selected_w1','selected_w4']:
  a=clean[clean.model==name].iloc[0];b=clean[clean.model=='G4'].iloc[0];differences.append(f"- {name}: test CV(q) {a.q_CV_pct:.5f}% (G4 {b.q_CV_pct:.5f}%), log-q std {a.log_q_std:.6g} (G4 {b.log_q_std:.6g}), 각도 MAE {a.angle_MAE:.5f}° (G4 {b.angle_MAE:.5f}°).")
 # Train and validation figures are distinct from fresh test; no calibrations enter them.
 text=f'''# 기준자 독립적 PBGIA 생성 커널 최적화: 실제 2D 연구와 검증된 3D 부분 구현

## 1. 결과와 연구 범위

**주 실험은2D에서 완료했다.3D는12orbit 커널 및 미분 수치 검증만 완료했으며,3D 최적화·독립 표면적 시험 결과는 없다.** 새로운 branch/directory에서 원본과 과거 결과를 바꾸지 않고 실행했다. 기준 원 하나의 보정 MARE를 학습하지 않았다. 모든 주 arm은 NN 교환합4,고정 G4 관측자,N10,같은 EDT 초기화,J1,D1,h2를 공유하며 **생성 미분6계수만** 바뀐다.

{chr(10).join(differences)}

Noise-weight1의 q-CV 차이는−0.39895pp이지만 원형 bootstrap95% 구간[−1.72717,+1.21215]가0을 포함한다. **학습·validation에서의 q분산 감소와 달리,보편적인 기준자 독립성이 확립됐다고 결론내릴 수 없다.** Clean reference c의CV는G4의1.0422%에서학습모델1.9187%로 커졌고,참조중최악MARE는2.7066%→4.2698%로 악화됐다.1% noise의paired RMS(log I change)도0.006347→0.010990으로 악화됐다. Noise를 넣은 후 q-CV가 내려가는 현상은 원형별 오차 상쇄일 수 있으므로 이를 곧바로 noise robustness 향상으로 해석하지 않는다. Noise-weight4는학습모델내paired드리프트를줄였지만G4보다좋지는않았다.

이는 기준자 선택과 무관한 같은 신규 test의 직접 비교다. 개선 여부는 아래 train/validation/test 표와 paired 원형 bootstrap으로 판단한다. c 또는 최저 test MARE를 주는 원을 골라 이 결론을 만들지 않았다. 원형17개는 일반적인 영상 모집단을 대표하는 대규모 외부 검증이 아니며 null/negative 항목은 그대로 남긴다.

동결한 구성:

{tbl(pd.DataFrame(selected))}

N10의 세 noise 설정을 주 결과로 보고, 별도 selected_N은 validation의 N5/10/20/30 민감도다. G4_matched_N은 그 구성의 동일 반복수 대조다. 서로 다른 N의 결과를 생성 커널만의 인과 효과로 해석하지 않는다. 동일 source/p/N의 preset 중복을 독립 복제로 세지 않았다.

## 2. 실제 원본과 이전 결과 검토

실제 참조 branch `codex/generator-first-20261116`의 commit `54287cdc0b1e4ce3fefecb2df42467c190985355`에서 operators/engine/study/observer/data/runner/report/test_summary/pretest_selection의 byte hash를 확인했다. Source review JSON에 경로/크기가 있다. 기존 joint/fixed numerical 함수 및 Colab도 확인했으며 screening 정책은 새 실행 경로에 가져오지 않았다. 과거 pilot 코드/비교 CSV/noise/optimizers/selection과 radius sensitivity는 history only로 읽었다.

첨부 원본 TextureFilterTrain5.ipynb 저장 출력:3×3 k11≈.30414468, successTrue,nit6,nfev18;5×5 k20≈.57063894,k21≈.33693168,k22≈.08973684,k12≈.09205035,successTrue,nit19,nfev115. Notebook cell31/38의 저장 출력을 `notebook_saved_outputs.json`에 추출했다. 이것은 이번 실행에서 원본451회전 최적화를 다시 수행한 결과가 아니다.5×5 matrix 배치(cell37/39)를 명시적으로 검사했다. 원본 convolve2d의 Kx 부호는 본 correlation Kx의 반대다. k12와 k21의 위치를 바꾸지 않았으며 unit ramp/transpose/convolve 대조로 검증했다. 파일럿은 단일 원 r10.3 보정에 기반하고 두 run 모두16회 반복 상한이므로 신규 독립 test나 수렴 인증으로 사용하지 않았다. 과거 radius sensitivity에서G4의 MARE는1.192–1.621%,learned5는0.542–1.815%였고,c범위는각각0.374422–0.381067와0.479120–0.491269였다. 이는과거동일test에서표준자변경이측정순위해석을바꿀수있음을보여주지만,아래새test의수치와직접합치지않는다.

## 3. 데이터·동역학·차분과 정확한 미분

{tbl(manifest().groupby(['split','role']).agg(prototypes=('unit','nunique'),images=('case','count')).reset_index())}

Train/validation/test는 각각17개의 **서로 다른 continuous prototype**이다. 각 split에 원5크기,ellipse/stadium/rounded_box/ring/pair/peanut 각2크기·파라미터를 분산했다. Train-only 참조는 별도8개(원5반지름,타원/스타디움/rounded_box 각1)다. 모든 prototype의3회전×2독립shift×noise0/.002/.005/.01=24파생 영상을 같은 split에 둔다. Seed={SEED}. 이전 fixed/joint/generator_first/geometry_only manifest 및 첨부 pilot test_specs/radius 목록을 읽고 새 prototype과 분리했다. old manifest 사본도 포함하므로 외부 환경에서 같은 검사를 재현할 수 있다. 같은 family 안의 파라미터 외삽이며 새로운 미지 family만의 검증은 아니다.

훈련 proxy는 train17원형 모두의 crossed two pose(rotation0/shift0,rotation1/shift1)다. Full validation/test의6pose와 다르며 회전 sampling이 제한된다. Noise training은4확률을 paired로 포함한다. 동일 uniform 난수장의 threshold를 사용해 오염이 nested되며, 서로 독립인 noise realization4개라고 세지 않는다. 잡음은7픽셀 collar 안쪽에 넣었다. GT는 **잡음 주입 전 연속 원형의 둘레**다. 염점이 실제 신호인지 보편적으로 판단하는 실험은 아니다.

원/타원/스타디움/ring/두 분리 원/두 원 union/rounded rectangle의 analytic boundary 또는 완전타원적분을 썼다. Peanut은 원호 노출 길이4R(π−acos(separation/(2R)))이며 타원으로 근사하지 않았다. rounded rectangle의 반장축a,b와 corner r의 둘레는4(a+b−2r)+2πr다. 일반 mask 고유의 진짜 둘레가 있다고 주장하지 않는다. test_by_curvature의1/R은 **scale proxy**이며 ellipse의 최소 곡률반경이나 peanut 교차점의 비매끄러움을 모두 나타내는 값이 아니다.

매 단계 H_v=NN(v)+G(phi)*s, H_s=NN(s)+2f, n=H/||H||. f불변, 현재 s의 동적 미분, synchronous exchange, periodic, norm≤1e−12이면 이전 상태를 유지한다. div(v)/grad(f)/Laplacian 대체는 없다. 기본 초기화는 이진 EDT/tanh width2.3 lattice이며 continuous GT distance는 유출하지 않았다.

6계수 순서는[w01,w02,w11,w12,w21,w22],x방향 홀수/y방향 짝수,Ky=Kx.T. **G4=[2/3,−1/12,0,0,0,0]를 정확히 포함한다.** 계수 bounds[−1.5,1.5] 이외에 gain/1차 moment/Σ|K| 정규화는 없다. vector 성분은 배열 row/column 순서이며 tensor 최소 고유축의 half-angle convention을 검사했다. 원/ring/pair에는 방향 GT를 부여하지 않고 ellipse/stadium/rounded_box/peanut만 평가했다. 작은 예측 gap은 CSV로 기록하며 낮은 confidence라는 이유로 사례를 제거하지 않는다.

PyTorch double precision reverse-mode autodiff + SciPy L-BFGS-B를 사용했다. 메모리 제한을 위해 chunk8의 forward I/theta를 저장하고, 작은 outer loss의 ∂L/∂I,∂L/∂theta를 구한 다음 각 chunk에서 정확한 chain rule을 역전파했다. 이는 approximation/minibatch stochastic objective가 아니다. 별도 NumPy forward-mode 정확 Jacobian과 tiny finite difference,PyTorch 직접 graph를 비교했다. 정규화의 도함수는(I−nnᵀ)dH/||H||,zero-H 분기는 이전 상태의 도함수다. Branch 경계/고유축 퇴화에서는 고전적 smoothness를 보장하지 않는다. I≤1e−14 또는 비유한 도함수는 수치 실패이지 geometry penalty가 아니다.

## 4. 기준자 없는 목적·예산·실제 최적화 종료

Clean 원형 평균q의 log variance + clean 원형 내부I의CV² +0.0001×mean((axis angle/1°)²) +w_noise×paired(log I_noisy−log I_clean)². 각 원형은 같은 가중치다. θ항은 orientation-defined 원형만 포함한다. w_noise=0,1,4를 validation 전에 선언했고 각각 네 시작값(G4,legacy,pilot,seeded perturbation)을 독립 학습했다. Clean-only는 noisy input을 전혀 읽지 않는다. **선택한 reference circle의 c나 MARE는 이 목적에 들어가지 않는다.** I의 공통 양의 배율 변화에 log-q 분산이 불변임을 테스트했다. q가 모두 같은 상수면 그 항이0이다. q절대값은 목표가 아니다.

추가 separable 5-point family는 linear moment1을 고정한2차 대조다. G4를 정확히 포함하며 동일 maxiter/maxfun으로 clean/noise1에서 학습했다. 주 비분리형 결과의 amplitude를 그에 맞추어 다시 정규화하지 않았다.

각 run은 maxiter60,maxfun100,maxls12,ftol1e−11,gtol1e−6이다. SciPy maxfun은 진행 중 line search를 끝내면서 소폭 초과할 수 있는 soft budget이며 실제 nfev를 아래에 공개한다. projected-gradient 종료와 전체 gradient norm은 다른 값이다. maxiter/maxfun 종료를 수렴이라고 쓰지 않는다. optimizer state의 계수/gradient norm/active bounds/모든 평가 trace를 보존했으며 SciPy 내부 L-BFGS history의 bitwise continuation checkpoint는 제공하지 않는다.

{tbl(trials)}

전체 training 실측: `{json.dumps(cost,ensure_ascii=False)}`. CPU1 thread이며 GPU는 쓰지 않았다. chunk화 전 대표136영상 Jacobian/autograd 비교를 먼저 실시하고 예산을 동결했다. Protocol/manifest가 첫 고비용 optimization 전에 기록되었고 smoke를 통과했다. 단계별 CPU/wall/cumulative child peak RSS는 stage_timings에 있다. 학습 시작 전 최초 smoke/hash snapshot과 최종 게시 source hash는 다른 목적이다.

![Optimization](figures/optimization.png)

## 5. Train·validation 선택과 새 test

Train/validation 모든 후보의 N10 지표:

{tbl(val[val.N==10],['split','model','N','q_CV_pct','log_q_std','prop','rotation','angle','noise','objective_w0','objective_w1','objective_w4'])}

동일 objective로 G4를 항상 anchor로 남겼다. 후보별 clean log-q variance와 해당 weight의 train/validation objective가 G4보다 나쁘면 선택하지 않는다. 허용 후보 중 validation objective 최소를 선택한다. 기하 MARE/c/reference를 보고 순위를 고르지 않는다. validation_pareto는prop/rotation/angle/noise의4목표를 기록한다. N5/10/20/30 sensitivity는 validation_summary의 모든N에 있다. test 전에 model_freeze_2d.json에 p/N/weight/observer 정의/protocol/models/manifest와참조set hash를 잠갔다. 이미 test가 존재하면 학습/선택/시험을 덮어쓸 수 없다.

독립 clean test:

{tbl(clean,statcols)}

Clean/noise별:

{tbl(test,['model','noise','q_CV_pct','log_q_std','rotation_shift_CV_pct','angle_MAE','angle_p95','angle_count'])}

Paired prototype bootstrap의 CV차이(학습−G4,음수 개선):

{tbl(boot)}

각 q-CV CI는17개 원형을2000회, paired 차이는3000회 재표본화했다. 모든 rotation/shift/noise 파생은 같은 원형 block에 묶는다. 작은 prototype 수와 같은 family의 크기 반복을 고려하면 descriptive uncertainty이며 실제 영상 일반성이나 유의성의 보증은 아니다. test_raw에는 모든I,q,angle,gap와finite state unit error가 있다. q는 양수지만 동일 q를 억지로 강제하는 물리적 보존 법칙이 존재한다고 주장하지 않는다. test_by_family/noise/curvature에서 subgroup 악화를 숨기지 않는다.

![Q and scale](figures/q_curvature.png)
Paired noise drift(오차 상쇄와 실제 입력 민감성을 구분):

{tbl(pd.read_csv(OUT/'paired_noise_drift.csv'))}

![Noise](figures/noise.png)

## 6. 표준자 반지름·형상 의존성과 MARE 진단

이미 동결한 I에 train-only8참조의 c=P_ref/I_ref를 모두 적용했다. **최고 test MARE의 reference를 골라 최종 성능으로 표시하지 않는다.** 아래 median/worst는 미리 고정한 참조 집합에 대한 민감도다. MARE는 continuous prototype truth 기준이며 calibration-free CV와 다른 질문이다.

{tbl(sensitivity)}

Clean ref별 전수(반지름과 family 포함):

{tbl(sweep[sweep.noise==0],['model','reference_family','reference_R','c','MARE','bias','p95','worst','MARE_rank'])}

모델별 ref 교체에 따른 순위 범위:

{tbl(ranks[ranks.noise==0])}

순위가 바뀌는 경우 reference-dependent 측정 순위임을 뜻한다. q/주 손실은 참조와 무관하게 고정되어 있다. 첨부 historical r10.3 보정 수치를 이번 독립 결과와 직접 합치지 않았다. reference circle 반지름을 바꿀 때 c와 MARE가 달라질 수 있음은 수치 검사와 이 sweep이 보여준다. 참조 원형마다1pose만 사용했으므로 c의 radius/shape 의존성에는 각 참조의pixel phase도 섞인다. 반지름만의 완전 통제 효과라고 해석하지 않는다. Pairwise q ratios의raw CSV는 기준자 없는 원형 간 spread를 그대로 보존한다.

{tbl(ratiosummary)}

![Reference radius](figures/reference_radius.png)
![Reference family](figures/reference_family.png)
![Reference rank](figures/reference_rank.png)

## 7. Gain·noise·회전/shift·물리 spacing·비용

각 최종커널의 계수/1차 moment는 selected_coefficients와kernel_coefficients에 있으며 **아무 진폭 정규화도 하지 않았다.** NN exchange합은 항상4라서 생성 gain 변화에 따른 실효 D/h 변화는 실제 학습 공간의 일부다. hypothetical raw tied sum=Σ_i G_i²는 해석용으로만 기록했다. 실제 tied/free exchange 업데이트를 이번 주 실험에서 실행한 것처럼 주장하지 않는다. 원본처럼 K→αK라면 tied exchange→α²KJ가 되므로 legacy full algorithm과 동역학이 다르다.

![Kernels](figures/kernels.png)
![Fourier](figures/fourier.png)

fourier_symbols는 128각도의 raw complex symbol magnitude/각도CV 및m1을 기록한다. 분석에서 scale을 비교하더라도 field 계수를 renormalize하지 않았다. 학습 과정에서 일부 해의m1은0에 가까워지거나 음수가 되었다. 이를+1로 재정규화하지 않았고 자동으로 탈락시키지 않았다. 이 경우 표준적인+gradient의 저파수 일관성 또는 물리적 연속체 해석을 주장할 수 없다. I는모든벡터/스칼라 성분의 제곱 gradient 합이며 양수/유한성으로 수치 퇴화를 검사했다.

{tbl(rawtest[rawtest.noise==0].assign(scalar_fraction=lambda z:z.I_scalar/z.I).groupby('model').agg(I_min=('I','min'),mean_scalar_fraction=('scalar_fraction','mean')).reset_index())}

I_scalar/I_vector 원시 분해를 추가해 m1이작아도I가0으로 붕괴하지 않았는지 확인할 수 있게 했다. m1만작다는이유로 수치적I=0과 동일시하지 않는다. Noise 악화와 고주파 응답의 상관은 가능한 설명이지 노이즈 원인을 인과적으로 증명한 것은 아니다. 분리된 rotation/shift CV는 같은 shift에서 rotation 변화,같은 rotation에서shift 변화로 계산한다. 그 각각의 각도·변위표본이 매우 적다는 한계가 있다.

Physical spacing a=.7,1,1.4에서 같은 물체를 새로 binary rasterize했다. EDT width를 lattice2.3 고정한 조건과 physical2.3 고정한 조건을 따로 비교하고 생성 미분1/a,관측tensor volume/derivative 합성 a^(2−2)=1을 적용했다. c 고정과 동일 train reference를 같은a로 재표본화한 c(a) 두 경로다. Domain padding80→112는 spacing1 고정의 boundary probe이며 refinement가 아니다. 작은 원은 post-freeze stress로 주 test 평균에 합치지 않았다. 단순 padding을 연속체 수렴이나 analytic spin 해로 부르지 않는다.

{tbl(transfer)}

![Spacing](figures/spacing.png)

실제 field/observer 분리 warm timing(5family별3반복):

{tbl(warm)}

first_call_in_process도 timings에 공개했다. 독립 process/OS cold cache startup을 측정한 것은 아니다. 이 속도는 NumPy/SciPy 추론 경로이고 학습은PyTorch다. tensor/GT 생성·disk쓰기/최적화 시간을 inference latency에 넣지 않았다. 같은 NN,N10,고정observer의 causal 대조이나 학습커널의 nonzero orbit 수에 따른 계산량이 달라진다. 코드에서0계수도 일반경로로 계산하므로 sparse-specialized G4 implementation보다 불리할 수 있다.

## 8. 3D 완료 상태와 미실행 항목

`{json.dumps(partial,ensure_ascii=False)}`

12parameter의5³ orbit는 미분축 odd,두 횡축 even,횡축 교환 대칭이며 G4를 정확히 포함한다. 축순열6개/반사 및 tiny3D3-step autodiff 대 유한차분을 검사했다. **이 수치 테스트는 표면적 정확도,3D 최적화,독립 area 검증이나 표준 구 sweep의 대체가 아니다.**3D fresh manifest/optimizer/validation/freeze/test는 남아 있다.2D 숫자를3D 결과로 복제하지 않았다. 현재 blocker는2D14run 완결을 우선한 사전 예산 배정이며,8GiB CPU에서3D volumetric training의 별도 최적화 예산을 배정하지 않았다.32³,N10의 단일 합성 integral-gradient runtime/RSS probe는 status에 기록했지만 전체3D train/area accuracy 실험의 비용이나 성능을 대신하지 않는다.3D 하드웨어 불가능의 증명은 아니다.

7×7 확대,교환 학습,geometry/orientation observer 재학습도 미실행이다. ablation_fixed_observer는 실제 주 실험의 raw CSV이고 ablation_retrained_observer 파일을 만들어 미실행을 숨기지 않았다. 5×5 validation 개선을7×7 일반화 보장이라고 주장하지 않는다. 이 test를 본 후 확장 후보를 같은 test에서 다시 선택하지 않는다.

## 9. 재현·정직한 해석

README의 smoke/tests→train2d→validate2d→freeze2d→test2d→diagnostics/report2d 명령으로 재현한다. 큰 field checkpoint는 저장/게시하지 않으며 mask seeds/model/source로 재생성한다. Source SHA256,노트북/파일럿 provenance,protocol,manifest,optimizertrace,rawCSV,freeze hashes와run_logs를 함께 제공한다. 기존 report/data는 수정하지 않고 별도PR로 제출하며 merge하지 않는다. 기존 single-ruler loss를 새 데이터에서 다시 최적화하는 별도 ablation은 수행하지 않았으므로,이전 결과와 달라진 원인을 최적화 알고리즘 하나 또는 목적함수 하나로만 인과적으로 분해했다고 주장하지 않는다. 추론 CLI는 `python code/infer.py mask.npy --preset selected_w1 --out prediction.json`이며 기준자 없이 I와 axis를 출력한다. 일반 mask의 P를 모르는 상태에서 q나 둘레를 자동으로 지어내지 않는다.

한계: synthetic2D binary 입력만,17train/17validation/17test prototype,극히 제한된rotation/shift,특정salt contamination모형,proxy학습,고정bounds/목적가중치,수치 eigenaxis 근처 비평활성,reference set에 의존하는MARE진단,데이터/선택 자유도가 있다. CV개선이 실제영상모든경계에 대한 참둘레나완전회전불변성을 보장하지 않는다. 소형객체의픽셀화/큰곡률은 설명가설이며spacing/noise결과와분리해해석한다. 재구성·Dice/IoU·연결성·Euler·degree·벽너비·잔차를손실/선택/fallback으로쓰지않았다. 위상현상은공통자화장의중요한잠재응용이지만이번최적화제약은아니다. 부정적결과와3D미완료를포함하는것이이연구의실제완료범위다.
'''
 (OUT/'RESEARCH_REPORT_KO.md').write_text(text);print('REPORT2D COMPLETE',flush=True)
if __name__=='__main__':report()
