# Codex 연구 의뢰서 — 보편적 비례성 중심의 PBGIA 생성 커널 재최적화

**2026-11-17 / 연구 우선순위:** 형상·곡률·표준자 선택에 독립적인 기하 측정량을 얻는 생성 자화장 학습. **이전 2D/3D generator-first 결과가 새 학습에서 개선되지 않았던 원인을 실제 연속 최적화로 검증한다.**

## 0. 가장 먼저 할 일: 실제 코드·노트북·이전 결과 검토

1. 저장소 `cywon-commits/Physics-Based-Geometric-Image-Analysis`, 브랜치 `codex/generator-first-20261116`에서 실제 파일과 commit SHA를 확인한다.
   - `generator_first/code/operators.py`, `engine.py`, `study.py`, `observer.py`, `data.py`, `run_all.py`, `output/RESEARCH_REPORT_KO.md`, `test_summary.csv`, `pretest_selection.json`.
   - 이전 코드 `joint_operator/reference/texturefiltertrain5 (1).py`, `lengthorientationapplication (1).py`와 `joint_operator/code/`.
   - **첨부된 원본 `reference/legacy/TextureFilterTrain5.ipynb`의 저장된 결과(output)**. 3×3 `k11≈0.30414468`, 5×5 `k20≈0.57063894,k21≈0.33693168,k22≈0.08973684,k12≈0.09205035`; 각 L-BFGS-B 성공·`nfev` 기록을 직접 재확인한다. 커널 인덱스·convolve/correlate 부호를 코드로 검증한다.
   - `reference/pilot_2d/`의 코드·보고서, `reference/RADIUS_SENSITIVITY_HISTORICAL.csv`. **이는 이미 보았던 과거 시험이며 새로운 독립 test로 쓸 수 없다.** 수치 재현과 실패 원인 설명에만 사용한다.
2. 새 브랜치/독립 디렉터리 `reference_invariant_generator/`를 만들고 원본 파일·old results는 수정하지 않는다. 모델 선택 전에 프로토콜·manifest·seeds를 동결하라. 새로운 clean/noise/rotation/test 결과를 test 후 다시 최적화하지 않는다.
3. `--smoke`로 2D 정확한 G4 reproduction, parity/부호, 관측 적분, 수치 도함수 vs autodiff, split 분리, 보정 함수 등 기본 수치 정합성을 먼저 시험한다.

## 1. 연구 질문을 다시 정의한다

자화장 (2D에서는 3성분, 3D에서는 4성분)에서 얻은 고정 관측량을 `I[f]=Tr(T[f])`라고 정의한다. 합성 연속 원형의 실제 둘레 `P[f]`, 3D 표면적 `A[f]`는 **평가용 정의**이며, 일반 이진 마스크에 고유한 '진짜 둘레'가 있다는 뜻이 아니다.

주 목적은, 한 **특정 표준 원**으로 보정한 오차가 아니라

\[
q_i^{(2)}=I_i/P_i,\qquad q_i^{(3)}=I_i/A_i
\]

가 형상·크기·곡률·회전·아화소 이동에 걸쳐 **하나의 양의 상수에 근접**하는 것이다. `q`의 절대값은 최적화 목적이 아니며, 이로 인해 생성 커널의 total gain을 임의 정규화하지 않는다.

**반드시 답할 질문:**
- G4를 정확히 포함하는 비정규화 5×5 학습공간에서 G4보다 **train 및 validation의 q-dispersion**을 실제로 낮추었는가?
- 같은 관측 커널·동일 반복수·동일 NN 교환일 때도 개선이 남는가?
- 캘리브레이션 기준 원의 반지름·모양을 바꾸면 `c`, MARE, 알고리즘 순위가 얼마나 변하는가?
- noise/작은 물체/해상도/새 형상에서 비례성이 유지되는가?
- 3D에서도 같은 원리로 표면적 비례성을 얻는가? 안 되면 어떤 물리·계산적 한계 때문인가?

## 2. 반드시 보존할 생성 동역학과 observer

\[
\mathbf H_v^{t}=J(K_J*\mathbf v^{t})+D(G_\phi*s^{t}),\quad
H_s^{t}=J(K_J*s^{t})+h f,\quad
\mathbf n^{t+1}=\mathbf H^t/\|\mathbf H^t\|.
\]

- 입력 `f` 고정, gradient는 **현재의 `s(t)`**에 적용, `J=1, D=1, h=2`, 2D 기본 `N=10`, EDT/tanh(width=2.3 lattice pixels) 초기장, periodic boundary. `norm H≤epsilon`이면 이전 상태. 원본 mask 재구성은 의무 아님.
- 주요 비교에서 **교환 커널은 NN axial(`K_J=sum_{NN}`)**, 관측 미분은 G4 고정: `T_ij = Σ_{r,a} (D_i^obs n_a)(D_j^obs n_a)` / `I=Tr T` / 방향은 고유축(2D min axis)으로 통일. 동일한 `N,init,observer`를 유지. 이 조건에서 생성 미분만 바뀌면 독립적인 인과적 비교가 된다.
- 3D는 `d=3` 축과 normal tensor convention 점검. 무방향 구/원/ring/torus에는 임의 방향 정답을 부여하지 않는다. 타원체 중 두 축이 비슷한 경우 eigengap / ground-truth 주축의 유일성을 별도 명시한다.
- **기하학적 크기/방향 observer의 최적화는 2차 단계**다. 먼저 frozen observer를 사용한다. 추가 observer retraining 결과와 generator effect는 반드시 별도 표로 보고한다.
- **총합 정규화 의무 없음.** 생성 미분 1차 모멘트와 `K_J` 전체합은 *관측·해석용*으로 기록한다. 단 상수에 대한 미분 0, parity/symmetry, NaN/Inf 없음, 정규화된 `n`, `f` 불변은 코드 정합성 테스트로 필수.

## 3. 생성 미분의 정확한 중첩 공간

### 3.1 2D 비분리 5×5: 6개의 독립 계수

`K_x[dy,dx]`는 **x 방향 홀수 / y 방향 짝수**이며, `K_y=K_x.T`. 독립계수 순서는 `[w_01,w_02,w_11,w_12,w_21,w_22]` (`|dy|=0,1,2`, `dx=1,2`).

\[
K_x(dy,dx)=\operatorname{sgn}(dx)w_{|dy|,|dx|},\quad dx\in\{\pm1,\pm2\};\quad K_x(dy,0)=0.
\]

직접 `torch.conv2d`가 **cross-correlation**임을 감안하여 물리적 x 방향 부호를 일치시켜라. 테스트:

\[
\phi_{G4}=[2/3,-1/12,0,0,0,0],
\]

이는 정확한 축방향 4차 중심차분이다. `phi=phi_G4+delta_phi` 또는 위 orbit 직접변수 방식 어느 쪽이든 **G4가 정확히 표현됨을 테스트**해야 한다. `phi_old=[1,.57063894,.30414468,.33693168,.09205035,.08973684]`는 원본 Colab **5×5의 이식 초기값**이며 동일한 업데이트/교환/초기화가 아니므로 'old full algorithm'이라고 부르면 안 된다. 2D pilot에서 찾은 `phi_pilot=[.997721901,.262736838,.618148076,-.053713896,.265501073,-.378951811]`도 선택적으로 초기값에 포함할 수 있으나 **이전 test를 학습에 포함하지 않는다**.

**비정규화**: `Σ|K|`, `Σ K_J`, `Σ xK_x` 등을 상수로 강제하지 않는다. 미분 0차 모멘트는 antisymmetry로 자동 0. 생성 미분 gain이 달라지면 이완 동역학의 상대세기가 변하므로 `m1=Σ dx K_x`와 고주파 응답을 보고하되 결과를 임의로 보정하거나 나쁜 candidate를 topology/shape 기준으로 걸러내지 않는다. `m1≈0`으로 붕괴하는 해는 `q` 계산의 수치적 퇴화로 **수치 실패**라고 기록하고 매끄러운 미분관측량이 유한하고 양수인 경우만 평가하라.

### 3.2 7×7 및 3D

5×5의 q/회전/방향 generalization 개선이 검증된 후에만 7×7 orbit로 확장한다. **이전 단계 계수를 고정할 필요는 없다.** 새 orbit를 0에서 시작하면 5×5와 G4를 정확히 포함하므로 최소 달성 가능한 train loss가 불리해지지 않는다.

3D는 5×5×5에서 하나의 미분축에 홀수, 두 횡축에 짝수, 횡축 맞교환 대칭을 갖는 orbit를 구성한다. `|dy|,|dz| ∈ {0,1,2}`의 unordered pair 6개 × 미분축 `|dx|=1,2` = **12 독립 파라미터**가 가능한 출발점이다. 다른 축은 축순열로 만든다. `G4`를 전체 3D 커널에 정확히 삽입하고, axis permutation/reflection covariance 및 signed derivative moment를 시험한다. 2D 파라미터를 그대로 3D의 최적해라고 간주하지 말라.

### 3.3 교환 커널 추가 ablation (2차)

생성 미분 최적화가 완료된 후 원본처럼 `K_J=Σ_i G_i ⊙ G_i` (Hadamard square)의 **raw, non-normalized** 교환과 independent free exchange를 별도 시험한다. 기본 주 실험인 NN exchange를 대체하지 말라. `K→α K`이면 tied `K_J→α² K_J`이므로 원래 노트북에서 D/h/J 실효 비율이 함께 바뀐다는 점을 보고한다. 정규화 버전은 부차적 통제군일 뿐 필수 제약은 아니다.

## 4. 학습 목적: 표준자 없는 비례성 (single-circle calibrated MARE 금지)

주 단위는 **형상 원형(prototype)**이다. 동일 원형의 여러 rotation/shift/noise는 서로 독립 샘플로 세지 않는다. 각 원형에 대해 `q_{u,r,s}=I_{u,r,s}/P_u`(또는 3D `I/A`)를 얻는다. 깨끗한 영상 기준 단위별 평균 `\bar q_u`, log-ratio `\ell_u=\log(\bar q_u)`를 사용한다.

\[
\mathcal L_{\rm prop}=\operatorname{Mean}_u(\ell_u-\overline{\ell})^2,\quad
\mathcal L_{\rm rot}=\operatorname{Mean}_u\left[\operatorname{Var}_{r,s}(I_{u,r,s})/\operatorname{Mean}_{r,s}(I_{u,r,s})^2\right],
\]
\[
\mathcal L_{\rm angle}=\operatorname{Mean}_{u\text{ oriented}}\operatorname{Mean}_{r,s}\left[\left(\frac{\angle(\hat e,e)}{\theta_*}\right)^2\right].
\]

\[
\boxed{\mathcal L=\mathcal L_{\rm prop}+\lambda_{\rm rot}\mathcal L_{\rm rot}+\lambda_\theta\mathcal L_{\rm angle}+\lambda_{\rm noise}\mathcal L_{\rm noise}}.
\]

`θ*`는 사전 정의한 각도 스케일(예: 1° radian)이며 weight를 검증 전에 고정하라. 부호·규약의 오류로 큰 angular loss를 숨기지 말고 방향 정의가 없는 원형은 제외한다. `L_noise`는 해당 원형의 **clean 대비 I의 상대변화** 또는 `log q_noisy-log q_clean`을 벌점으로 주는 항이다(잡음 0%,0.2%,0.5%,1%; clean과 paired). `noise`는 사용자가 지정한 salt-and-pepper 모델의 입력 오염이고, 원형 연속 GT는 변하지 않는다. Noise weight 값은 여러 **사전 선언된** 설정을 train/validation에서 비교하여 Pareto report. Clean-only 학습도 반드시 대조군으로 유지.

- **주 손실에 reference-circle calibration `c`를 넣지 않는다.** `q`의 CV, log-variance, pairwise spread는 기준 원 선택과 무관하다.
- 도함수 `I`가 양수·유한인 모델만 허용한다. `q≈0`, NaN/Inf, degenerate eigenaxis 등 수치 특이점에 적절한 안전장치를 적용하고 기록한다. 단 0이 아닌 전체 gain이 서로 달라도 동등하게 취급하라.
- 관측량 `I`와 기하학적 GT의 관계를 검증하는 연구이므로 `P,A` 정의는 필요하다. synthetic continuous GT와 일반 마스크의 고유 '참둘레'를 혼동하지 않는다.
- `L_prop`는 학습/검증에서 baseline G4가 후보에 항상 남아 있도록 하되, **신규 test 향상은 보장되지 않는다.** validation Pareto 선정과 test lock 준수.

## 5. 독립 데이터 구성과 calibration sweep

**절대 조건:** 과거 `generator_first/output/test*`, `joint_operator` 시험, 첨부 2D 파일럿의 `test_specs` 및 `RADIUS_SENSITIVITY_HISTORICAL.csv`에 등장하는 test 원형/seed/회전 샘플은 새 독립 test가 아니다. 과거에 본 radius·shape 조합을 정확히 새 test로 복사하거나 기존 시험에서 튜닝하지 말라. 과거 수치는 **탐색적 비교와 source regression** 용도로만.

2D: 다양한 반지름 원(최소 5개), 곡률이 변화하는 타원·stadium·rounded rectangle, ring, 두 물체, 비대칭/peanut을 train에 분산. 최소 각 family의 여러 크기/종횡비를 포함. Train/validation/test는 radius, aspect ratio, gap, curvature, shift seed **prototype 단위**로 분리. 새 test family/size는 가능하면 train/validation과 구분; 각 군의 개수 및 coverage를 정량 기재.

3D: sphere, ellipsoid, capsule, torus, separated pair, union/peanut. 3D 회전은 SO(3)의 다양한 tilt를 포함하고 단순 xy-plane 회전만으로 '3D 회전검증'이라고 부르지 말라. 3D GT는 sphere/capsule/torus 등 정확식, 일반 triaxial ellipsoid는 수렴된 quadrature를 사용. 중첩형 union/peanut은 실제 경계 노출 영역을 정확히 계산하거나 충분히 수렴한 독립 연속 기준값을 사용하고 '근사'를 정확식으로 오기하지 않는다.

**외삽을 보고할 변수:** 작은 반지름(큰 곡률), 관측장 wall width/물체 크기 비, 상이한 aspect ratio, rotate, subpixel shift, salt noise, physical spacing `a`, padding. 해상도 재표본화와 단순 도메인 padding을 구별한다. `a` 변경 시 EDT width(격자 단위)를 고정할지 물리적 벽 폭을 유지할지 별도 두 프로토콜로 구분하고 calibration `c(a)` 전이와 `c` 고정 전이를 모두 측정한다.

**Calibration/reference-ruler sweep** (주 손실과 완전히 분리):
1. `I[f]`를 고정하고, train-only reference로 정의한 `radii` 여러 개와 서로 다른 shape family(원/타원/stadium 등)를 각각 표준자로 사용한다. 동일 테스트 집합에 대해 `c_r=L_r/I_r`를 적용, `MARE_r`, bias, p95/worst, `c_r` CV와 range를 평가한다. 3D는 구의 다양한 R과 ellipsoid/capsule 표준자를 사용.
2. **새 test의 GT를 calibration에 절대 사용하지 않는다.** reference sets는 학습용으로 생성한 별도 단위여야 한다. 테스트에서 최저 MARE를 주는 표준자를 골라 '최종 성능'이라고 하지 않는다.
3. 비교 지표는 `calibration_free_q_dispersion`, `calibration_free_log_q_std`, `rotation_CV`, `ruler_sensitivity(c, MARE)`, `MARE_median_over_refs`, `MARE_worst_ref`, `noise`별 성능, 방향 MAE; 본문에 **고정 ref r=10.3에만 해당하는 수치**는 보조로만 표기한다.
4. 다수 참조에 대한 train-only global calibration(weighted least squares)은 실용적인 대안으로 추가 보고 가능하지만 '원형 선택 불변'이라는 근본 속성을 대체하지 못한다.

## 6. 최적화 및 비교 실험

**최소 우선순위 2D**: 
- 동일 NN·observer·N에서 (a) fixed G4, (b) 원본 노트북 5×5 이식, (c) 지난 파일럿 계수 이식, (d) **새 L-BFGS-B 학습형 5×5** 비교. (b)/(c)는 역사적 계수일 뿐 새 test에서 선택 후보는 train/validation만으로 결정한다.
- Multi-start: 최소 G4, original-5×5, pilot-5×5 그리고 적어도 하나의 작은 random perturbation 초기값. 수치 미분이 아니라 **자동미분/정확한 Jacobian**을 구현하고, L-BFGS-B의 시작·종료 목적함수, 반복 횟수, gradient norm, active bounds, stopping reason을 저장한다. 원본 5×5 단계 115 evaluations에서 착안하되 **최대 반복 16회만으로 최적화 충분이라고 주장하지 않는다**. 수렴 기준 도달 또는 예산 한계 명확히 보고. `maxiter`와 평가횟수는 실제 장비에서 현실적인 범위로 사전 확정 후 기록.
- G4는 최적화 공간에 **정확히 포함되는 anchor**. 각 train/validation selection은 동일 objective+같은 observation으로 비교. G4보다 손실이 나쁜 학습 결과를 선택하지 말되, test에서 원치 않는 결과가 나와도 test를 보고 모델을 변경하지 말라.
- 5×5 분리형 대조(고정 linear moment G4 포함), 7×7 확대(개선 확인 후), optional tied/free exchange ablation. 3D는 2D 주요 결과가 완료된 뒤 resource-bound run; 완료 범위 표기를 엄격하게.
- 길이/표면적 geometry head 및 angle head를 나중에 별도 재학습할 수 있지만 (i) frozen observer generator only, (ii) retuned observer를 반드시 분리. `N={5,10,20,30}`은 validation에서 선택하되 초기 주 비교는 `N=10` 고정.

**성공 조건은 positive result가 아니라 연구 질문에 재현성 있게 답하는 것**. 단순히 validation G4와 동점이거나 실패해도 `G4`를 유지하고, 최적화의 한계를 보고한다.

## 7. 실제 수치 검사 및 통계

- 커널: antisymmetry, transverse even symmetry, G4 exact embedding, derivative on constant zero, gradient of linear ramp (boundary 제외), axis permutation/reflection covariance, PyTorch/scipy convolution vs correlation, 2D/3D sign/transpose mapping.
- 동역학: `f` 원본 불변, 모든 유효 위치에서 `||n||≈1`, zero-H 대응, finite gradients, differentiation through N steps, branch definition, equal baseline rollout.
- 목적함수: 글로벌 `I` multiplicative rescale에도 `L_prop` 불변(positive scaling), 모든 unit의 `q`가 같은 상수면 `L_prop≈0`, 한 radius를 다른 radius로 바꿔도 q/primary loss 불변, **reference sweep에서는 MARE가 바뀔 수 있음을 확인**.
- split: prototype-level non-overlap, training-only reference, no test retuning. 출처별 SHA256, 고정 seed, env/python/thread settings, optimizer state, frozen selected coefficients.
- 결과: shape family / radius curvature / aspect ratio / clean vs noise / rotation vs shift / N / spacing별 `mean`, `median`, `p90`, `p95`, `max`, signed bias, bootstrap CI(원형 단위). 각도는 oriented prototype에만 MAE/RMSE/p95와 수. test 수치가 train과 다른 이유와 각종 failure까지 공개.
- Fourier: `Kx(k)`의 low-frequency first moment, high-k gain/anisotropy 및 noise-amplification 가능성 표시. **미분 모멘트 정규화/고주파 페널티를 의무화하지 말 것**. 필요하면 별도 ablation으로만 시험.
- timing: training wall/CPU, 2D/3D peak RAM, process runtime, 1회 추론 field vs observer, G4 vs trained cost. full convergence certificate가 아니라 `success`, `message`, `nfev`, `nit`, gradient norm 구분.
- **금지:** reconstruction `Dice/IoU`, connected components, Euler, topological degree, wall width, residual convergence thresholds를 geometry selection/gating에 사용하지 않는다. 위상현상은 PBGIA의 핵심적인 잠재 응용임을 보고서에서 언급하되 이번 optimization constraint는 아님.

## 8. 요구 실행 명령·결과물

새 디렉터리 `reference_invariant_generator/`에서 다음처럼 실행 가능하게 구현하라(실제 명령 이름 변경 시 README에 명확히 설명):

```bash
python run_all.py --smoke --tests
python run_all.py --train2d
python run_all.py --validate2d
python run_all.py --freeze2d
python run_all.py --test2d
python run_all.py --report2d
python run_all.py --train3d
python run_all.py --validate3d --freeze3d
python run_all.py --test3d --report3d
```

테스트는 pretest frozen hash 검증 및 중복/재시험 금지를 포함한다. 이미 test가 존재하면 train/validation을 덮어써서 시험 선택을 재수행하지 않도록 한다. 커맨드가 실행 중 예산초과 시 진행 상황/중간 checkpoint/raw logs/남은 블로커를 기록하고 가능한 범위의 **완성된 2D 결과**를 먼저 제출하라.

필수 파일:

- `README.md`, `requirements.txt` 또는 환경 lock, 독립 `code/`, `tests/`, 원본 provenance, `output/RESEARCH_REPORT_KO.md`, 보고서 실행 예시.
- `manifest_{2d,3d}.csv` (role/train/validation/test/reference 및 prototype 그룹, 각도/shift/noise/GT 수식 출처), `protocol.json`, `train_trials.csv`, `train_optimizer_trace.csv`, `validation_summary.csv`, `validation_pareto.csv`.
- `model_freeze_{2d,3d}.json` + hash, `test_raw_{2d,3d}.csv`, `test_by_family.csv`, `test_by_noise.csv`, `test_by_curvature.csv`, `reference_ruler_sweep.csv`, `pairwise_q_ratios.csv`, `calibration_free_summary.csv`, `rotation_shift_summary.csv`, `timings.csv`.
- `ablation_fixed_observer.csv`, `ablation_retrained_observer.csv` (해당 단계가 실행된 경우), `models.json`, `environment.json`, `source_sha256.json`, `run_logs/`, 그림(수렴 곡선, q vs curvature, ref radius vs c, ref family vs MARE, noise, 2D/3D 결과).
- GitHub PR 생성, 변경 파일/결과 재현 명령과 미완료 작업을 명시. **merge하지 말 것**.

## 9. 연구 보고서에서 반드시 답할 항목

1. 표준자를 특정하지 않은 주 지표 `CV(q)` 및 `log q` dispersion가 G4에 비해 개선됐는가? 둘레를 몇 %의 오차로 맞춘다는 표현은 **어느 calibration 기준인지** 설명하는가?
2. 다양한 크기의 원·타원·stadium 표준자로 보정했을 때 `c`와 MARE의 범위 및 모델 순위가 어떻게 변하는가? 작은 곡률 반경은 왜 어려운가(추측과 증거 구분)?
3. 회전 불변성, 아화소 이동, 방향 MAE, salt noise와 비례성 사이 trade-off는 어떤가? 1% noise 악화가 줄었는가?
4. 비정규화 커널의 gain과 tied exchange 실효 coupling 효과를 분리했는가? 선형 모멘트/hf/교환 스케일을 단지 arbitrary renormalization으로 숨기지 않았는가?
5. 생성 커널만 바뀐 fixed-observer 실험에서 개선됐는가? observer 재학습 효과와 혼동하지 않았는가?
6. 3D는 실제 완료했는가? 3D의 독립 표면적 검증, 원/구 curvature 및 표준 구 sweep 결과를 제공했는가? 완료 못했으면 정직하게 단계 보고했는가?

**부정적 결과 또는 미완료 결과도 연구 결과다.** 옛 test를 다시 사용하거나 test 성능을 보고 재학습하여 개선을 주장하는 일은 절대 금지한다.
