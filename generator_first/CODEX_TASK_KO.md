# PBGIA — 자화장 생성 연산자 우선 최적화 (2D/3D) : Codex 연구·구현 의뢰서

**상태:** 신규 실험 요청서 (2026-11-16)  
**저장소:** https://github.com/cywon-commits/Physics-Based-Geometric-Image-Analysis  
**참조 브랜치:** `codex/joint-operator-2d3d-20261116`  
**핵심 방향:** 측정용 커널의 사후 보정보다 **좋은 자화장 생성**을 최우선으로 연구한다. 특정 과거 커널 계수를 복사하지 않는다.

## 0. Codex가 먼저 해야 할 일

1. 위 저장소의 현행 코드와 보고서를 직접 읽고, **실제로 실행 가능한 구조와 정의**를 먼저 확인한다.
   - `fixed_iteration/code/fixed_iteration_checkpoints.py` (`initialize`, `gradient`, `one_step`)
   - `joint_operator/code/joint_study_core.py` (`generated_gradient`, `exchange`, `step`, `observe`)
   - `joint_operator/code/train_joint.py` (`learn_observers`, `candidate_score`, `sample_candidates`, `train`)
   - `joint_operator/code/design.py` 및 `joint_operator/output/RESEARCH_REPORT_KO.md`
   - `joint_operator/reference/texturefiltertrain5 (1).py`, `joint_operator/reference/lengthorientationapplication (1).py` (실제 존재 경로 확인)
   - 기존 PBGIA-30의 frozen configuration, calibration, 원자료 및 split provenance.
2. 재현성 테스트와 짧은 smoke run을 먼저 실행한 후, 실제 신규 실험은 **새 브랜치 및 `generator_first/` 디렉터리**에 독립적으로 만든다. 기존 보고서/결과는 변경하거나 덮어쓰지 않는다.
3. 이 의뢰서의 **연구 목적·평가 기준 수정**을 코드에 적용한다. 구 버전의 shape/topology acceptance gate나 기존 test 데이터를 재사용해 새 모델을 선택하면 안 된다.

## 1. 연구 원칙 — 무엇을 만들고 무엇을 강제하지 않는가

PBGIA는 입력 이진 마스크 `f(r) ∈ {-1,+1}`에서 보조 정규화 자화장 `n(r)=(v(r),s(r)) ∈ S^d` (`d=2,3`)을 생성하고, **둘레(2D), 표면적(3D), 주축 방향**을 추정한다. **위상 현상·위상결함 분석은 이 공통 자화장의 핵심적인 향후 활용**이지만, 이 연구에서는 위상 보존을 목적으로 커널을 최적화하지 않는다. 위상 분야가 부차적이라는 뜻이 아니다.

- 입력 마스크 `f`는 매 반복마다 고정하고, 출력 정규화와 유한성 등 계산의 기본 정합성은 지킨다.
- `sign(s_N)==f`, Dice, IoU, Euler characteristic, 연결 성분, degree, 위상 certificate, 자구벽 너비, 잔차 임계값을 **손실항·성능 점수·후보 탈락·배포 gate에 넣지 않는다**.
- **Dice를 계산하거나 필수 산출물로 요구하지 않는다.** 이전 구현에서 Dice를 얻게 되는 경우에도 이번 선택 로직에는 절대 사용하지 말 것.
- 위상수는 필요 시 별도 물리 연구용 참고 데이터로 계산할 수 있지만 **학습/선택과 완전히 분리**한다. 이를 필수 실험으로 삼지 않는다.
- 부호장 `s_N`이 원본 마스크를 일부 수축/팽창시키더라도, 원본 `f`가 보존되고 기하학적 추정이 정확하다면 실패가 아니다.
- 회전·변위·노이즈에 따른 **원본 형상의 측정 오차**가 핵심이다. 위상/형상 재구성은 평가 목표가 아니다.

## 2. 물리·수치 모델: 생성용 연산자를 학습하라

현재 기준 모델(고정된 생성 미분=중심차분, 교환=nearest-neighbor axial sum):

\[
\mathbf H_v^{(t)}=J\sum_{\delta\in NN}\mathbf v^{(t)}(\mathbf r+\delta)+D\,\nabla_c s^{(t)}(\mathbf r),
\quad H_s^{(t)}=J\sum_{\delta\in NN}s^{(t)}(\mathbf r+\delta)+h f(\mathbf r),
\quad \mathbf n^{(t+1)}=\frac{\mathbf H^{(t)}}{\|\mathbf H^{(t)}\|}.
\]

여기서 `∇c`의 각 방향 성분은 `0.5*(roll(s,-1)-roll(s,+1))`, boundary는 periodic이며, 미분하는 대상은 **매 단계 업데이트된 `s(t)`**이다. 기존 코드의 `J=1,h=2,D=1`, EDT 기반 initialization을 baseline과 기본 비교 조건으로 고정하고, zero-H일 때 이전 state를 유지한다. 표준 모델에 없는 `div(v)`, `grad(f)`를 임의로 넣거나 exchange를 Laplacian이라 바꿔 설명하지 않는다.

새 생성 모델:

\[
\boxed{\mathbf H_v^{(t)}=J\,[K_J(\eta)*\mathbf v^{(t)}]+D\,[G_{\rm gen}(\phi)*s^{(t)}],\quad H_s^{(t)}=J\,[K_J(\eta)*s^{(t)}]+hf,\quad \mathbf n^{(t+1)}=\operatorname{normalize}(\mathbf H^{(t)}) .}
\]

`G_gen`과 `K_J`를 **생성 커널**로 간주하고, 뒤의 `K_geom`, `K_orient`는 **관측 커널**로 분리한다. 각 커널이 어디에 적용되는지 명백하게 기록한다.

### 2.1 생성 미분 커널 계열

우선 2D/3D 공통의 separable 5-point family:

\[
d(b)=[-b,-(1/2-2b),0,(1/2-2b),b],\qquad
q(t,u)=[u,t,1-2t-2u,t,u].
\]

2D `G_x=d⊗q`, `G_y=q⊗d`; 3D `G_x=d⊗q⊗q` 등. 다음을 필수 비교한다.

- G0: 중심차분 `(b,t,u)=(0,0,0)` (frozen generator baseline)
- G1: 4차 중심차분 `b=-1/12, t=u=0` (고정 대조군)
- G2: **학습형 `b,t,u`** (미분의 상수 응답 0, 선형 응답 1, 좌표축 순열 대칭 유지)
- G3: G2로 개선이 확인되면 비분리형 소수 orbit 또는 최소 7-point extension을 별도 후보로 추가. 추가 자유도와 비용을 명확히 보고하고, 자유도 증가 자체를 우수성으로 주장하지 않는다.

데이터 스케일은 `spacing=a`로 명시하며, 임의 `a`에서 미분의 선형 응답이 `1/a`가 되게 구현한다. 각 커널에 대해 Taylor expansion / Fourier symbol과 저파수 등방성 오차를 계산한다. **최적의 이산 미분 커널이 반드시 4차 정확도를 가져야 하는 것은 아니다.** 학습 목표는 생성된 장의 실제 기하측정 성능이다.

### 2.2 교환 커널 계열

- E0: 기존 2d axial NN sum. 가장 중요한 통제군.
- E1: 원래 Python의 `K_J(r)=Σ_i G_i(r)^2` **원소별 제곱(Hadamard)** 결합을, 2D/3D 모두 새 `G_gen`으로 생성. `-G†G`와 혼동하지 않는다. 총합 정규화 버전과 원래 raw 버전을 분리한다.
- E2: 양의 축·대각선·2차 근접 shell의 독립 orbit weights를 학습하는 exchange; 가능하면 shell별 weight와 self-coupling을 분해하여 비교한다. 2D/3D 각 차원에 맞는 점군 대칭을 보존한다.
- E3(선택): 합이 고정된 범위에서 추가 shell을 허용하되, 정확도와 비용이 정당화될 때만 확장한다.

**중요한 스케일 통제:** `Σ K_J=2d` 정규화 계열을 기본 비교로 하여 `J/h/D`의 상대 세기가 은밀히 달라지지 않게 한다. `raw tied`는 별도 실험이며, 성능 이득이 정규화 때문인지 상대 세기 변화 때문인지 `h,D` 재스케일 대조를 통해 분리한다. 복수 generator family에 동일한 search budget 및 wall-time budget 민감도 분석을 제공한다.

### 2.3 시작 상태와 반복횟수

- 기본 initialization: pinned PBGIA-30과 동일한 **이진 EDT에서 계산한 `s0`**와 정규화한 `v0` (analytic GT distance 유출 금지).
- 대조군: 원래 Colab의 binary `v=0,s=f` 초기화, 그 외 initialization은 분리된 ablation으로만.
- 고정 `N ∈ {5,10,20,30,40}`. 2D와 3D는 서로 다른 `N`을 선택할 수 있다. 필요할 경우 1,2,3회도 **추가 탐색 그룹**으로 확장해 초기 과도장 특성을 조사한다.
- 매 N에서 생성 미분과 exchange의 최적값이 같다고 가정하지 않는다. 계산 예산상 공유할 경우에도 **N별 generator 재학습 대조군**을 최소한 5/10/30에서 실행하고 공유 선택의 한계를 밝힌다.
- `N`은 학습 후 결과를 보고 잘라내는 adaptive stopping이 아니라 배포 시 고정 호출 횟수다.

## 3. 관측량은 공정하게 재학습하되, 연구 초점은 생성장이다

기하량:
\[
T_{ij}^{(g)}=\sum_{\mathbf r,a}[D_i^{(g)}n_a(\mathbf r)] [D_j^{(g)}n_a(\mathbf r)],\quad
\widehat P\;\text{or}\;\widehat S=c_g\operatorname{tr}T^{(g)}.
\]

주축 방향은 **독립** `K_orient` 또는 `Qbond = sym Σ(H_J⊗v) - I tr/d`, `Qdirect = Σ(v⊗v) - I tr/d`, `T_grad`의 고유축/혼합을 검토한다. 2D에서 half-angle/부호/90° convention 및 3D에서 어느 고유축이 장축인지 반드시 검증한다. `orientation`이 정의되지 않는 원·구·정대칭 물체에는 각도 GT를 부여하지 않는다.

측정량을 따로 학습한다:

- `K_geom`과 배율 `c_g`는 **둘레·표면적 손실만** 기준으로 학습한다.
- `K_orient`, orientation tensor 조합과 고유축 convention은 **각도 손실만** 기준으로 학습한다.
- `K_geom == K_orient` 공유 모델은 추가 단순화 대조군이다. 이전 코드는 하나의 `p`가 둘 다에 영향을 주었다.
- 반드시 **두 가지 통제 평가**를 별도 보고한다: (i) 관측 커널을 고정하고 generator만 교체; (ii) generator마다 같은 예산으로 관측 커널을 재학습. (ii)만 보고하면 개선이 생성장 때문인지 후처리 때문인지 해석할 수 없다.
- 측정 보정계수는 train-only reference 또는 train-only regression으로 결정하며 모든 후보와 conventional baseline에 공정한 보정 정책을 제공한다. test GT를 calibration에 사용하지 않는다.

## 4. 1차 연구 질문과 실험 행렬

핵심 질문: **동일한 관측 연산자로 읽었을 때도 학습형 생성 미분이 더 좋은 자화장을 만드는가? 교환 커널을 함께 학습하면 추가로 개선되는가?**

| 실험 ID | 생성 미분 | 교환 연산자 | 관측 연산자 | 의도 |
|---|---|---|---|---|
| A | 고정 중심차분 | axial NN | PBGIA-30 frozen | 완전 고정 기준 |
| B | 고정 중심차분 | axial NN | 별도 재학습 | 관측 최적화만의 효과 |
| C | **학습형 생성 미분** | axial NN | B와 동일하게 고정한 관측 | **생성 미분 효과만** |
| D | **학습형 생성 미분** | axial NN | 독립 재학습 | 생성+관측 실제 성능 |
| E | **학습형 생성 미분** | tied Hadamard | C와 동일 관측 / 독립 재학습 병렬 | 기존 생성 설계 계승의 효과 |
| F | **학습형 생성 미분** | learned positive free | C와 동일 관측 / 독립 재학습 병렬 | 교환 커널 학습의 추가 효과 |
| G | fixed fourth-order | axial NN | 조건 일치 | 고차 유한차분의 효용 |
| H | legacy Colab entire algorithm | 원래 방식 | 원래 방식 | 역사적 대조; boundary/h/D/init 차이 명시 |

A/B/C처럼 **관측이 동일해야 하는 비교**는 calibration의 동결 또는 train-only 동일 재보정 2가지 모두 보고한다. B의 관측을 C에 적용할 때 성능이 낮더라도 해당 결과를 보고하고, D의 결과와 분리한다. 2D와 3D 모두 동일 프로토콜; 계수는 독립 학습.

단계별 우선순위: (1) G only vs baseline, (2) E1/E2 with G fixed/learned, (3) observer independent retuning, (4) wider kernel/orbit/initialization sensitivity. 자원이 부족하면 먼저 (1)을 제대로 완수하고 진행도를 보고한다.

## 5. 손실·검증·모델 선택 — Dice 없음

원형 geometry unit을 동일 가중하는 목적함수의 출발점:

\[
\mathcal L_g=\mathbb E_{\text{unit}}\!\left[\rho\!\left(\frac{\widehat G-G_{\rm GT}}{G_{\rm GT}}\right)\right],\qquad
\mathcal L_o=\mathbb E_{\text{oriented unit}}\!\left[\rho_\theta\!\left(\frac{180}{\pi}\arccos|\hat e\cdot e_{\rm GT}|\right)\right].
\]

단위가 서로 다른 loss를 무차원 기준으로 정규화하고, 사전 지정한 가중치 복수 및 Pareto front로 비교한다. 평가/보고는 최소 다음 항목을 제공한다.

- 2D 둘레, 3D 표면적: **MARE, median, RMSE, p90, p95, worst unit**, signed bias. 2D/3D를 섞어 하나의 평균으로 만들지 않는다.
- 방향: **MAE, RMSE, p90, 최대 오차**, 3D는 공간 회전을 다양하게 포함; 방향이 없는 물체는 분모에서 제외하고 그 개수 기록.
- 회전/아화소 이동: 동일 원형의 분산/CV와 GT 대비 오차를 구분; 특히 lattice-angle anisotropy.
- 노이즈: clean / 0.2% / 1% 이상. 노이즈 투입 전의 원형 GT 기준으로 평가하고, 작은 구멍이나 salt noise가 실제 신호인지 선험적으로 단정하지 않는다.
- 효율: cold/warm latency, field generation vs observation 분리, 평균/중위 실행시간, 메모리와 유한-budget training cost.
- 기본 수치 정합성: `f` 불변, `||n||≈1`, NaN/Inf 없음, exchange scale, derivative moments, 회전/축순열 공변성, zero-H 처리. 이들은 geometry 학습 손실이 아닌 **코드 정확성** 조건이다.
- 생성장 품질의 선택적 설명 지표: 자구벽 수직정렬, 각도별 벽 프로파일, Fourier symbol anisotropy 등. **측정 성능이 나쁜데 '물리적으로 좋아 보인다'는 이유로 모델을 채택하지 않는다.**

**절대 금지:** `Dice / IoU / Euler / component count / topological degree / wall width`로 후보를 탈락시키거나 penalty를 주거나 production fallback을 결정하는 것. 이들은 모델 선택과 무관하다.

모델 선택은 **train/validation만**으로 실행하며, 정확도·방향·계산량의 Pareto frontier에서 사전 선언한 규칙으로 frozen candidate를 고른다. `N`과 kernel convention까지 frozen hash로 잠근 뒤 새로운 independent test를 한 번만 실행한다. Test에서 성능 저하가 나오면 그 결과를 보고하되, test를 보고 재조정하지 않는다. 기존 test에선 새 모델 선택 금지.

## 6. 데이터 설계 및 연속체 해석의 정확한 위치

**가장 큰 오차원 중 하나가 픽셀화라는 가설**을 실험적으로 분해해야 한다. 연속적 형상이라는 사실만으로 자화장 해 또는 모든 기하적 적분이 자동으로 해석적으로 구해진다고 주장하지 않는다.

- Train / validation / **새 test**는 형상 원형·반지름·종횡비·경계 곡률 조합 단위로 나누고, 같은 원형의 회전/shift/noise 샘플이 다른 split으로 새지 않도록 한다.
- 2D: 원·타원·stadium·ring·비대칭·복합 형상, 넓은 크기 범위. 3D: 구·타원체·capsule·torus·곡률 혼합/복합 형상, 작은 구 포함. 각 형상의 GT는 가능하면 analytic 또는 독립적인 수렴된 고정밀 연속체 수치 기준을 사용한다. 잘못된 근사식을 exact GT라 표시하지 말 것.
- 독립 test는 기존 fixed_iteration 및 joint_operator 시험과 겹치지 않는 원형, 종횡비, 크기, 회전, subpixel shift, noise seed를 사용한다. 이전 test 수치는 참고하고 데이터는 새 학습에 사용하지 않는다.
- 픽셀화 오차 분리: (a) analytic continuous GT, (b) digitized input의 conventional estimator, (c) PBGIA estimator를 비교; 가능한 경우 antialiased sample 또는 더 높은 해상도의 binary rasterized shape도 제공하되 모델 입력 규칙은 동일하게 유지한다.
- 해상도 실험은 **물리적인 크기/간격**을 명시하고, 같은 물체를 `a` 감소와 함께 다시 표본화하여 grid-convergence 확인. 단순 48³→80³ domain padding은 refinement가 아님.
- periodic boundary의 image interaction을 막도록 충분한 margin을 확보하고 필요 시 boundary ablation을 실시한다.
- angle split: 2D 0–π 전체 및 3D azimuth·polar tilt·축 교환·반사. 3D true SO(3) 회전에 대한 일반화를 확인하되 finite sampling을 완전 불변성이라고 표현하지 않는다.

## 7. 최적화 방법·계산 예산

과거 joint study는 generator별 적은 proxy 표본 및 큰 shape penalty가 있어 generator 학습 효과를 공정하게 보이기 어려웠다. 이를 교정한다.

1. 작은 smoke로 끝까지 pipeline을 검증하고, **generator 중심의 예산**(다중 초기값, 더 충분한 clean/noisy/rotation proxy, 2D/3D 별도)으로 확장한다.
2. 후보별 동일 rollout/sample count를 기본으로 하고, 비슷한 CPU walltime에서의 sensitivity도 제시한다. 고차원 family에 동일 evaluations만 할당하는 편향을 기록한다.
3. 경사 기반 학습이 가능하면 2D differentiable rollout (예: PyTorch/JAX)을 구현해 DE/CMA-ES와 비교할 수 있다. 하지만 TensorFlow/GPU를 필수 의존성으로 새로 강제하지 말고, 기존 SciPy 기반 CPU 경로도 유지한다. `normalize`와 update의 도함수/비평활점 처리는 문서화한다.
4. 최적화 실패, optimizer success=false, bounds hitting, 좁은 parameter space, 해가 지닌 trade-off 등을 숨기지 않는다.
5. GPU가 있든 없든 재현 가능하도록 `--smoke`, `--train`, `--validate`, `--test`, `--report` 명령을 분리하고, 정해진 budget 밖을 자동 실행하지 않도록 한다.

## 8. 요구 산출물과 코드 변경 지시

새 디렉터리 `generator_first/` 안에 다음을 마련한다.

- `README.md`: 목적·실행환경·짧은 실행 예제·전체 실행 명령, baseline 기준 및 비용.
- `code/` 독립 훈련 모듈: generator/observer/metric/split 명료하게 분리. 기존 baseline 재사용 시 provenance를 남기고 옛 결과 파일은 read-only.
- `tests/`: 차분 moment, 2D/3D 축순열/reflection, normalization, f immutability, baseline exact reproduction, independent split non-overlap, frozen selection hash, fair calibration, observer-fixed ablation 재현. Dice 관련 acceptance test 없음.
- `output/`: `protocol.json`, `manifest.csv`, `generator_trials.csv`, `observer_trials.csv`, `ablation_fixed_observer.csv`, `ablation_retrained_observer.csv`, `train_summary.csv`, `validation_summary.csv`, `pretest_selection.json` + sha256, `test_results.csv`, `test_summary.csv`, `test_family_summary.csv`, `paired_bootstrap.csv`, `timings.csv`, `environment.json`, `run_logs`, 회전/해상도/노이즈 figure.
- `output/RESEARCH_REPORT_KO.md`: **(i) 실제로 생성 미분 학습이 우수했는가? (ii) 교환 커널까지 학습했을 때 추가 개선이 있었는가? (iii) 2D/3D에서 최적 N과 학습계수는 무엇인가? (iv) 측정커널 재학습의 기여는 어느 정도인가? (v) 전통적인 Crofton/PCA/MC 대비 어떤 이익이 있는가?** 를 수치로 명확히 답할 것. 부정적 결과도 모두 제시.
- 실험용 추론 CLI와 버전/선택된 model metadata; 새 candidate가 baseline보다 나쁠 경우 이를 솔직히 표기하고 자동으로 '개선'이라 부르지 않는다.

**보고서 작성 규칙:** train/validation/test 성능을 명확히 구분, 동일 test에서의 비교만 표기, 전체 단위당 표본 수/CI, failure mode(기하오차·각도오차·수치 안정성만), 하위집단 성능을 제공한다. 기술 논문에 쓸 수 있는 최적화 목적·편향과 한계를 명시한다. 결정/실패 내용은 한국어로 설명한다.

## 9. GitHub 작업 절차와 완료 조건

1. `generator_first/` 작업용 독립 branch를 만들고, 기존 PR/결과를 훼손하지 말 것.
2. 예비 결과를 얻은 뒤 의미 있는 새 실험을 계산한다. 시간이 부족하면 smoke만 계산하고 성공한 연구인 양 보고하지 말고, 다음 실험을 수행할 커맨드와 blocker를 기록할 것.
3. PR을 작성하되 **자동 병합하지 말 것**. 테스트 및 재현성 검증 결과를 PR에 포함한다.
4. '성공'은 신규 generator가 반드시 기존보다 이기는 것이 아니라 **관측 후처리 효과에서 분리된 생성용 커널 최적화의 효과를 정량적으로 규명하는 것**이다.
5. 사용자가 계속 연구할 수 있게 config·학습/선택 기록·새 독립 데이터 분할과 checkpoint 관리 방식을 남긴다.

### Codex의 첫 보고에서 확인할 체크리스트

- [ ] 기존 generator가 중심차분 `G0`인지 실제 함수로 확인
- [ ] 기존 Dice/위상 gate를 새 훈련 및 선택 경로에서 완전히 제거
- [ ] 생성 미분만 학습하는 C 모델의 **동일 observer** ablation 실행
- [ ] E1·E2 교환 학습과 `ΣKJ` 통제 실행
- [ ] 둘레/표면적과 방향의 관측 커널 독립 학습
- [ ] 2D/3D, N별 수치 결과와 실행비용 보고
- [ ] 새 independent test는 사전 선택 동결 후에만 사용
- [ ] 결과와 코드가 재현 가능한 GitHub PR 제출

**한 줄 요약:** *PBGIA의 핵심은 입력 마스크를 복원하는 것이 아니라, 기하량과 방향을 잘 담는 자화장을 생성하는 것이다. 생성 미분/교환 연산자부터 학습하고, 동일 관측 연산자 대조 및 독립 관측 커널 재학습을 모두 수행하라. 위상 현상은 핵심 활용이지만, Dice/위상 보존은 학습 제약이 아니다.*
