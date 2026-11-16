# Codex 실행 의뢰서 — PBGIA 자화장 생성 커널과 기하·방향 측정의 공동 최적화 (2D/3D)

**연구 목표:** 기존 `TextureFilterTrain5 (1).py`와 `LengthOrientationApplication (1).py`에서 채택한 **커널 구조–자화 이완–네마틱 방향 계산을 함께 최적화**하는 철학을 보존하되, 이전 커널의 **숫자를 새 모델에 복사해서 끝내지 않는다.** 동적 `grad(s)` 정규화 맵을 새 표준으로 유지하면서, 주어진 고정 연산 횟수 안에서 형상·위상을 보존하고 정확한 기하량/주축을 복원하도록 `G_gen`, `K_J`, `K_measure`, 방향 측정 구조와 `N`을 학습한다.

**중요:** 이 패키지 `code/joint_core.py`는 수학식, 2D/3D 공변성, 구현 스모크 검증용 **최소 동작 코드**다. `smoke_results/`의 수치는 적은 표본에 대한 예비 점검이며 학습·validation·독립 test 결과가 아니다. 대규모 독립 벤치마크와 실제 공동 최적화는 Codex가 새로 완성하고 실행해야 하는 본 과제다. **계획만 제출하지 말고 재현 코드를 실제 실행하라.** 3D에 계산 자원이 부족하면 실행한 범위와 이유를 명확히 보고하고 거짓 성능값을 만들지 말 것.

## 0. 반드시 먼저 읽을 자료

1. 이 패키지 `README.md`, `code/joint_core.py`, `tests/test_joint_core.py`, `smoke_results/smoke_raw.csv`.
2. `reference/texturefiltertrain5 (1).py`, `reference/lengthorientationapplication (1).py` (사용자 원래 코드의 정확한 식 확인).
3. `reference/PREVIOUS_ORIENTATION_COMPARE_KO.md` (원래 5회 모델의 2D 방향 0.315°, 현재 30회 약 0.412°, Dice 차이; **다른 synthetic suite** 결과임).
4. `reference/PBGIA_theory_algorithm_KO_2026-11-16_v1.2.md` (연속 스텝장 해석해가 새 normalized map의 정확해가 아님).
5. GitHub https://github.com/cywon-commits/Physics-Based-Geometric-Image-Analysis 의 `fixed_iteration/output/REPORT_KO.md`, `selected_configuration.json`, `fixed_iteration/code/`, `results/`. **GitHub 기존 시험을 다시 학습에 사용하지 않도록 별도의 신규 split을 만들 것.** 코드를 가져오면 버전과 SHA 기록.
6. `reference/PBGIA_Codex_fixed_iteration_20261116.zip`와 `reference/PBGIA_orientation_legacy_vs_dynamic_experiment.zip`는 재현 출처 및 대조용이다. 그 안의 기보고 성능 숫자를 새 모델의 결과처럼 인용하지 않는다.

## 1. 공통 수학 모델 — 각 단계 동기 업데이트

2D는 `n=(v0,v1,s)∈S²`, 3D는 `n=(v0,v1,v2,s)∈S³`, 이진 영상 `f∈{-1,+1}` 고정 (`+1` 내부). 배열 공간축 i와 v_i 순서를 일치시킨다. 배포에선 `N` **고정** (적응형 종료 없음).

```
H_v(t) = J [K_J(phi) * v(t)] + D [G_gen(phi) * s(t)]
H_s(t) = J [K_J(phi) * s(t)] + h f
n(t+1) = H(t) / |H(t)|
```

`*`는 명시된 correlation/convolution 중 하나로 일관되게 구현. 하나의 갱신에 이전 상태 `n(t)`만 사용. `|H|<=eps`면 전 단계 유닛벡터 유지. `grad(f)` 외부 고정 방향장, `div(v)` 추가, `J∇²n`으로 무단 교체, 변분 DMI 최소화라고 주장, 다른 완화 규칙으로 바꾸기는 모두 **금지**. 단위길이 오차/한 단계 잔차/2주기 진단. 최우선 주 파라미터 `J=1,h=2,D=1`, 추가 transfer `h=4, D=.5` 등. 진단용 60~100회는 가능하지만 최종 배포에서는 사전 선택 고정 `N` 사용. Binary EDT 초기화 기본; 원래 코드의 `v=0,s=f` 초기화는 **독립 ablation**으로만 사용.

## 2. 기존 논문의 기하학적 해석과 원래 코드의 학습 원리

원래 코드의 필터 설계는 x/y 편미분 커널 `G_x`, `G_y`에서 원소별 제곱으로 **양의 교환 커널** `K_J=G_x∘G_x+G_y∘G_y`를 유도한다. ∘는 **계수의 Hadamard 제곱**이지 연산자 합성 `-G†G`가 아님. 3D 후보는 `K_J=G_0∘G_0+G_1∘G_1+G_2∘G_2`. 원래는 미분 커널 수정이 생성되는 자화장도 바꾸므로, 이전 학습의 이점은 **커널과 동역학의 결합 설계**에서 올 수 있다.

반드시 분리할 교환 커널 계열:

- `E0`: current axial 2d-neighbor positive sum, 기존 GitHub 모델 완전 재현.
- `E1`: user-inspired tied `Σ_i G_i²` (Hadamard), 음수 없음. 원래 코드의 커널 스케일(unnormalized)과, 총합 `2d`로 맞춘 scale-normalized 버전 둘 다 비교; **스케일 자체가 J/D/h의 상대값을 바꾸므로 반드시 통제**.
- `E2`: x/y/z 축치환 대칭성, nonnegative, inversion symmetry, 정규화 `Σ K_J=2d`를 만족하는 **독립 학습 positive exchange kernel**, `G`에 묶이지 않음.
- `E3`: 원래 binary 초기화 + 사용자 원래 모델을 **historical end-to-end control**로 별도 재현(2D). 원 모델의 boundary='symm'/'wrap', J scale, h,c,D 차이를 기록하라. 이 방식을 현 normalized standard라고 슬쩍 합치지 말 것.

편미분 생성 커널 `G_i`는 2D에서 반대칭 5×5, 3D에서 축 방향 반대칭·나머지 축은 짝대칭인 5×5×5를 시작으로, 상수장 미분=0, 좌표 i 선형 기울기=1, 타축 선형 응답 0, 축치환/반사 공변성 확인. 5^3 full unrestricted coefficients는 비용과 과적합이 크므로 처음에는 axial 5-point × transverse smoothing의 **저차원 separable family**로 학습, 필요할 때 증가. 계수 제한/주파수 응답/유효장 near-zero/벽 해상도를 기록. **SO(3) 완전 회전 불변, BPS, 정확한 물리 DMI** 등은 증명 없이 주장하지 않는다.

## 3. 방향 관측량: 2D 및 3D를 엄밀히 비교

### A. 기존 2D bond nematic

현재 교환장 `H_J = K_J*v`로 정의 (D grad(s), hf와 구별!). 원래 코드:

```
T0 = sum (HJ_0 * v_0 - HJ_1 * v_1)
T1 = sum (HJ_0 * v_1 + HJ_1 * v_0)
angle = 0.5 * atan2(T1,T0)
```

이 값이 **법선 축**인지 **물체 장축**인지 입력 공간축/부호와 물리적 진단으로 결정하고, ±90° 오차 보정/축 선택을 *test 이후* 결정하지 말 것. 2D `Q_bond=[[(T0)/2,T1/2],[T1/2,-T0/2]]`의 축과 일치하도록 unit test.

### B. 3D generalization (비동등성을 인정하라)

```
B_ij = sum_x 0.5*(HJ_i(x)*v_j(x)+HJ_j(x)*v_i(x))
Q_bond = B - tr(B)/3 * I
Q_direct = sum_x (v_i*v_j - |v|^2/3 * delta_ij)
```

실수 대칭 traceless 텐서 고유축(최소/최대 eigenvalue), eigengap, 방향의 부호 불변성. 2D 수식과 3D symmetrized product의 2D restriction이 동등함을 테스트. `H_J`가 `v`에 평행이 아닐 때 3D는 2D 반각 공식만으로 표현할 수 없다. 3D 기울어진 축·축 교환·반사에 대한 공변성을 시험하고, 분리된 구체/완전 구에서 전역 장축을 무리하게 예측하지 말 것.

### C. 새 gradient-structure tensor 및 ensemble

`T^G_ij = Σ_{x,a} (K_obs,i*n_a)(K_obs,j*n_a)`; 원래 Colab의 `T_x,T_y`와 구분. 벽 법선 분포의 정렬과 같은지 검증. 방향 계산에는 `Q_bond`, `Q_direct`, `T^G` 세 방법과 사전 학습된 **조합** `Q_ens = z-score/trace-normalized tensors`를 평가. 조합의 eigenvector sign/축반전은 validation에서 고정하고, test GT를 사용하지 않는다. 구조가 축퇴된 경우 orientation=undefined와 confidence 기록. 모든 방법에 동일 dataset 적용. 형상 인식 후 GT를 보고 방향 관측량을 선택하면 leakage임.

## 4. 실험 구조: 변화 원인 식별용 Factorial Ablation

`G_gen`(central / trainable), `K_J`(axial / tied / independently learned), `orientation`(bond/direct/gradient/learned combination), `K_obs`(central/fourth/Scharr/learned), `initialization`(EDT / legacy binary control), `N∈{5,10,20,30,40}`를 분리하여 실시. **전체 Cartesian product는 필요 없음**: 아래 순서로 계산을 절약.

1. **Stage 0 correctness:** 같은 모델/seed/shape의 old GitHub PBGIA-30 baseline 재현; 2D historical full path를 원 첨부 소스와 수치 교차검증; 3D tensor identity, axis permutation/sign inversion, kernel moments tests.
2. **Stage 1 observer-only:** 생성장을 axial+central, EDT로 고정하고, 같은 40회 rollout에서 N별 state 저장. 그 상태에서 `bond/direct/gradient`/몇 가지 혼합법을 공정하게 비교. 2D에서 기존 텐서만 이식 시 악화된 이전 관측을 재현.
3. **Stage 2 generator-only:** 같은 `K_obs` 및 방향 텐서(2D/3D 개별)를 고정하고 `G_gen, K_J` family별 자화장 다시 이완. 형태/위상 실패를 포함해 측정. exchange amplitude 재정규화로 단순 스케일 착시 제거.
4. **Stage 3 constrained joint learning:** Stage 1/2에서 확인한 소수 조합만 골라 `G_gen,K_J,K_obs,N` 최적화. Nested validation 또는 2-stage alternating optimization을 사용할 수 있지만 **유한 예산, multi-start, calibration/reference, train/validation/test 금지 경계** 공개. 모든 후보 동일한 eval-budget 또는 wall-time budget으로 비교; 성능 상위 조합은 equal-budget rerun.
5. **Stage 4 independent frozen test:** 미사용 test split에서 2D/3D 별도 평가 및 기존 Codex, original legacy, raw/calibrated Crofton·marching cubes·PCA 비교. 

### 점수 및 제약

- 주축: 2D 각도 180°-periodic MAE/RMSE/max, 3D `acos(|u·u_GT|)` 각도(도). 각 축의 eigenvalue gap/confidence, miss-rate.
- geometry: 독립 원/구 보정 `c`를 이용한 둘레/면적 MARE, p90/p95. `trace(T^G)`는 **비변분 측정량**, Hamiltonian energy라고 부르지 않음.
- stability: clean/noisy mask Dice, IoU, components, Euler, topological degree (3D). `S³` degree는 exterior collar compactification과 piecewise interpolation nonvanishing certificate, 여러 targets 확인. binary mask Euler와 degree 혼동하지 말 것.
- geometry constraints: 최소 clean Dice≥0.995 (우선 실험상 목표; 결과가 불가능하면 제약 변경은 새 validation protocol로만), failure count가 기존 PBGIA-30보다 늘지 않음, independent topology degradation 없음. 원래 legacy의 0.99669 평균 Dice는 최악값을 가리므로 평균만 통과 기준으로 삼지 않는다. 작은 물체/약한 장/높은 곡률 별도 분석.
- 비용: relaxation + observation + initialization 총 CPU/GPU time, memory, training expense 별도. 배포는 **고정 N 하나**; adaptive stop 사용 금지. 실제 수치로 Pareto 지배관계를 평가.
- 학습 손실 예시 `L = wg geometry_MSE + wtheta robust_axial_error² + wrot rotation/shift_CV² + wall/conditioning penalty`; shape/topology는 가능하면 **hard constraint**. GT는 train objective에만, validation은 선택에만, test는 한번만.

## 5. 분할/학습에 대한 엄격한 원칙

- 각각 train / validation / final-test **shape prototype, radius, axis, noise seed 파생 데이터** 전체 단위 분리. 2D stadium/ellipse/circle/ring/multi, 3D capsule/ellipsoid/sphere/torus/multi/small sphere. 특히 **shape-family-held-out**과 novel scale, orientation, subpixel shifts 별도 분석.
- 원래 코드의 200×200 회전된 rounded rectangle 451장(0..90 step .2°)은 학습 시 rotational augmentation으로는 유용하나 **독립 표본 451개가 아님**. test에 같은 geometry prototype의 회전본이 들어가면 안 됨.
- 모든 생성/실험 필드는 binary EDT initialization만 기본. 분석해/정확한 normal은 synthetic truth 생성·진단에만 사용. 
- 고정 N은 각 **차원별 독립 선택 가능**; previous shared N=30은 현 baseline으로 유지. N=40을 accuracy GT로 착각 금지. 각 N에 최적화된 kernel/coeff 별개; 수십·수백번 이완을 학습시 요구하지 않음.
- trained kernel `K_obs`와 generator derivative `G_gen`은 **서로 다른 커널**, 의도적으로 tie할 경우 별도 실험군으로 기록. 
- GT 레이블/각도 convention 및 min/max eigenvector mapping은 train/validation에서만 결정. 작은 eigengap인 원/구/토러스는 방향 ground truth의 존재 여부를 먼저 따짐.
- 각 parameter의 bounds, fixed seed, objective eval count, final optimizer status, floating precision, convolution scaling, axis order를 JSON으로 기록하고 `pretest_selection.json`을 frozen SHA로 묶은 뒤 test 실행. 보고서만 생성하지 말고 원시 CSV와 그림 보존.

## 6. 기대 최종 산출물

- `run_all.py` 또는 CLI (one-command reproduce), `requirements.txt`, 테스트, version lock, split manifest, experiment config, cache keys, checkpoints.
- 2D/3D factorial-ablation raw CSV + metric aggregation + angular-error-vs-rotation plots + Dice-vs-angle + eigengap confidence + kernel frequency response + residual/2-cycle + wall thickness.
- `trained_generator_kernels.json`, `trained_measurement_kernels.json`, `kernel_J_tied_vs_free.csv`, `orientation_observer_comparison.csv`, `joint_pareto.csv`, `baseline_comparison.csv`, `test_results.csv`, `timing.csv`, `topology_audit.csv`.
- 한국어 상세 `RESEARCH_REPORT_KO.md`에 **실제 실행** 방법/범위, 2D/3D 별도 best fixed N, 학습 파라미터, 필요 재보정, novel shape 일반화, 3D 위상/경계/구조 보존 실패, 이전 모델 대비 통계적 paired comparison, 부정적 결과, 향후 작업을 포함.
- 선택된 구성의 **frozen inference CLI**: mask → fixed iteration n → perimeter/surface & orientation, confidence, topology validity warnings; 원래 커널의 magic numbers를 고정해 쓰지 말 것.
- 결과가 기존 PBGIA-30 또는 Crofton/PCA보다 나쁘면 그대로 보고하라. 논문상의 새로움/보편성 주장은 별도 검증 없이는 하지 말 것.
- 가능하면 같은 GitHub repo의 **새 브랜치에서 PR을 만들어** 코드/문서를 제출; `main`과 현재 수치·학습 파일을 덮어쓰지 말 것. 계산 환경에서 GitHub 사용이 어려우면 완결 ZIP 결과물과 재현 지침을 제공.

## 7. 선별된 예비 계산의 의미

이번 패키지의 최소 스모크는 2D/3D 합성 stadium/ellipse/capsule/ellipsoid에서 `K_J={axial,tied,free}`, `G={3-point,4th-order}`, `N={5,10}`와 방향 관측량 5개를 교차 비교한다. **720개 measurement rows** (일부 동일 필드의 반복 관측량)이며 5개 pytest가 통과했다. 이 코드와 함께 포함된 `smoke_results`는 학습도 아니고 최종 정확도 검증도 아니다. 2D에서는 bond 네마틱이 좋은 조건도 있고, 3D에서는 gradient 텐서가 좋은 조건도 있으므로 단일 관측량의 보편적 우월성을 미리 가정하지 말 것.
