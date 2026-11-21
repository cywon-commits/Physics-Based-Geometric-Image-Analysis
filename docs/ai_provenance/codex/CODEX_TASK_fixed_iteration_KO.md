# Codex 작업 의뢰서 — PBGIA 고정 반복횟수와 전용 미분 커널 공동 최적화

**우선순위:** 실행 가능한 수치 실험 및 재현 결과를 작성하라. 계획만 쓰고 끝내지 말 것. 현재 패키지의 `code/fixed_iteration_checkpoints.py`는 모델 정의와 체크포인트 저장을 위한 검증 가능한 출발점이며, 대규모 벤치마크/최적화는 이번에 새로 완성해야 할 작업이다. 별도로 제공된 `reference/prior_codex_package.zip`와 기존 연구 결과를 먼저 참고하되 오래된 projected-gradient 결과를 현재 모델과 혼합하지 말 것.

## A. 확정된 모델: 수정 금지

2D: `n=(vx,vy,s)∈S²`; 3D: `n=(vx,vy,vz,s)∈S³`. 이진 입력 `f=+1` 내부, `-1` 외부, 고정. 동기식 정규화 반복:

```
H_v(t) = J * sum_{2d axial neighbors} v(t) + D * G_generation[s(t)]
H_s(t) = J * sum_{2d axial neighbors} s(t) + h * f
n(t+1) = H(t)/|H(t)|
```

- `div(v)` **추가 금지**. `grad(f)` **방향장으로 사용 금지**. positive neighbor-sum을 Laplacian으로 무단 대체 금지. `alpha=1`을 기본으로 하고 damping은 별도 대조군.
- 한 반복의 모든 항은 같은 상태 `n(t)`로 계산. `|H|≤1e-12`면 이전 스핀 유지. 매 반복 단위길이 유지. 경계조건은 주기적일 경우 충분한 배경 여백 사용.
- **비변분적인 정렬 알고리즘:** 단일 에너지의 변분 최소화나 에너지 단조 감소를 주장하지 않는다. 측정량 `Σ tr(T)`는 에너지로 부르지 않는다.
- 마스크 초기화는 binary-mask EDT만 사용; 생성할 때 사용한 해석적 signed distance/법선은 진단/GT 전용.

## B. 연구 질문과 필수 실험

**핵심:** `N∈{5,10,20,30,40}`에서 계산한 **비평형 또는 부분 이완장**에 맞춘 측정 커널 `K_N` 및 기준 보정계수 `c_N`를 학습하여, 정확도-시간 Pareto 곡선을 만들고 **하나의 고정 N**을 선택한다. 적응형 종료는 배포 모델에서 사용하지 않는다. 80/100회 reference는 학습/진단용 참조로만 보관 가능하다.

### B1. 체크포인트 생성 및 비용

- 동일 이미지·동일 `(J,h,D, G_generation, dtype, boundary, initialization)`에 대해 **40회만 1회 수행**하면서 `N=5,10,20,30,40` 상태 저장. 각 N용 독립 이완 5회 중복 금지. `N=0,1,2,3` 진단 가능.
- 필드 생성용 gradient kernel `G_generation`은 **메인 실험에서는 중앙차분으로 고정**. 측정용 `K_N`만 최적화. 생성 kernel 학습은 별도 ablation으로 하여 각 변화에 대해 필드를 새로 이완할 것.
- `J=1,h=2,D=1` 주 분석. 최소 하나 이상의 `h` 값과 `D` 값에서 transfer 검사. 과거의 해석적 벽폭 `sqrt(2A/h)`를 새 모델의 폭이라고 가정하지 말고 측정할 것.
- 체크포인트마다 fixed-point one-step RMS residual, 최대 국소 변화, 비정상 상태와 2-cycle 진단, mask Dice, Euler number, wall fraction, 법선 정렬도, degree(최소 대표 3D)를 저장. **최종 관측량의 정확도가 좋다고 고정점 수렴을 주장하지 말 것.**
- 실제 벽이 해상되지 않은 샘플은 보고하고 커널 비교에서 별도로 표시.

### B2. 사전 지정 데이터 분할

- `train`: kernel parameter 및 calibration coefficient만 학습.
- `validation`: `(N, K_N, h)` 선택 및 목표 지표 trade-off 결정.
- `test`: 위 선택 완전 고정 후 한 번만 최종 점수 계산. 테스트 결과에 맞춘 재조정 금지.
- 무작위 분할은 **기하학적 원형(shape family/size/geometry instance)** 단위로 하되, 그 원형의 회전/이동/노이즈 파생 이미지는 반드시 동일 split. 단순한 회전/위치 변화만으로 독립 test라고 주장 금지.
- 2D 원/스태디움/타원/링/다중 객체, 3D 구/캡슐/토러스/타원체/다중 객체/작은 물체 포함. 각각 radius, orientation, subvoxel translation 다양화. train/calibration 원/구는 test와 분리.
- noise-free 및 고정된 복수 noise rate, 독립 seed. clean benchmark와 noisy benchmark는 별도 집계. ground truth는 **측정 평가**에만 사용, 초기화/이완/테스트시 calibration 금지.
- 가능하면 동일 물리형상에 대한 grid-refinement, domain-margin 점검 별도; 이미지 크기만 크게 하는 것과 해상도 개선을 구분.

### B3. 커널과 측정량

- 각 `N`마다 **공정하게 동일한 parameter dimension과 동일 optimizer budget**으로 `K_N` 최적화. 후보: 3-point central, 5-point fourth-order, Scharr-like, 이전 학습 5×5, 새로운 antisymmetric separable 5×5.
- 상수 미분=0, 선형 gradient=1, 축치환 공변성, `x/y/z`의 일관성 및 단위 응답, 주파수 응답 검사. 대칭/반대칭 계수와 admissible parameter range 기록.
- 2D 둘레/3D 표면적: `T_ij=Σ_a (K_i*n_a)(K_j*n_a)`, `M=Σ tr(T)`, `L/S=c_{N,K,d,h,D} M`; 기준 원/구에서 독립 환산계수 학습. 기존 Codex 보고서가 사용한 바로 이 정의 유지.
- 주축 각도: 전체 텐서 고유축. 스태디움/캡슐의 장축에는 작은 고유값 축을 검토하되, 축 선택/평가 원칙 사전 명시. 이축/축퇴 및 분리된 구형 물체에서 global center-to-center axis를 텐서가 복원한다고 주장하지 말 것. 적절한 PCA baseline 포함.
- 측정 커널이 국소화되어 있음에 유의: 수렴하지 않은 벽과 nonlocal residual 오염이 있을 수 있으며 `N`별 재학습으로 덮을 수 없는 실패도 명확히 제시.

### B4. 최소 필요한 N 선택 규칙

- 우선 `N=40` baseline 참조로 `N=5,10,20,30`의 측정값 차이를 제시하되, 40회가 정확한 GT는 아님. 진짜 형상 길이·표면적·방향 오차는 해석적 GT와 직접 비교.
- Validation의 geometry MARE, 90/95 percentile error, orientation MAE, rotation/shift CV, mask/degree validity, runtime을 사용하여 Pareto front 생성.
- **주요 목표:** baseline 40회 대비 geometry MARE 증가 `≤0.1 percentage point`, orientation MAE 증가 `≤0.1 degree`, 대표 위상/마스크 실패 증가 없음, 계산시간 개선; 모두 사전 선언된 예시 허용오차이므로 지나치게 엄격하면 tolerance sensitivity도 보고. 이 기준을 만족하는 **가장 작은 N** 선택. 만족 조합이 없다면 40회 유지 혹은 성능 요건 재검토, 20회를 임의 확정하지 말 것.
- 위 조건은 데이터가 드문 orientation undefined fixtures에 대해 억지로 적용하지 말 것. test dataset에는 선택 규칙 변경 불가.
- 실제 경과시간 측정은 field relaxation, kernel inference, 전체 파이프라인, baseline 분리. 체크포인트 기반 다중 N 실험 비용과 배포되는 **선택된 단일 N** 실행 비용을 혼동하지 말 것.

### B5. 공정한 대조군

- 기존 코드에서 실행 가능한 baseline: 2D Crofton, marching squares, Gaussian σ=1 threshold / gradient-integral, PCA/structure tensor. 3D voxel faces, marching cubes, Gaussian + marching cubes, PCA, surface-normal tensor.
- baseline의 참조 보정계수도 동등하게 train에서 선택, sigma hyperparameter는 validation에서만 선택. test leakage 금지.
- 정수 `S³` degree는 외부 negative collar의 정당한 compactification, 비특이 보간 조건과 목표점 민감도 점검하에 사용. 디지털 Euler 수와 degree는 서로 다른 관측량.

## C. 재현 산출물 (필수)

1. `README.md` 한 명령 실행, `requirements.txt`, fixed random seeds, code version/commit, CPU/GPU + dtype.
2. Split manifest (`split_manifest.csv`), synthetic generators, saved checkpoint state or reproducible cache keyed by input/kernel/J/h/D/N.
3. `kernel_by_iteration.json`: 각 N의 kernel, normalization conditions, optimizer result, calibration, validation scores; `selected_configuration.json` test 보기 전에 생성.
4. `checkpoint_metrics.csv`, `train_results.csv`, `validation_results.csv`, `test_results.csv`, `baseline_results.csv`, `timing.csv`.
5. 개별 사례 error + paired differences, bootstrap CI **independent geometry units**로 재표본화. shape-family held-out 결과를 분리.
6. 성능-반복횟수 곡선, 커널 계수 히트맵, time/accuracy Pareto chart, wall profiles, 실패 사례. 2D/3D 별도 작성.
7. 한국어 최종 보고서: 고정 `N*` 및 커널을 추천하거나 근거 부족하면 그렇다고 명시. 수렴/위상/정확도/비용과 일반화 제약 구분.
8. 자동 테스트: 원본 `f` 불변, 2D/3D 배열 크기, `|n|=1`, 동기식 map 일치, checkpoint 복원/재사용 일치, split leakage, kernel moments, calibration train-only, 2-cycle smoke, axis permutation.

## D. 실행 순서와 성공 기준

1. 포함된 체크포인트 미니 스모크 테스트 실행 및 수치 재현.
2. 기존 Codex 결과(연결 보고서 아래)를 학습/검증 이력으로만 참조하고 새 실험에서 모든 `N`을 재계산.
3. **학습/검증에서 결론을 먼저 고정**한 후 test 수행.
4. 선택된 고정 N과 전용 K를 최종 코드로 동결하여, 사용자에게 단일 이미지에서 바로 돌아가는 CLI 제공.

이전 Codex 보고서: `https://drive.google.com/file/d/1YBFjIst8FEnsPX_qtI-cNUgQNXZiM3jN/view` (2026-11-16). 내용에는 2D 학습 5×5의 우위, 3D 4차 중심차분의 우위, 분리된 구형 객체의 방향축 실패 등의 반례가 있으므로 재현하고 숨기지 말 것. 원본 full raw CSV 및 코드는 **이 패키지에 없음**; Codex가 이전 실행 저장소에서 접근할 수 없으면 명시하고 여기 코드로 독립 재실행할 것. 보고서만으로 숨은 CSV 존재를 가장하지 말 것.
