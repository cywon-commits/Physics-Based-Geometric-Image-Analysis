# PBGIA: 이전 업로드 코드와 최신 정규화 반복의 방향 측정 비교 — 실행 결과

2026-11-16 · 실제 로컬 Python 합성 실험 · **GitHub 기존 전체 데이터셋을 재실행한 것은 아님**

## 결론 요약

이전 `TextureFilterTrain5 (1).py`의 **커널과 5회 반복을 모두 사용한 원래 방법**은 이번 2D 합성 실험에서 현재 PBGIA-30보다 장축 방향을 더 잘 복원했다. 반면 **같은 최신 정규화장에 옛 네마틱 텐서만 얹으면 성능이 나빠졌다.** 따라서 관측 텐서 하나보다 초기화·교환 커널·이완 갱신·관측 텐서 전체 설계의 상호작용이 중요하다. 원래 방법은 mask 보존 측면에서 더 불리했다.

## 재현 설정

- 원자료: 이전 첨부 `texturefiltertrain5 (1).py`, `lengthorientationapplication (1).py`; 비교 모델: 이전에 전달한 `fixed_iteration_checkpoints.py`.
- 대상: 80×80 2D stadium/ellipse, 반지름 6·10, 각도 3°~87°를 6° 간격으로 스캔, subpixel shift 2개. 기하 도형 120개; 잡음 0%, 0.2%, 1%를 추가해 총 360개 영상.
- **기존 원형 경로**: 학습된 5×5 `kernel_dmx`, `kernel_HJ=kernel_dmx²+kernel_dmy²`, 고정 1 이진 입력, `Sx=Sy=0`, `Sz=f`, 5번의 정규화 이완, `D=1`, pinning 계수 `c=1`, periodic boundary. 원래 소스에서는 교환장 `HJ`와 이완장 결합으로 `T_x = H_{Jx}S_x-H_{Jy}S_y`, `T_y = H_{Jx}S_y+H_{Jy}S_x`; `1/2 atan2(T_y,T_x)`로 director 각도 추출.
- **최신 원형 경로**: mask-EDT 기반 초기화; 축 4-neighbor exchange, `J=1,h=2,D=1`, 최신 `grad(s)` 갱신, 30번 반복; Scharr형 scalar+vector gradient structure tensor의 최소 고유값 방향.
- 별도 **같은 최신장** 비교: 기존 교환장 네마틱 텐서, 직접 `vv` 네마틱 텐서, Scharr 구배 텐서. 기하/각도 convention은 180° director 축으로 맞춤.
- 판정: 1 영상은 GT 생성용으로만 쓰고 실제 입력은 이진 mask, 잡음은 내부 픽셀을 salt-and-pepper flip. 코드 실행 환경의 ndarray axis-0은 `y`, axis-1은 `x`임을 반영.

## 실제 결과

| 잡음 | 이전 5회 교환-네마틱 MAE | 이전 5회 + gradient tensor MAE | 최신 PBGIA-30 + gradient tensor MAE | 이전 mask mean/min Dice | 최신 mask mean/min Dice |
|---|---:|---:|---:|---:|---:|
| 0% | 0.3147° | 0.3448° | 0.4124° | 0.99669 / 0.97987 | 1.00000 / 1.00000 |
| 0.2% | 0.3318° | 0.3793° | 0.4333° | 0.99637 / 0.97987 | 0.99974 / 0.99668 |
| 1% | 0.4537° | 0.4526° | 0.5382° | 0.99472 / 0.97643 | 0.99807 / 0.99029 |

잡음 0%에서 전체 평균 MAE 차이는 약 0.098°, 상대적으로 약 24% 감소(0.412°→0.315°)다. 다만 *최대/평균 Dice와 주축 오차는 다른 목표 함수*이므로 이를 모델 전체가 개선됐다고 해석하면 안 된다.

![Orientation comparison](orientation_comparison.png)

![Mask fidelity](dice_comparison.png)

### 고정된 최신 이완장에서 방향 판독법만 바꾸었을 때

30회 최신장 + Scharr gradient tensor는 stadium에서 **0.529°**, ellipse에서 **0.295°**. 동일장에서 기존 `HJ·v` 이중각 네마틱 지표를 사용하면 stadium **1.476°**, ellipse **2.127°**로 악화된다. `vv` tensor도 각각 **1.781°**, **2.421°**. 이로써 **이전 방향 텐서만 새 모델에 이식하는 것은 도움이 되지 않는다**는 점이 확인된다.

### 스핀장 생성 방식과 커널을 함께 복원했을 때

오래된 코드의 완전한 5회 경로는 stadium 6px/10px에서 bond nematic MAE **0.472°/0.190°**이고 ellipse 6px/10px에서 **0.406°/0.191°**였다. 이 경우 gradient tensor를 써도 상당히 높은 정확도를 보였으므로 생성된 spin profile 자체가 다름을 시사한다.

### 측정 커널을 다시 고르거나 조합한 예비 실험

고정된 최신장에 `scalar gradient tensor + w × vector-gradient tensor`를 적용하고 w={0,0.1,0.25,0.5,0.75,1,2,4} 및 central/Scharr/fourth 차분을 비교했다. 소수 prototype만 train에 둔 시험에서 Scharr가 대체로 우수했으며 **train-selected `w=2`는 heldout에서 `w=1`보다 못했다.** 즉 이 단순 결합으로 일관된 개선을 입증하지 못했다.

또한 기존 생성모델의 `(N,h,D)`를 작은 train split에서 탐색해 8회/h=0.5/D=2를 선택했지만 heldout test MAE 0.256°로, 원래 5회/h=1/D=1의 0.245°보다 나빴다. **재최적화가 기존값보다 낫다는 보장은 없다.** 이 실험은 해당 synthetic split에서만 수행했으므로 커널의 전역 최적성을 주장할 수 없다.

## 신뢰 범위와 결론

1. 이번 2D 실험은 새 GitHub의 원본 2D/3D 독립 시험 세트가 아니며, 3D에는 옛 2D 네마틱을 그대로 확장할 수 없다.
2. 세대별 모델의 `h`, 초기화, 교환 커널, 벡터장, 이완횟수 등이 달라 어떤 요소가 독립적으로 각도 향상을 일으켰는지는 **인과적으로 분리되지 않았다**.
3. 구조 보존은 최신 PBGIA-30이 우수하다. 예전 방식에는 실제 object erosion이 있어, 방향오차만으로 통일된 새 알고리즘을 대체할 수 없다.
4. 검증할 다음 과제: 기존의 **양의 5×5 교환 커널**을 최신 `f`-고정, 동적 `grad(s)` 모델에 넣고 **고정 EDT 초기화, h/D, N까지 같은 조건**에서 교차 비교한다. 생성 커널/측정 텐서의 factorial ablation과 train/validation/test 분리를 수행해 성능과 Dice·topology를 공동 최적화한다. 3D에서 축에 대한 director 텐서를 정당하게 일반화한다.

## 실행 파일

- `compare_orientation.py`: 동일 최신 이완장 네마틱 vs gradient vs PCA.
- `end_to_end_orientation.py`: 원래 5회 모델 vs 새 30회 모델.
- `noise_test.py`: 합성 잡음과 mask fidelity.
- `original_kernel_eval.py`: 원래 생성 교환 커널을 최신장에 적용한 대조.
- `compare_tensor_variants.py`: scalar/vector gradient tensor 혼합.
- `tune_original.py`: 이전 모델 `(N,h,D)`의 제한된 train/test 탐색.
- CSV 파일은 각 실험의 개별 각도 오차와 요약을 보존한다.

실행: `python noise_test.py` 및 각 .py 실행. Python/Numpy/Scipy/Pandas/Matplotlib 필요. 각 스크립트는 같은 폴더의 `PBGIA_Codex_fixed_iteration_20261116/code/fixed_iteration_checkpoints.py`를 참조한다.
