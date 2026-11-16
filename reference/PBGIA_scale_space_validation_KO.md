# PBGIA: 외부장 제어 다중 스케일 및 잡음 검증

2026-11-16. 본 문서는 실제 실행한 소규모 2D 검증 결과이며, 임의의 자연 영상에서의 성능을 입증하지 않습니다.

## 목적

외부장 세기 `h`가 정하는 도메인월 길이 `λ = sqrt(2 A / h)`가 작은 돌출부·구멍과 잡음에 대한 감쇠를 조절할 수 있는지, 그리고 이미지 형태의 개선이 실제 **에너지 기반 둘레 추정**에서도 나타나는지 분리 검증한다.

## 모델과 실제 구현

- 112 × 112 이진 이미지 `f=±1`만 입력. 연속 도형의 ground-truth 거리를 이완 초기화에 사용하지 않음. 거리 변환은 **이진 이미지에서** 계산.
- 3성분 단위장 `n=(vx,vy,s)`, 에너지 `E=A sum_bonds |n_i-n_j|² + h sum(1-f_i s_i) - ε sum(v_i · ∇G_0.85*f)_i`.
- `A=2`, `ε=0.1`. periodic 4-neighbor 차분, 명시적 projected-gradient dynamics `dt=.025` 450회; 마지막 잔여 gradient RMS 확인. `h∈{.35,.65,1,2,4,8}`.
- 분석용 이진 분할 `s>0`, 실제 에너지(교환+외부장+경계장)는 별도 측정.
- 큰 원 반지름 27픽셀에 반지름 5,3 돌출부, 반지름 4,2 구멍, 독립된 작은 원 반지름 3을 결합한 **합성 마스크**.
- 각각 독립적인 random pixel flips 1.2% 4종. 별도 강건성 실험: 0%, 1.2%, 4.0% 및 각각 3 seeds.
- Gaussian baseline: `sign(G_σ * f)` (여러 σ) 및 동일한 reference. 최적 σ는 **정답을 이용해 사후 선택한 oracle**이며 실제 현장 적용 알고리즘에 주어지는 이점이 아니라는 점에 주의.

## 확인된 결과 A: 물체 구조의 선택적 감쇠

`h=.35, .65, 1, 2, 4, 8`에 대해 `λ=3.38, 2.48, 2.00, 1.41, 1.00, .71` 픽셀.

- 작은 독립 원(r=3) 복원 픽셀 비율: `0%, 0%, 28.4%, 69.8%, 86.2%, 99.1%` (1.2% noise, 4 seeds).
- 작은 내부 구멍(r=2) 복원 픽셀 비율: `0%, 0%, 0%, 0%, 57.7%, 94.2%`.

따라서 외부장으로 효과적인 구조 선택 척도를 제어한다는 가설을 지지한다. 단, 결과는 하드 컷오프가 아니고 형상/초기 위상조건/격자 해상도에 의존한다.

## 확인된 결과 B: 순수 분할 마스크의 복원

잡음 1.2%에서 PBGIA는 `h=4`에 Dice 0.9958, `h=8`에 Dice 0.9988 (4 seeds). 그러나 Gaussian(σ≈0.8)도 약 0.9957이다. 잡음 4%에서는 PBGIA `h=8` Dice 0.9957, Gaussian(σ=0.8) Dice 0.9939 (각 3 seeds). 본 실험만으로 일반적인 우위를 주장할 수 없다.

특히 작은 섬과 구멍이 소멸하면서 **clean ground truth 자체와 다른 위상**이 생기는 구간이 있다. 스케일 변화에 따른 위상지표는 별도로 기록해야 한다.

## 핵심적인 발견 C: 분할된 형태가 깨끗해 보여도 에너지 둘레는 과대 추정될 수 있음

기준 원(R=27)의 동일한 h에서 계산된 E_ref로 전체 에너지를 환산: `L_E = 2π·27·E/E_ref`. 정답으로 clean 합성 마스크의 Crofton 둘레를 참고(241.50px); 이는 연속 도형의 정확한 해석적 둘레가 아니므로 정확도 지표로 오해하면 안 됨.

- Clean 합성 물체의 보정 에너지 둘레: `h=.35` 214.36, `h=2` 228.93, `h=8` 241.44px.
- 잡음 1.2% 합성 이미지 4종의 평균: `h=.35` 252.33, `h=2` 320.94, `h=8` 469.91px.
- 같은 h에서 clean *magnetic* 에너지에 대한 noisy 에너지 초과율: **+17.7%, +40.2%, +94.6%**.

따라서 `sign(s)`로 얻은 영역이 매우 깨끗하더라도, 이완장 내부의 잔여 교환/field 에너지가 픽셀 잡음에 민감할 수 있다. 경계/길이 추정에는 모형 에너지 자체에 대한 잡음-강건화 검증이 별도 필요하다. 소수의 결함성 국소 자화 텍스처 또는 잔류 고주파 구조가 원인일 수 있지만, 개별 자화 위상수를 계산하지 않았으므로 이를 위상적 안정화로 단정하지 않음.

## 기술적 한계

1. 450회 이완은 대부분 gradient RMS가 ~1e-3 이하이지만, 가장 약한 외부장의 clean/ noisy 사례는 0.003–0.004로 더 크다. 정확한 최소에너지라고 주장하지 않는다.
2. 사각 격자 4-neighbor, finite-grid periodic boundaries. 구조는 경계에서 충분히 떨어져 있으나 일반적 영상에서는 padding 필요.
3. Gaussian baseline은 각 h의 λ에 단순 일대일 대응하지 않는다. σ sweep 및 oracle best 비교를 사용했다.
4. 4 seeds 및 3 seeds는 신뢰구간 확정을 위해 충분하지 않다.
5. 이번 검증은 **2D**, 3D에서 동일한 특징 선택/잡음 강건성이 성립한다는 주장은 아직 실험하지 않았다.

## 후속 실험을 위한 명확한 가설

- 적절한 h에서 `sign(s)`의 geometry reconstruction은 잡음에 강건하고 작은 곡률 특징을 선택적으로 억제한다.
- 하지만 raw domain-wall total energy는 노이즈 입력을 직접 구동하는 필드 때문에 비정상적 작은 자화 영역에 민감할 수 있으므로, (i) initialization 비교, (ii) `h` schedule/anneal, (iii) reference circle과 같은 에너지 스케일 보정, (iv) energy threshold/connected-component removal 등의 효과를 분리 실험해야 한다.
- 비교 방법에는 Gaussian, median, morphological opening/closing, TV/ROF 모델을 포함하고, 가능한 한 **동일 clean geometry reference**를 사용한다.

## 재현

`python pbgia_scale_space_experiment.py` (4 seeds, noise 1.2%, 6h + montage)  
`python pbgia_scale_space_followup.py` (3 noise strengths × 3 seeds × 5h and Gaussian sweep)  
`python pbgia_scale_space_energy_calibration.py` (6h clean/reference circle calibration)
