# Codex 작업 의뢰서 (개정) — 동적 법선 구배 PBGIA, 정규화 반복, 커널 최적화

## 최우선: 사용자 지정 모델을 정확히 구현
기존 코드 `reference/lengthorientationapplication (1).py`, `reference/texturefiltertrain5 (1).py`를 기준으로 해석한다. 3D 질서변수는 `n=(vx,vy,vz,s)∈S³`, 2D는 `(vx,vy,s)∈S²`. 영상/voxel에서 얻은 `f=±1`은 **처음부터 끝까지 고정**한다.

반복 t에서:

```
Hv = J * sum_neighbors(v(t)) + D * grad_discrete(s(t))
Hs = J * sum_neighbors(s(t)) + h * f
H  = concatenate(Hv,Hs)
n(t+1) = H / norm(H)
```

- 모든 계산은 각 단계의 **갱신 중인** `s(t)`, `v(t)`를 사용한다. `grad(f)`를 정렬장으로 쓰지 않는다.
- **`div(v)` 결합항은 넣지 않는다.** 이 모델을 `-D v·∇s` 에너지의 변분 최소화라고 주장해서도 안 된다.
- **정규화 정렬**이 핵심이다. 투영된 gradient-flow/LLG는 별도 대조군일 수 있으나 사용자 지정 기본 알고리즘이 아니다.
- `J sum_neighbors`는 기존 업로드 코드의 positive coupling convolution과 대응한다. `J Δn`과는 `6J n` 차이가 있는데, 정규화 fixed-point에서 둘은 동등하지 않다! `lap(n)`을 무심코 대체하지 말 것.
- 작은 H일 때 처리, 2-cycle 안정성 및 b1-mixing 필요 여부 검증. `alpha=1` 기본, damping은 대조군.
- 주기적 경계조건은 충분한 margin에서만 사용; 축 회전 대칭 검증 필수.

## 기존 연구에서 이어받을 내용
- `reference/` 연구제안서/실험보고서, `code/` 과거 계산, `pilot/` 커널 학습 결과. **이전 `∇f` 및 projected GD 모델의 결과는 새 모델에 대한 수치 증거가 아니다.**
- 자화 초기화는 **이진 마스크로부터 얻은 EDT**만 허용. 해석적 signed distance는 ground-truth 생성/법선 평가용으로만 사용.
- 독립 reference circle/sphere로 geometry coefficient 고정; test GT 누출 금지.
- 이 반복에는 단일 변분 에너지 감소 정리가 없으므로 `E_total` 단조성으로 수렴 주장 금지. 수렴성은 fixed-point residual `||n(t+1)-n(t)||`, 반복 변화와 홀짝 진동으로 평가.

## P0: 바로 시행할 사전 검증
1. `code/pbgia_dynamic_normal_test.py` 실행, torus `R=12,r=4.5` / `sphere R=9`에서 `h, D, J` sweep. 포함된 CSV는 **제한적 예비검증**, 확정적 우수성 증거 아님.
2. 정량 기록: torus input 내부지름 15, 외부지름 33 voxel; Dice, components, Euler chi, core hole occupancy, 법선 signed/absolute cosine, `v·grad(s)` alignment, wall voxel count, 스칼라 전이 profile width, residual, fixed-point 2-cycle, 수렴.
3. 예전 고정 `grad(f)` 정규화 모델 대조와, 과거 projected GD 모델(별도로 표기) 차이 명확히 유지.
4. `Q3_fd` 외에 `code/pbgia_s3_integer_degree.py`로 정수 degree를 여러 목표점에서 반복 측정. 경계 상수성/격자 특이성/위상전이 여부 검사.
5. 토러스 축을 회전하고 R/r을 늘려 곡률/격자방향 효과 분리; sphere/capsule 비교. 작은 객체 붕괴 여부 조사.
6. 시간 별 3D 스냅샷 및 원본 이론적 법선과 비율/부호 그래프 생성. 데이터를 먼저 보고 결론 판단.

## P1: 미분 커널 최적화
- 먼저 h에 따른 실제 **완성 자구벽**의 폭·프로파일 측정. `λ=sqrt(2A/h)` 식은 이전 gradient-flow 에너지 모델의 개략적인 값; 새로운 normalized update에 그대로 보장되지 않는다. 폭을 새로 계측할 것.
- **동적 구배 생성 커널**과 **측정용 미분 커널**은 분리해 실험. 먼저 3/5point central, Scharr, 4차 정확도; 그 후 5×5 trainable low-rank/separable antisymmetric kernels. 동적 구배 kernel 변화는 self-consistent field까지 바꾼다.
- 조건: 상수장 도함수=0, 선형장 기울기=1, rotational covariance, 주파수 응답, 안정된 normalized fixed point.
- 둘레/표면적 (reference calibrated gradient tensor trace or perimeter integral), 주축 각도, rotation/shift CV의 다목적 Pareto. calibration에 쓸 원/구와 test shape 분리.

## P2: 독립 벤치마크
- 2D circle, stadium, ellipse, ring, multi-object, noise levels; 3D sphere, capsule, torus, ellipsoid, high-curvature fixtures.
- 대조: binary Crofton, marching squares, PCA, structure tensor; 3D marching cubes, voxel geometry, PCA, surface tensor. Gaussian+baseline은 별도 기록.
- run별 seed, h, D, J, iteration, residual, profile width, mask fidelity, topological validity, trained kernel, 2D/3D dimensions, unit conversions, runtime, errors, CI 모두 CSV.
- 과거 결과와 다른 모델에서 얻은 값 혼용 금지. 실제로 PBGIA가 기존 방식보다 나쁘면 그대로 결론.

## 결과물
재현 코드/테스트, 실행 config, raw CSV, figures, 종합 report, 실패 케이스, CLI 및 환경 설치 문서. 긴 계산은 checkpoint를 사용하고 수행 범위/미수렴을 숨기지 말 것.

## 최초 지시문
"이번에는 `CODEX_TASK_KO.md`의 **동적 ∇s 정규화 반복 모델**을 고정된 정의로 사용하세요. `div(v)`와 `∇f` 정렬항을 넣지 마세요. 우선 P0 검증부터 실행하고, 그 결과가 안정적이면 P1/P2로 진행하세요. 과거 projected-gradient와 fixed-grad(f) 데이터는 독립적인 대조군으로만 표기하세요."
