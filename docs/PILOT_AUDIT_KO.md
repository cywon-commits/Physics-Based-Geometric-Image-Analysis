# 예비 연구 결과 점검 (2026-11-16)

새로 검토한 `pbgia_joint_benchmark_v2.py`는 2D 140회, 3D 125회의 projected gradient 이완을 사용하며 최종 projected gradient residual이 2D ≈0.0018, 3D ≈0.0091까지 남는다. 이 수치로 '완전 이완'을 주장하면 안 된다.

고정 조건 A=2, h=2, ε=0.1, λ=√2이고, 이완에 쓰는 교환커널은 nearest neighbor이다. 학습 대상은 자화 계산 이후의 5×5 **측정용 미분 커널** (longitudinal difference + transverse smoothing)이다. 2D train R=14, test R=23의 원/스태디움/타원; 3D transfer는 구 R=8,10, 캡슐 R=7 총 세 사례. 기준 2D 원 R=14와 기준 3D 구 R=8을 사용한다.

검증 2D 전체 길이 MARE: Scharr-like 0.674%, learned 0.639%; 방향 MAE: Scharr 0.197°, learned 0.284°. 3D 예비 길이/표면적 MARE: Scharr 0.518%, learned 0.488%, 3D 방향은 단 1개 캡슐이므로 Scharr 0.219°, learned 0.207°. 표본이 매우 작아 수치 우위 단정 불가.

학습된 파라미터 p=(b,t1,t2)=(-0.0948075, ~0, 0)는 최적화한 제한된 파라미터 공간의 경계에 놓인다. 일반적인 의미에서 isotropic optimal derivative라고 해석할 수 없고 다른 h에 대한 전이도 미검증이다. `pbgia_learned_derivative_5x5.csv`는 x방향 좌표 convention을 확인한 뒤 사용한다.

**관찰된 코드 점검 항목:**
- `shape3`의 capsule half-cylinder length는 `R`; 따라서 캡슐 표면적 `4πR²+4πR²=8πR²`의 geometry definition을 테스트에서 명시.
- 전역 `np.roll` periodic boundary 사용: buffer margin은 있지만 GT와 경계조건 일치 여부 확인.
- `s=np.tanh(EDT/...)` 초기조건은 mask-derived. 원본 연속 signed-distance 유출 없음.
- 3D angle MAE는 sphere가 NaN이라 캡슐 단일 표본뿐이며, `n=3`가 angle 평가 사례 3개를 의미하지 않는다.
- `trace(T)` ≠ exchange energy if the measurement derivative differs from the exchange discretization. 두 추정량은 별도로 기록.
- 반복수만으로 수렴 판정 금지. 토러스는 기존 결과에서 500회 이후에도 위상전환/구조재편 경험.
- 상수/선형장 미분 정확성, x/y/z 회전 covariance, 대칭 텐서, calibration leakage 테스트 필수.
