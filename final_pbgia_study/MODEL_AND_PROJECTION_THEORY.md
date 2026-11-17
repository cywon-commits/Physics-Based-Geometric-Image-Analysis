# Z₂→Sⁿ 생성, 물리적 에너지, 기하 관측과 RP 사영

## 1. 입력과 장의 차원

공간 차원을 d=2 또는 3이라 한다. 고정 이진 입력 fᵢ∈{−1,+1}, unit spin nᵢ=(vᵢ,sᵢ)∈Sᵈ⊂R^(d+1)이며 v∈Rᵈ다. 본 연구에서 S²와 S³는 공간 자체가 아니라 spin의 target이다. f의 부호는 inside/outside를 표현한다. f를 고정하면 전역 n→−n은 일반적으로 같은 문제의 해가 아니다. (f,n)을 함께 뒤집는 변환은 동역학과 대응한다.

binary EDT의 signed difference를 ξ=EDT(f>0)−EDT(f<0)라 하고, s₀=tanh(ξ/2.3), v₀=√(1−s₀²) Gs₀/max(|Gs₀|,ε)로 놓고 전체 n을 정규화한다. GT 연속 거리장이나 법선은 이 식에 들어가지 않는다. EDT는 grid sample의 거리이며 ξ의 벽 근처 jump는 연속 signed distance와 다를 수 있다.

## 2. 기본 normalized alignment

Gⱼs=(sᵢ₊ₑⱼ−sᵢ₋ₑⱼ)/2와 periodic boundary를 사용한다.

```math
H_{v,i}=J\sum_{j\in NN(i)}v_j+D Gs_i,\qquad
H_{s,i}=J\sum_{j\in NN(i)}s_j+h f_i,
\qquad n_i^{t+1}=\begin{cases}H_i/|H_i|,&|H_i|>10^{-12}\\n_i^t,&\text{otherwise.}\end{cases}
```

J=D=1,h=2, N=30이다. 양의 NN 이웃합은 Laplacian으로 바꾸지 않는다. s와 exchange는 같은 이전 시점에서 동기식으로 계산한다. 현재 s의 gradient를 매번 사용하며 grad(f)로 대체하지 않는다. 기본 scalar 식에 div(v)는 없다.

단위길이는 유효 H의 정규화로 유지되지만 이것만으로 수렴, 에너지 감소 또는 시점 사이 topology 보존이 따라오지는 않는다. residual r₁(t)=RMS|F(n_t)−n_t|와 r₂(t)=RMS|F(n_t)−n_(t−1)|를 구분한다. 큰 r₁, 작은 r₂는 2-cycle과 양립한다. 둘 다 작다는 수치 결과는 해당 상태의 안정성을 지지할 뿐 일반적 수렴 정리는 아니다.

## 3. Hamiltonian과 비상호 결합

periodic central G의 이산 adjoint는 G* v=−div_G v다. 실제 계산한 후보 Hamiltonian은

```math
\mathcal H[n;f]=-{J\over2}\sum_i n_i\cdot\sum_{j\in NN(i)}n_j
-h\sum_i f_i s_i-D\sum_i v_i\cdot Gs_i .
```

따라서 ambient negative gradient는 v 성분에서 JΣNNv+DGs, scalar 성분에서 JΣNNs+hf−D div_G v다. 기본 모델은 마지막 reciprocal 항이 없으므로 이 Hamiltonian의 gradient가 아니다. v←Gs라는 일방향 결합을 변분적으로 일관된 상호작용이라고 부르면 안 된다. 이 논의는 모든 가능한 비표준 Lyapunov 함수의 존재를 부정하는 정리는 아니다.

비교군은 H=−∂ℋ/∂n에 대해 tangent vector p=H−n(n·H)를 사용하고 n′=(n+ηp)/|n+ηp|로 retraction한다. η=.1부터 backtracking하며 ℋ(n′)≤ℋ(n)−10⁻⁴η||p||²를 확인한다. 기본 알고리즘과 다른 모델/solver다. reciprocal 항만 넣고 full synchronous alignment를 하면 자동 descent가 되는 것도 아니다.

이를 보이는 checkerboard를 만들 수 있다. 짝수 periodic grid에서 σᵢ=(−1)^(Σi), fᵢ=σᵢ, nᵢ=σᵢ(√(1−.7²),0,…,.7)로 놓는다. central Gs=0, 모든 이웃은 −n이므로 다음 spin은 여전히 checkerboard이며 scalar 계수는 (2−2d·.7)/√((2d)²(1−.7²)+(2−2d·.7)²)다. exchange 값은 변하지 않고 외부장 정렬이 나빠져 ℋ가 site당 +1.93936(2D)/+2.31350(3D) 증가한다. 이것은 EDT 입력에서 흔히 발생한다는 주장이 아니라 엄밀한 monotonicity 주장의 반례다.

## 4. 기하 관측은 다른 양이다

고정 관측 미분 K로 Tᵢⱼ=∫Σₐ(Kᵢnₐ)(Kⱼnₐ)dV, I_E=tr T=I_s+I_v를 정의한다. 이 양은 위의 exchange/external/cross Hamiltonian과 다르다. 연속 단위는 I_E∼L^(d−2)이고 구하려는 경계 크기는 L^(d−1)이므로 환산계수 c_E가 길이 차원을 가진다.

국소 평면벽을 n=(ν sinθ(u),cosθ(u)), |ν|=1로 쓰면 |∇n|²=(θ′)²이고 단위 경계 크기당 I_E는 ∫(θ′)²du다. 벽 폭·profile이 일정하면 경계 크기에 비례한다. 곡면에서는 ∇ν의 기여가 생긴다. 대략 I_v에는 sin²θ|∇ν|²가 더해지므로 곡률과 벽 중첩이 하나의 c로 사라진다고 가정할 수 없다.

TV=½∫|∇s|dV는 완전한 단조 평면 전이의 경우 폭과 무관하게 단위 경계 크기를 준다. 일반적으로 coarea는 s level-set 크기의 평균을 주므로 항상 s=0 크기와 같지는 않다. G4의 이산 absolute gradient는 sharp wall에서 ringing·sampling 편향을 가질 수 있다. s=0 marching도 scalar interpolation의 기하량이며 원래 연속 GT와 동일한 보장은 없다.

관측 kernel은 axial weights ±(.5−2b),±b와 횡방향 [u,t,1−2t−2u,t,u]를 조합한다. 차원별 p=(b,t,u)는 [frozen 설정](reference/PBGIA30_frozen.json)에 고정했다. 이는 5×5/5×5×5 관측자이고 생성 중앙차분과 다르다. TV의 K는 별도 G4다. spacing a에서 물리 I_E는 a^(d−2), TV와 contour 크기는 a^(d−1)로 환산한다. 같은 pixel 모델을 그대로 확대하면 물리 c_E도 a 배여야 한다.

normal tensor(energy T 또는 ∫ρ ννᵀ)와 위치 covariance ∫ρ(x−μ)(x−μ)ᵀ/∫ρ는 서로 다르다. 같은 round component를 이동시키면 normal 분포는 거의 같고 위치 covariance는 중심 간 거리를 반영한다. 따라서 normal tensor에서 임의의 물체 연결축을 읽을 수 없다는 것은 정상적인 비식별성이다. 원/구에는 고유 전역 축이 없다.

## 5. 점별 projector와 부분공간 평균

unit n에 대해 Q=nnᵀ는 symmetric PSD, trace1, rank1, Q²=Q이며 n과 −n을 식별한다. 따라서 Q는 RPᵈ의 한 점이다. 원래 3D n은 RP³이고, v≠0에서 ν=v/|v|로 만든 ννᵀ는 RP²다. 코드의 vector 입력 vvᵀ는 amplitude |v|²를 유지하는 rank-one PSD tensor이므로 trace1 projector와 구분한다. v=0에는 vector director가 없다.

fiber coordinates u를 평균하면

```math
\bar Q(X)={\int w(X,u)Q(X,u)du\over\int w(X,u)du}
```

이며 일반적으로 여러 방향의 양의 합이다. 예를 들어 e₁e₁ᵀ와 e₂e₂ᵀ의 같은 가중 평균 diag(.5,.5,0,…)는 rank2, Q²≠Q이고 최대 eigenvalue가 퇴화한다. 이를 RP 점이라고 부를 수 없다. 계산에서는 full nnᵀ와 vector vvᵀ, w=1와 1−s²를 별도로 사용한다. 분모가 0이면 유효 tensor가 없는 것으로 처리한다. vector vvᵀ를 평균 전에 정규화하는 다른 방법은 이번 대조에 넣지 않았다.

trace>0인 평균을 trace로 정규화하고 λ₁≥λ₂≥…라 하자. λ₁>λ₂이면 유일한 leading eigenline [d]∈RP^(m−1)을 정의할 수 있다(m은 matrix 크기). λ₁=λ₂에서는 그 방향이 비식별적이다. 코드에서는 trace>10⁻¹⁰, λ₁−λ₂>.001로 수치적 정의 영역을 표시했다. purity=tr(Q_normalized²), effective rank=1/purity를 함께 기록한다. rank>1인 tensor에서 leading line을 추출해도 tensor 자체가 rank1이 되는 것은 아니다.

대칭 perturbation ΔQ에 대한 leading eigenline의 변화는 Davis–Kahan 유형의 bound에서 ||ΔQ||/gap에 의해 제어된다. 따라서 작은 gap에서는 작은 잡음에도 큰 회전이나 axis swap이 가능하다. 실측 PSD 혼합 perturbation은 이 조건부 안정성을 지지하지만 모든 입력 잡음의 성능 정리는 아니다.

전체 nnᵀ의 uniform 평균은 bulk n≈(0,…,±1)에서 같은 scalar pole projector를 많이 더하므로 ‘정의된 director’가 실제 공간 법선이 아닐 수 있다. wall 가중 또는 vector 투영은 이를 완화하지만 zero trace와 혼합 방향의 문제를 남긴다. 적분 방향이 달라지면 다른 fiber를 합하므로 결과 tensor와 결함 패턴도 달라질 수 있다.

## 6. 방향선, loop lifting, 위상량

native v의 integral curve는 공간적 방향 질서의 시각화다. 외부 f에 대한 sign 규약으로 inward normal을 고를 수 있지만 속도나 물질 이동량이 주어지는 것은 아니다. projector만 주어지면 ±d가 같으므로 별도의 lift/부호 경계조건 없이 signed flow를 복원할 수 없다. 임의로 정한 plot 화살표도 관측된 수송이 아니다.

RP¹에서는 doubled angle z=(d₀+i d₁)²의 loop winding이 정수다. director charge는 winding/2다. RP², RP³에서는 π₁=Z₂이며 인접 eigenvector를 가까운 부호로 lift하고 폐곡선의 부호 변화로 trivial/nontrivial을 구분한다. 코드의 product sign(dᵢ·dᵢ₊₁)은 각 eigenvector 부호 선택에 불변이다. gap/trace 부족이나 인접 line이 거의 수직이면 판정을 유보한다. RP¹의 정수 charge와 RP²의 Z₂ class는 같은 양이 아니다.

smooth globally defined n의 점별 nnᵀ는 이미 연속 lift n을 가지므로 contractible 공간의 국소 loop에서 nontrivial Z₂를 임의로 가정하면 안 된다. 반면 fiber 평균의 leading eigenline은 n 그 자체가 아니며 eigenvalue degeneracy를 가질 수 있다. 이때 projected tensor의 eigenline 결함은 발생할 수 있지만 원래 spin amplitude의 singularity나 S³ degree와 같은 결함은 아니다. bilinear discrete tensor의 loop에서 확인한 class도 연속 원래 형상의 GT 결함과 구분해야 한다.

S³ degree는 compactified 3차원 domain→S³ map의 π₃(S³)=Z 불변량이다. RP 투영의 covering map은 k≥2 homotopy를 단순히 전부 없애는 연산이 아니다. 그러나 **부분공간 평균 및 차원 감소**는 보통 역변환할 수 없으므로 S³ degree를 2D eigenline loop 수로 환산할 수 없다. binary Euler characteristic과 component 수 역시 별개의 정의다.

본 연구의 degree는 음의 외곽 collar를 south pole로 cap한 normalized Freudenthal interpolation에서 정의한다. target count 일치 외에 interpolation norm의 양의 수치 하한을 확인했다. 이는 해당 endpoint 보간의 감사이며 모든 시간 step 사이의 homotopy 보존 또는 interval-certified theorem은 아니다.

## 7. 정보 보존의 한계

모든 생성과 관측은 고정 binary raster R의 함수다. R이 같으면 deterministic n와 모든 readout도 같다. 따라서 이 프레임워크는 입력의 정보를 재표현·정규화하지만, 잃어버린 연속 경계를 일반적으로 유일하게 복구하지 못한다. 동일 R에서 다른 둘레/면적/축을 갖는 매끈한 예와 오차 하한은 [PIXEL_INFORMATION_LIMIT.md](PIXEL_INFORMATION_LIMIT.md)에 제시한다.
