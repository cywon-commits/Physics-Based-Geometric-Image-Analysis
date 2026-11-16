# 연구계획서: Physics-Based Geometric Image Analysis (PBGIA)

**부제:** 이산 영상의 물리 기반 이완을 통한 연속 경계 길이·방향·표면적·위상 정보 복원  
**버전:** 2026-11-16, 연구 실행 계획 v1.0  
**위치:** 방법론 및 전산 검증 연구; 아직 확립된 신기술 또는 통계적 우월성 입증으로 간주하지 않음.

## 1. 연구 목적과 중심 가설

일반적인 픽셀 및 복셀 영상의 이진화는 연속 경계를 계단형으로 왜곡한다. 본 연구는 영상의 영역 값을 **고정 외부장**으로 해석하고 단위 길이 제약을 갖는 물리적 질서변수장을 이완시켜 유한 폭의 계면을 생성한다. 단일 기준 원/구에서 보정한 에너지 비례계수로 경계 길이/표면적을 추정하고, 구배 텐서의 고유성분으로 주축·비등방성을, 고차 위상 적분으로 위상 정보를 구한다.

**가설 H1 (길이·면적):** 한 가지 기준 도형으로 보정한 에너지가 독립적인 도형의 둘레/표면적에 비례하고, 회전 및 서브픽셀·서브복셀 이동에 대한 변동이 기존 대표적인 방법보다 경쟁력 있다.

**가설 H2 (방향):** 이완장의 국소 구배 텐서는 원래 도형의 경계 방향 정보를 유지하여, 특히 복잡한 경계·잡음·가까운 구조에서 구조 텐서 또는 PCA와 상보적이다.

**가설 H3 (통합성):** 동일한 이완장으로 길이/표면적·방향·국소 텐서·위상적 지표를 함께 얻을 수 있으며, 최종 정확도 대비 계산비용을 정량화할 수 있다.

**가설 H4 (차원 및 영상 채널 확장):** 2D→3D, binary→grayscale/RGB 단계에서 동일한 구조를 유지한다. RGB 확장은 level-set/채널 투영의 중복 계수와 색공간에 주의한다.

본 연구의 목표는 임의의 측정량에서 수치오차 0을 달성하거나 위상 정수성 자체를 새로 증명하는 데 있지 않다. **기하학적 추정값의 교정 후 일반화** 및 복수 측정량의 통합성이 핵심이다.

## 2. 모델과 관측량

### 2.1 2차원 binary 모델

영상 마스크 `f(x,y) ∈ {−1,+1}`와 `m(x,y) ∈ S²`에 대해

\[
 E_2[m;f] = A \sum_{\langle i,j\rangle} w_{ij}\|m_i-m_j\|^2 + h\sum_i(1-f_i m_{z,i}) + E_{\rm SB}.
\]

초기 `m`은 **오직 이진 마스크**의 거리변환으로 생성한다. 형태를 정의한 해석적 signed distance 또는 실제 매개변수를 솔버에 넘기지 않는다. 소량의 interfacial DMI 또는 영상 경계 법선 기반 field로 면내 회전 방향을 정렬한다. 모든 spin은 각 단계에서 정규화한다.

비교 커널은 (a) 4-neighbor, (b) 9점 등방, (c) 적은 자유도에서 조정한 5×5 대칭 커널의 세 가지로 제한한다.

측정량:

\[
 \widehat L(f)=L_0\frac{E_2^*(f)}{E_2^*(f_{\rm circle,ref})},\qquad
 T_{ab}=\sum_i D_a m_i\cdot D_b m_i,\quad a,b\in\{x,y\}.
\]

텐서의 traceless 부분으로 주축 방향과 비등방성 `η=(λ₊−λ₋)/(λ₊+λ₋)`을 구한다. 주축 각도는 방향축이므로 180° 주기 오차를 사용한다. 정수 스커미온 수는 미분 근사가 아닌 삼각형 solid angle 합으로 측정한다.

### 2.2 3차원 binary 모델

`f(x,y,z) ∈ {−1,+1}`, 네 성분 `n=(v_x,v_y,v_z,s)∈S³`에 대해

\[
 E_3[n;f]=A\sum_{\langle i,j\rangle}w_{ij}\|n_i-n_j\|^2 + h\sum_i(1-f_i s_i)-\epsilon\sum_i v_i\cdot (\nabla_h f)_i.
\]

여기서 4번째 `s`는 2D의 `m_z`와 유사한 **질서변수 성분**이지 시간축이 아니다. `v∝∇f` 항은 경계에서 회전 방향을 선택하는 고정장 결합이며, 실제 DMI와 동일하다고 주장하지 않는다. 임의 방향의 3차원 법선을 표현하기 위해 `v_x,v_y,v_z` 세 성분을 사용한다.

\[
 \widehat S(f)=S_0\frac{E_3^*(f)}{E_3^*(f_{\rm sphere,ref})},\qquad
 T_{ab}=\sum_i D_a n_i\cdot D_b n_i,\quad a,b\in\{x,y,z\}.
\]

위상적 제약의 검사에는 `π₃(S³)=Z`에 대응하는 degree를 사용하되, 유한차분 `Q₃`가 정확한 정수라는 주장은 하지 않는다. 위상 섹터 사이의 전이·collapse를 별도 모니터링한다.

### 2.3 일반 grayscale/RGB 영상

스칼라 intensity `I`의 level set `f_t=sign(I−t)`를 분석한다. `∫ Per({I>t})dt=∫|∇I|`라는 coarea 관계는 기존 수학적 정리이며 새 이론적 발견이 아니다. RGB에서는 (i) 채널별 cutoff, (ii) 컬러공간의 투영 `u·I`에 대한 threshold를 비교한다. 4×4×4 partition은 64개 **서로 배타적인 클래스 마스크**를 주며 64개 독립 threshold 채널과 구별한다. 경계 중복 계수를 보정한다. 필요 시 OKLab/CIELAB 등의 색 공간을 비교한다.

## 3. 실험설계: 엄격한 데이터 분리와 공정한 비교

| 단계 | 합성 데이터 | 주 측정 | 기준법 |
|---|---|---|---|
| A. 2D 이진 | 원, 스태디움, 타원, 다각형, 링, 복수 도형 | 둘레, 회전 변동, 방향 | Crofton, marching squares, PCA, structure tensor |
| B. 2D 명암 | soft edge, threshold 8~32개, RGB 투영 | level-set 누적 둘레 및 텐서 | TV, vectorial TV, Gaussian derivatives |
| C. 3D 이진 | 구, 캡슐, 타원체, 토러스, 쉘, 복수 구 | 면적, 법선/축, 수렴 | marching cubes, 노출 voxel face, surface estimators |
| D. 견고성 | 잡음, 잘림, 얇은 통로, 좁은 간격 | 실패율 및 정보 보존 | 같은 입력/보정 조건에서 비교 |
| E. 실제 데이터 | 공개 또는 사용 허가된 segmentation image/volume | 정량화 가능 측정 + 전문가 표시 | 동일한 자료의 표준법 |

**보정 및 교차검증**: 기준 원/구의 한 크기·한 위치에서 환산계수를 먼저 정하고 잠근다. 같은 도형의 회전 시험은 회전 등방성 평가이지만 *형태 일반화* 시험과 구분한다. 다른 크기·형태·위상 클래스·영상 획득 조건을 전부 독립 테스트로 둔다. 커널의 자유도가 학습되면 training에서만 맞추고 validation으로 선택한 후, test는 최종 1회 평가한다.

**평가지표**: 길이/면적 `|prediction/ground_truth−1|`의 MAE·RMSE·95백분위·최대값, 같은 원형 도형의 중심·회전 변동계수 CV, 주축 각도 오차(2D modulo 180°, 3D `acos |u·u_true|`), 결함 전하의 성공률, 이완 잔차·에너지 변화, 벽 collapse 비율, 데이터 1개당 계산시간/메모리. 평균뿐 아니라 **표본 수·부트스트랩 신뢰구간**을 보고한다.

**필수 정보 누출 방지**: 합성 도형의 해석 signed distance를 자화 초기화나 스무딩에 쓰지 않는다. `f` 이외 정보를 사용할 수 있는 것은 ground-truth 계산뿐이다. marching cubes와 보간법의 교정 여부를 자화모델과 동일하게 설정한다.

## 4. 과제별 실행 계획

- **WP1 — 기준선 고정:** 현행 2D·3D 코드 통합, sign convention/격자 간격/경계조건 기록, 이완 잔차와 에너지 변화 기반 수렴 진단; binary-only initialization 보장.
- **WP2 — 2D 표준 도형:** 원 반지름 8–40 px, 스태디움·타원·링, 독립적인 난수 회전/중심 ±0.5 px. 약 300개 이상(계산 여건에 따라 단계적 증설). 보정한 둘레·축 방향·위상 지표와 baseline 비교.
- **WP3 — 3D:** 구 반지름 7–20 vox, 캡슐·타원체·토러스·얇은 쉘. 최소 100개 독립 표본을 목표로 하되, 반드시 이완 수렴 문제부터 해소. 서브복셀/회전 통계, voxel spacing 이방성 검사.
- **WP4 — multilevel & RGB:** soft-edge로 ground truth가 알려진 영상에서 threshold-integrated length, color projections 및 structure tensors를 확인. 실제 RGB/volume 1–3종에 적용.
- **WP5 — 물리 모델·커널 ablation:** kernel 4/9/5×5, DMI vs `∇f` field, 벽 두께 `sqrt(2A/h)`, 이완 허용오차, 노이즈, 계산 시간의 trade-off를 보조적으로 평가.
- **WP6 — 문서화 및 논문화:** 공개 재현 코드, seed, 전체 CSV, 수렴 그래프, 실패 사례까지 포함. 논문의 novelty는 유한 폭 물리적 representation의 **통합 측정 및 교정 후 일반화**로 정당화한다.

## 5. 진행·중단 기준

**Go**: 하나의 reference calibration을 다른 크기/방향/형태에 그대로 적용했을 때 여러 대표 데이터셋에서 baseline 대비 유의한 장점 또는 상보적인 측정 기능을 보이고, 수렴 여부가 충분히 기록되어 있는 경우.

**Revise**: 특정 형태에만 개선되거나, 커널 최적화가 test에서 일반화되지 않는 경우. 가장 단순하고 안정적인 9점 커널로 환원한다.

**Stop/limit**: 얇은 구조·토러스에서 비정상적인 topology change/relaxation collapse가 일어나 면적 추정이 수렴하지 않으면 해당 영역에서는 실패를 공개하고, topology-preserving constraint 또는 finite-time measurement를 별도 가설로 취급한다.

## 6. 선행연구와 신규성 점검

이미 Modica–Mortola 계면 에너지의 perimeter 근사, phase-field segmentation, vectorial TV 및 structure-tensor 기반 기하학 분석이 연구되어 있다. 그러므로 '물리적 장의 최소화가 경계를 측정한다'는 명제만으로 신규성을 주장하면 안 된다. 본 연구의 구별점은 `S²/S³` 회전 자유도와 symmetry-breaking boundary anchoring, **동일한 장으로부터 에너지·방향·topology를 함께 측정**, binary-only voxel에서의 공정한 benchmark, single-reference calibration, 2D/3D/RGB에 걸친 검증 가능성이다. 실제 일반화가 확인되지 않으면 이 제안도 exploratory methodology이다.

참고 문헌 및 링크:

1. Dondl, Novaga, Wirth & Wojtowytsch, *A Phase-field Approximation of the Perimeter under a Connectedness Constraint*, SIAM J. Math. Anal. 51 (2019). https://doi.org/10.1137/18M1225197
2. Jung, Kang & Shen, *Multiphase Image Segmentation via Modica–Mortola Phase Transition*, SIAM J. Appl. Math. (2007). https://doi.org/10.1137/060662708
3. He, Huska & Liu, *Vectorial Total Symmetric Variation and Applications to Color Image Decomposition*, J. Math. Imaging Vis. (2026). https://doi.org/10.1007/s10851-026-01312-x
4. Survey, *The phase field method for geometric moving interfaces and their numerical approximations*. https://www.sciencedirect.com/science/article/pii/S1570865919300043

## 7. 즉시 산출물과 후속 작업

본 배포본에 동봉된 `PBGIA_execution_report_KO.md`는 기존 결과를 재검토하고 수행한 binary-only 3D extended relaxation 및 2D held-out rotation/multilevel 분석을 기술한다. 후속 계산에는 이미지·마스크·초기화·최종장·에너지·잔차를 case 단위로 저장하는 것을 필수화한다.
