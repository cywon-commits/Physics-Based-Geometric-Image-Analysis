# 최종 결론과 연구 종료 판단

## 검증한 내용

1. 고정 Z₂ 입력과 current grad(s), 양의 NN 교환합, 정규화 동기식 갱신은 2D/3D의 유한한 주요 형상군에서 안정적인 S²/S³ 경계 texture를 생성했다. 옛 N30 2D/3D 사례도 저장된 spin과 비트 단위로 재현했다.
2. 같은 field에서 경계 크기, 국소 법선, normal distribution tensor, 위치 covariance, 조건부 사영 eigenline을 실제 계산했다. clean 국소 법선 cosine 평균은 0.99378/0.99667이었다.
3. 제한된 validation grid에서 α=0(2D), α=.5(3D)를 선택했다. 새 clean test MARE는 0.893%/0.732%지만, 같은 기준자로 보정한 전통 방법의 0.738%/0.563%보다 나빴다.
4. full projector와 vector tensor의 fiber 평균은 일반적으로 혼합 PSD tensor였다. 단순 최대 고유값과 충분한 signal이 있는 부분에서만 RP eigenline을 정의할 수 있었다. uniform full 평균은 scalar pole에 지배되어 공간 방향 해석에 거의 도움이 되지 않는 경우도 있었다.
5. 합성 RP¹/RP² 판정·유보 대조 10개가 맞았고, 실제 투영 map에서는 bilinear tensor loop의 비자명한 후보들을 검출했다. 이는 원래 3D 자기적 singularity의 검출과 다르다.
6. cap과 nonvanishing audit를 적용한 생성 sphere/capsule/torus/pair의 S³ degree는 1/1/0/2였다. binary component와 구별했다. 이 유한 endpoint 결과를 일반적인 Euler 대응 또는 시점 사이 topology 보존 정리로 부르지 않는다.
7. 동일 raster인데 참 둘레/표면적 또는 축이 다른 매끈한 연속 형상 쌍을 실제 구성했다. 따라서 피할 수 없는 측정 불확실성이 있다. 동시에 얇은 positive phase가 생성장에서 사라지는 추가 손실도 확인했다.

## 부정적 결과와 이론적 한계

기본 normalized alignment에는 reciprocal scalar 결합이 없어 단일 Hamiltonian의 gradient descent가 아니다. 완전한 병렬 정렬의 명시적 에너지 증가 반례도 있다. 물리적 Hamiltonian이 더 낮은 비교군이 기하 정확도에서 더 낫지는 않았다.

법선 분포와 공간적 장축·중심 연결축은 다르다. projected tensor의 leading line은 averaged tensor 자체와 같지 않다. director는 signed transport를 정의하지 않는다. 원·구 및 고유값 퇴화 영역에는 임의 정답 방향을 만들 수 없다.

0.75 voxel 3D shell의 669개 positive 입력 voxel이 N30에서 모두 사라졌다. grid 방향에 따라 최종 벽 폭도 달랐다. 보편적인 shape/topology 보존과 subpixel 완전 복원을 지지하지 않는다.

최종 readout 조합의 형상별 이득은 일관되지 않았고 n=6 prototype bootstrap 구간은 두 차원 모두 0을 포함한다. 자유 생성 kernel 학습의 과거 이득도 reference/noise 안정성과 일치하지 않았다. 속도는 현 CPU 구현에서 전통적 기하 방법보다 상당히 느리다.

## 아직 가설이거나 미검증인 내용

자연 영상으로의 일반화, 비등방 voxel, 다른 잡음, 작은 구조에서의 일반적 안정성, 최적 field/observer의 존재는 검증하지 않았다. 실제 자기적 결함의 의미, projected defect의 연속 GT 대비 FP/FN, spin degree와 영상 topology의 일반적 대응, 전체 궤적의 homotopy 보존도 확립하지 않았다. 새로운 커널 학습 및 moment 제약군은 이번에 `NOT_TESTED`다.

## 종료할 것인가

**현재 연구는 정리하고 종료하는 편이 합리적이다. 새로운 대규모 최적화를 계속하지 않는다.**

그 이유는 핵심 표현 가설에 대한 제한된 성공 사례와 명확한 실패 사례가 모두 확보됐고, 미세한 평균 오차 감소가 공통장의 새로운 기능을 보장하지 않기 때문이다. N30 기준 field, frozen observer와 validation 선택 readout, 실패 조건 및 재현 자료를 최종 산출물로 남긴다. test를 보고 α를 바꾸거나 더 나은 숫자가 나올 때까지 prototype/커널을 추가하지 않는다.

향후 별도 연구가 필요하다면 특정 응용의 구조 크기·잡음·비등방성·연산시간 또는 실제 물리적 에너지 일관성이 요구될 때로 제한해야 한다. 이는 이번 연구를 연장할 실행 계획이 아니다. 학문적 메시지는 ‘만능 영상 알고리즘’보다 **하나의 spin texture가 제공하는 여러 관측량과, 사영·픽셀화·비변분 생성에서의 식별 한계**에 있다.
