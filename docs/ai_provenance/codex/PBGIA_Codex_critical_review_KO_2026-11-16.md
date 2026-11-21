# PBGIA Codex 실행 보고서: 비판적 검토 노트

**출처**: `PBGIA_Dynamic_Normal_Research_Report_2026-11-16.pdf`, 사용자가 제공한 Google Drive 링크 (2026-11-16). 본 노트는 PDF의 추출 텍스트 및 이전 기준 문서 v1.1을 대조한 평가다. 원시 필드·전체 CSV·로그의 독립 재실행은 하지 않았다.

## 1. 재현성·모델 충실도

- 기준 모델 `Hv = J Σnn v + D ∇s`, `Hs=J Σnn s+hf`, 매 반복 `n←H/|H|`이며 `div(v)`가 없다. `α=1`이 기본이다. `J Σnn`은 라플라시안이 아니다.
- EDT 초기화만 사용하며 정확한 연속 법선은 평가에만 사용한다. 이는 정답 누출을 방지하는 설계다.
- **중요한 잔여 차이**: 구 Colab 코드의 positive convolution exchange 커널과 새 보고서의 axial-neighbor exchange는 동일하지 않다. 구현을 옮길 때 생성커널을 명시해야 한다.
- 48³의 29개 P0 종점이 RMS<2e-6에 도달, 주 도형은 2,000회에서 roundoff 수준. 고정점 잔차뿐 아니라 최대 국소 잔차와 2주기 오인을 확인한 것은 적절하다. 다만 전체 파라미터 영역의 수렴 정리는 아니다.

## 2. 지형 보존과 정수 위상

- 토러스, 구, 캡슐의 주 조건은 Dice=1, χ=(0,1,1); degree=(0,1,1)이다. 토러스는 2,000회 이후에도 유지된다.
- Euler 수는 재구성된 바이너리 영역의 디지털 불변량; degree는 **compactification한** S³ 사상의 정수. 서로 다른 주장이다.
- 바깥쪽 음의-s collar 3 voxel을 남극으로 연결하고, 사면체의 hemisphere/convex hull 검사로 0 통과 가능성을 조사했다는 절차는 단순 유한차분값보다 강하다. 다만 다른 강한/약한 필드 조건에서의 보간 퇴화와 경계 여유를 계속 경고해야 한다.

## 3. 표면 법선과 자구벽 폭

- 이론적 **안쪽** 법선에 대한 signed cosine ~0.995–0.997, `v` 대 `∇s` ~0.987–0.991로 정렬이 확인된다.
- wall-profile proxy ~1.27–1.28 voxel, ray [-0.7,+0.7] 전이폭 ~1.28–1.39 voxel. 서로 다른 정의이며 §0의 해석 λ와 같지 않다.
- `h=8,D=.16`에서는 벽 voxel 집합이 비므로 Dice가 높아도 올바르게 해상된 벽이라고 결론 낼 수 없다.

## 4. 측정 지표·학습 분리

- 관측은 `tr(T)`의 reference-calibrated 양이며 비변분 모델의 실제 최소화 에너지가 아니다. 이 구분은 필수.
- 2D 학습 5×5 MARE=0.29987% (clean n=15), Scharr 0.36968%, Gaussian σ=1 gradient-integral 0.46411%, Crofton 1.0807%.
- 3D 최선 PBGIA fourth 1.3872%(n=18), reference-calibrated marching cubes 1.3798%. 아주 작은 차이로 우열 선언 불가.
- 잡음 0.2% 시험에서는 PBGIA가 좋았지만 단일 noise rate와 제한된 clean margin 6 voxel. 일반적 잡음 우월성 주장은 금물.
- 분리된 원/구의 중심간 장축은 전체 구배 텐서로 잘 추정되지 않는다. local boundary orientation과 global component position은 서로 다른 정보다.

## 5. 개선 제안 및 우선순위

1. 제공된 `REPRODUCE.md`와 `reproduce.py`가 실제 환경에서 동일 CSV를 생산하는지 독립 재실행하고 실행 환경·해시를 보관한다.
2. geometry benchmark의 fixture별 CSV를 공개해 2D/3D 성능에 어떤 도형이 기여했는지, 같은 geometry에 반복된 clean mask를 어떻게 가중했는지 분리한다.
3. kernel의 생성 `G_dyn`과 관측 `G_obs`에 대한 독립 reference calibration을 실시하고 교환 커널 변경의 영향을 재시험한다.
4. grid-refinement `(a,R/a)`를 함께 조절해 continuum-scale 오류와 특정 48³/56³ 해상도 성능을 분리한다.
5. 잡음 비율·유형, 열린 경계 오차, 토러스 minor radius, 마스크 손상, 자연영상에서도 같은 benchmark를 적용한다.
6. 2D Gaussian baseline은 threshold+marching/PCA뿐 아니라 structure-tensor, shape PCA, Euler 분석을 포함한 pipeline 단위 비교로 확장한다.

## 6. 문서에 반영해야 할 결론

**강한 수치적 근거**: 합의된 동적 `∇s` 정규화 모델은 선택된 2D/3D 합성 도형에서 수렴하며, 토러스의 이진 구멍과 정수 degree를 장시간 유지한다. **제한적 개선**: 선택된 데이터에서 2D 둘레와 소량 잡음은 유망. **부재한 근거**: 전 영역에서 커널 최적성, Gaussian/marching cubes/PCA 대비 보편적 우수성, 실제 영상 일반화. 기존 step-field 해석해는 이론적 동기이며 정규화 반복의 해석해가 아니다.
