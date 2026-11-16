# 스모크 수치 실행 결과 — 제한적 예비 결과

2D/3D 작은 합성 stadium/ellipse/capsule/ellipsoid 비교. **미학습 커널**, **학습/검증/시험 분리 없음**. 결과는 후보 선정이나 일반화의 근거로 사용하지 않는다.

| 차원 | 10회 예비 측정에서 낮은 MAE 후보 | MAE |
|---|---|---:|
| 2D | exchange=free, observer=bond_min | 0.760° |
| 2D | exchange=free, observer=direct_min | 0.868° |
| 2D | exchange=axial, observer=bond_min | 1.683° |
| 2D | exchange=tied, observer=bond_min | 1.691° |
| 3D | exchange=tied, observer=gradient | 1.551° |
| 3D | exchange=axial, observer=gradient | 1.554° |
| 3D | exchange=axial, observer=bond_min | 1.809° |
| 3D | exchange=tied, observer=bond_min | 1.825° |

공정한 실험에서는 현재 GitHub의 frozen PBGIA-30을 새 데이터에서도 반드시 재실행하고, 그와 별개로 이전 코드의 생성 구조 + 네마틱 텐서의 효과를 요인별 분리해야 한다.

기본 smoke 행 수: **720**. 각 영상/필드에서 여러 observer를 평가하므로 행은 독립 사례가 아니다.
