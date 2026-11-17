# 누적 검토 결과를 반영한 통합 교정표

## 반영된 도판 수정

| 지적 | 변경 | 확인 자료 |
|---|---|---|
| Fig.1 타원의 x/y 두께 차이 | 원형 도메인의 signed-distance 자구벽으로 교체, 일정 국소 법선 거리를 도식화 | `Fig01_conceptual_pipeline.pdf` |
| Fig.3 타원 둘레가 간단한 해석식이 아님 | 회전 stadium, `P=2πR+2L`, 실제 N=30 텍스처+θ–θ+size fit | `stadium_orientation.csv`, `stadium_scaling.csv` |
| 회전 사진의 모서리 artifact | 원본·회전·director·교정 모두 공통 유효 258×258 중심영역으로 crop | `Fig09_multithreshold_deskew.pdf`, `brick_valid_crop.csv` |
| Fig.7이 분석적 그림 | 실제 N=30 입력 생성장 Q₂ 적분(1,2,0,1) | `Fig07_2D_generated_Q2.pdf`, `2d_spin_topology.csv` |
| 3D S³를 최대한 시각화하고 degree를 제시 | 3D 실제 생성된 s 단면, preimage integer degree, 4성분 표현으로부터 mesh 및 transverse v 화살표 | `Fig07_3D_generated_Q3.pdf`, `FigS_3D_volume_S3_visualization.pdf`, `3d_synthetic_validation.csv` |
| 3D 표면적/방향 응용 | 구 참값 1개로 보정; 다른 4개 분석형 도형에서 면적/주축 검증 | `3d_synthetic_validation.csv` |
| 회전 영상의 정확도 오해 | 절대 방향 정확도가 아닌 회전 등변성으로 명시 | `main.tex` |
| 한계 문구 반복 | 필수적인 계산 범위만 Methods와 짧은 본문에 배치, 장황한 별도 한계 표 제거 | `main.tex` |

## 새 결과의 과학적 해석

- 2D degree: 단위화된 생성장으로 계산한 구면 삼각형의 연속 근사 실험. 입력 Euler characteristic과 맞은 네 형태가 존재한다. **별개의 연결 요소 수와 Euler 수는 고리 형태에서 구별**된다.
- 3D regular-value degree: 극값이 아닌 일반 target preimage를 사면체 보간장으로 세므로 정수임. 세 generic target에서 동일한 값이 나옴. 원시 finite-difference determinant 적분은 얇은 격자 자구벽에서 1에 크게 못 미칠 수 있으므로 본문의 integral accuracy와 혼동하지 않음.
- 3D 면적: 보고된 PBGIA readout은 이번 파일럿의 중앙차분 모델이며 동결된 최적의 geometry observer와 다른 값임. Sphere reference를 사용한 단일 기하 보정만 적용했고 각 테스트 도형에는 재보정하지 않음.
- 3D 섬유: 볼륨 입력은 물리적으로 가능한 **합성 캡슐 배열**; 실제 CT 섬유 표본의 ground truth는 아님. 최소 tensor 고유축이 장축임을 확인.
- 나노입자와 건물 수: 사용자가 제공한 출처 미확인 나노입자 사진은 논문에 포함하지 않았음. `unvalidated_image_counts.csv`은 watershed 탐색 결과만 기술; 이미지별 분할 주석이 없고 재사용 허락도 정리되지 않아 투고 근거로 사용 불가.

## 실제 3D 영상 관련

실제 3D 데이터 후보는 PoreML-Geo(https://huggingface.co/datasets/PoreML/PoreML_geo) 및 공개 microscopy cells3d 등이 있으나, 이 작업 환경에서는 원본 바이너리 다운로드에 실패했습니다. **실제 CT를 사용했다고 원고에 기록하지 않았습니다.** 실제 3D 응용이 투고에 필수라면 공개·허가된 TIFF stack 또는 binary voxel array를 확보한 뒤 동일 코드로 재계산해야 합니다.

## 추가 확인 필요

- 저자 및 연구비/기여·이해상충 항목은 미확정.
- 본 원고는 편집·심사를 위한 중립적 LaTeX 포맷으로 최종 저널 전용 템플릿이 아님.
- 새 파일럿 코드를 동결 저장소와 연결하는 별도 커밋/DOI는 아직 발행하지 않았음.
- 심사자 독립 R5 및 출처·서지의 전수 확인은 별도 단계.
