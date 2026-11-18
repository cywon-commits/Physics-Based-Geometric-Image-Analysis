# PBGIA 통합 수정 원고 — 2026-11-17

## 제공 파일

- `main.tex`, `refs.bib`, `main.pdf`: 최신 영문 원고 및 24쪽 미리보기
- `figures/Fig01_conceptual_pipeline.pdf`: 타원 대신 원형 도메인; signed-distance 기반 일정 법선 두께
- `figures/Fig03_stadium_quantitative.pdf`: 실제 생성된 stadium + θ–θ + 둘레 scaling
- `figures/Fig09_multithreshold_deskew.pdf`: 공통 유효 ROI만 표시한 건축 texture 회전 실험
- `figures/Fig07_2D_generated_Q2.pdf`: 실제 N=30 생성장 2D 위상수
- `figures/Fig07_3D_generated_Q3.pdf`: 실제 N=30 3D S³ degree 및 fiber 방향
- `figures/FigS_3D_volume_S3_visualization.pdf`: 3D 표면 mesh와 v 방향의 3차원 시각화
- `data/stadium_orientation.csv`, `data/stadium_scaling.csv`, `data/2d_spin_topology.csv`, `data/3d_synthetic_validation.csv`, `data/3d_target_stability.csv`: 추가 수치 계산
- `data/unvalidated_image_counts.csv`: 나노입자/벽돌 이미지 탐색적 분할 개수; 논문 본문에는 사용하지 않음
- `code/new_quantitative_studies.py`, `code/pbgia_s3_integer_degree.py`, `code/final_visual_corrections.py`, `code/verify_target_degree.py`: 수치 재현 코드
- `REVISION_AND_EVIDENCE_KO.md`: 데이터 근거, 수정, 향후 작업

## 핵심 수치 (추가 실험)

- 2D stadium P=2πR+2L, L=2R. R=17에서 12개 회전, 주축 MAAE 0.2141°.
- TV 계수 R=9, θ=0 단일 참조로 결정; 나머지 8개 크기×3회전(24포즈)에서 MARE 0.9314%. 자유 직선 fit y=1.0087x+0.105.
- 2D 실제 생성장: disk/two disks/annulus/disk+annulus에서 Q₂≈1,2,0,1.
- 3D 구/캡슐/토러스/두 구/네 섬유의 compactified regular-value S³ degree: 1,1,0,2,4 (각각 서로 다른 세 일반 타깃에서 재현).
- 3D 구로 보정한 중앙차분 면적 관측: 캡슐 +1.146%, 토러스 −1.337%, 구 2개 +0.744%, 네 섬유 −2.452%.
- 네 정렬 섬유의 법선 tensor 최소 고유축 오차 3.758°.
- 나노입자 사진의 추정 개수는 검증 전 탐색값이므로 과학적 ground-truth 계수로 인용 금지.

## 본문에서 유지한 구분

- Figs. 1–2: 개념/연속체 이론. 신규 Fig. 3/7/8: 실제 생성한 이산 스핀장.
- 원래 frozen benchmark(저장소 커밋 `b2107084d19fcdf54c9350cf3f0e013fed8afa4c`)는 그대로 둠. 이번 새 그림 실험은 별도 소규모 파일럿.
- 원고에 포함한 3D 볼륨은 **합성 생성 볼륨**임. 실제 micro-CT 원본 다운로드를 확보하지 못했으므로 실제 재료 관측치라고 주장하지 않음.
- CC0 벽돌 그림의 결과는 rotation equivariance이며 원본 사진의 절대 방향 정확도가 아님.

## 빌드

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

추가 실험을 복제할 때에는 `numpy`, `scipy`, `scikit-image`, `matplotlib` 등이 필요합니다. 3D regular-value calculation과 선형대수 연산은 머신에 따라 수분 걸릴 수 있습니다. 파일에 포함된 원래 source image 목록과 출처를 확인한 뒤 사용하십시오.
