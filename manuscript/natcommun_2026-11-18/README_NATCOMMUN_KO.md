# PBGIA — Nature Communications 투고 초안 (2026-11-18)

`manuscript/integrated_2026-11-17` 이후의 수정 전체를 담은 폴더입니다.

## 주요 파일
| 경로 | 내용 |
|---|---|
| `nc/nc_main.tex`, `nc/nc_main.pdf` | Nature Communications 형식 본문 초안 (그림 8, 표 1). 저자가 채울 부분은 노란 상자와 빨간 ▶[AUTHOR: …] 표시 |
| `nc/nc_si.tex`, `nc/nc_si.pdf` | Supplementary Information (Supplementary Notes 1–7) |
| `nc/NC_AUTHOR_CHECKLIST_KO.md` | 투고 전 저자 확인 목록 |
| `main.tex`, `main.pdf` | 통합 원고(rev12, 긴 형식) |
| `figures/` | 모든 그림 (PDF/PNG). `nc/figures`는 이 폴더를 가리키는 심볼릭 링크 |
| `figures_src/` | 그림 생성 스크립트 |
| `code/`, `code/revision2/` | 분석 스크립트 (t30–t60) |
| `data/revision2/` | 수치 결과 (CSV/JSON/NPZ) |
| `NUMBERS_MAP_KO.md` | 원고의 모든 수치 ↔ 데이터 파일 ↔ 스크립트 대조표 (rev8–rev14) |

## 이번 단계에서 추가·변경된 내용
- **그림 1**: (a) 이진 영상 → 장 → 스핀 텍스처 → 판독의 절차도(원근 3D). (b–e) 판독 확대: 꽃 모양 둘레, 방향자, Y자 결함(±½), 영역별 차수.
- **그림 7 (위상)**: 2D는 기울어진 판 위의 3D 원뿔 스핀, 3D는 등위면으로 다시 그림.
- **표 1**: 열별 최소 오차 굵게 표시. "no calibration"은 "모양별 보정 없음(평면 벽·원판·구에서 한 번 정한 상수)"으로 표현을 바꿈.
- **업데이트 횟수 N**: 수렴률 λ = zJ/(zJ+h). 2D·3D 벤치마크, 벽 법선 각도, 잡음 제거, 실행 시간을 N에 따라 정리 (SI Fig. `FigA7_nsteps`; `code/revision2/t55*`–`t57*`, `t56b*`).
- **벽 법선 오차**: 픽셀 단위 오차(직선 벽 3.7°)는 래스터의 한계임을 확인. 체계적 편향은 δφ = Σ a_j sin 4jφ (a₁ = 2.39°; 예측식 (k²/24) sin 4φ) (`t58*`, `t59*`).
- **세포 핵 영상(본문 그림 8)**: BBBC038 / StarDist 하위집합으로 수를 세어 비교 (`t60*`, `figures_src/fig_cells.py`).
- **자성 영상**: 본문은 한 문단으로 줄이고, 그림과 방법은 SI로 이동.

## 외부 데이터
- **세포 영상**: 저장소에 포함하지 않음. 실행하려면 받아서 `/home/claude/celldata/raw/dsb2018` 위치에 풀어 두면 됨.
  - 내려받기: https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
  - BBBC038v1, CC0. 인용: Caicedo et al., *Nat. Methods* 16, 1247 (2019); Schmidt et al., MICCAI 2018.
- **Lorentz 현미경 영상**(SkyrmNet 데모, MIT)과 **KerrPy 시퀀스**(GPL-3.0): 원본은 포함하지 않음. 이들로부터 계산한 큰 중간 배열 `t51_ltem.npz`, `t52_kerr_fields.npz`(각 50 MB)도 제외했으며, `t51_ltem.py`, `t52_kerr.py`로 다시 만들 수 있음.
- **미로 무늬 영상**(`data/revision2/magnetic/labyrinth/`): Zhao et al. 데이터셋의 일부, CC-BY 4.0.

## 재현 시 유의
- 스크립트에는 작업 환경의 절대 경로(`/home/claude/work`, `/home/claude/celldata`)가 들어 있음. 다른 환경에서는 각 스크립트의 `sys.path`와 데이터 경로를 바꿔야 함.
- 원고 빌드는 `nc/`에서 `pdflatex nc_si && bibtex nc_si && pdflatex nc_si && pdflatex nc_main && bibtex nc_main && pdflatex nc_main ×2` 순서로 함. 본문이 `xr-hyper`로 SI의 레이블을 참조하므로 SI를 먼저 컴파일해야 함.
