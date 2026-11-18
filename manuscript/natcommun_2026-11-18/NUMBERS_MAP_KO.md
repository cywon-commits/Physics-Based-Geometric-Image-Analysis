# 수치 대응표 (P2) — 2026-11-18

`data/revision2`의 경로를 기준으로 적었다. 모든 값을 데이터 파일에서 다시 계산해 본문·그림·표·초록과 맞춰 보았다(✓ = 일치).

| 값 | 데이터·스크립트 | 그림 | 표 | 본문·초록 | 일치 |
|---|---|---|---|---|---|
| 벽 폭 collapse 오차 0.3% / 0.9% (ℓ = 7.1a / 3.5a) | t1_wall_collapse.csv (t1_wall.py) | Fig. 2(b) | — | 초록 "1% 이내", §3.1 | ✓ |
| K = 2√2/15 = 0.1886 | 해석식, t1_wall.py | Fig. 2(c) | Table 1 | Eq. (11) | ✓ |
| h/J = 2에서 폭 1.03ℓ(κ = 0.71), 0.71ℓ(κ = 0) | t1_wall_collapse.csv | Fig. 2(b) 화살표 | — | §3.1 | ✓ |
| m̄₂ = 1.0189 (D = 1), 1.0437 (D = 0); m̄₃ = 1.0160 | t2_angle.csv, t5_planes3d.csv, t10_constants.json | Fig. 3(a) 점선, 별 | — | §2.3, §3.2, Discussion | ✓ |
| m(κ) 계수 | t10_constants.json → code/revision2/readout.py | Fig. 3(b,c) 선 | — | Eq. (16) | ✓ |
| ⟨u⟩ < 0.021 px, ⟨u²⟩ = 0.525 | t3_disk.csv, t5_spheres.csv | — | — | §2.3, §3.2 | ✓ |
| 국소 규칙: 엄격 판정 24개 중 23개 일치 | t4_local_rules.csv | Fig. 4(a) | — | §3.3 | ✓ |
| 잡음 0.2% / 1%: 평면 상수 0.45 / 0.43, 곡률 보정 0.40 / 0.58, median + Crofton 1.20 / 1.15, raw 88%, N = 0 148%, 5%에서 10.0 대 3.5 | t4_noise.csv, t12_check.csv | Fig. 4(b) | — | §3.3 | ✓ |
| frozen test MARE (전체 행) | frozen_rescore/weighted_readouts_all.csv + frozen geometry_errors_long.csv (t11_frozen_w.py, t7_summary.py) | Fig. 5 | Table 2 | 초록 0.28 / 0.56 / 0.89 / 0.74 / 0.37 / 0.93, §3.4 | ✓ |
| 쌍대 bootstrap: −0.29 [−0.51, +0.02], −0.46 [−0.77, −0.05], +0.15 [−0.38, +1.06], −0.46 [−0.84, −0.10] | frozen_rescore/paired_vs_conventional.csv | — | — | §3.4 | ✓ |
| train 0.58% / validation 0.46% (3D) | frozen_rescore/curv_readouts_all.csv | — | — | §3.4 | ✓ |
| near-touch 2D 2.9%, 간격 안 s 평균 −0.86, 나머지 다섯 종류 0.48% 대 0.77% | weighted_readouts_all.csv, 진단 스크립트 | Fig. 5(a) | — | §3.4 | ✓ |
| stadium 27개: 0.39% (최대 1.10%), Crofton 1.13% | t13_stadium.csv | Fig. 6(b) | — | §3.4 | ✓ |
| 3D 도형 5종: −0.12 / −0.22 / −0.14 / +0.59 / −0.22% (MARE 0.26); 구 보정 marching cubes 0.15% | t12_check.csv, t6_3dshapes.csv | — | — | §3.4 | ✓ |
| stadium 축 0.214°, 법선 cosine 0.9938 / 0.9967 등 | ../data/stadium_orientation.csv, frozen 보고서 | Fig. 6(a) | Table 3 | §3.5 | ✓ (기존 값) |
| Q₂ = 1, 2, 0, 1; Q₃ = 1, 1, 0, 2, 4; 다공체 −5 | ../data/2d_spin_topology.csv, ../data/3d_synthetic_validation.csv, t6_3dshapes.csv | Fig. 8 | Table 3 | §3.6 | ✓ |
| 다공체: 입력 χ = −4 (6-연결) / −6 (26-연결), 231 voxel (188 제거, 43 추가) | t6b.py 출력 | Fig. 8(h) | — | §3.6 | ✓ |
| deskew 추정 −12.83° | fig_images.py 출력 (12.828) | Fig. 9 | — | 캡션 | ✓ |
| 다중 임계 회전 오차 9종 | ../data/multithreshold_applications.csv | Fig. 10, 11 | Table 4 | §3.5 | ✓ (기존 값) |

## 추가 (2026-11-18, director 위상 전하와 σ 스케일)
| 값 | 데이터·스크립트 | 그림 | 본문 | 일치 |
|---|---|---|---|---|
| 코어 법칙 F(ρ), 격자와의 최대 편차 0.03 | t14_core.csv (t14_core.py) | Fig. 7(h) | Eq. (23), §3.6 | ✓ |
| 소멸 상수 σ_c/d = 0.4413 (구적법), 격자 0.443–0.469 (d = 48–8) | t14_pair.csv, t14_core.py | Fig. 7(c) 점선 | Eq. (24) | ✓ |
| 합성 텍스처: 결함 정확히 4개(PBGIA σ = 6–14, ST σ = 4–16), 가까운 쌍은 16–18 사이 소멸(예측 17.7), 먼 쌍은 40–50 사이 소멸(예측 48.5) | t15_lic_sigma.csv | Fig. 7(a–c) | §3.6 | ✓ |
| grass 결함 밀도 / 무작위장 법칙 = 0.74–1.0 (σ = 2–16); brick·wood는 σ ≥ 12에서 0 | t16_real_sigma.csv | Fig. 7(i) | §3.6 | ✓ |
| strata 국소 방향 오차: PBGIA 2.3° → 0.9° (σ = 2 → 16), ST 최소 0.2° (σ = 4) | t16_real_sigma.csv | — | §3.6 | ✓ |
| 국소 회전 일관성: ST < 1°, PBGIA 1–7° | t16_real_sigma.csv | — | §3.6 | ✓ |


## 새 벤치마크 (2026-11-18, 기존 frozen 재채점을 대체)
기존 frozen 재채점 행(0.28 / 0.56 등)은 원고에서 빠졌다. 해당 데이터는 `data/_archive/frozen_rescore/`로 옮겼다.

| 값 | 데이터·스크립트 | 그림 | 표 | 본문·초록 | 일치 |
|---|---|---|---|---|---|
| 비접촉 6종 평균 MARE (2D: 0.55 / 0.58 / 0.96; 3D: 0.35 / 0.36 / 0.49) 및 비교 방법 | benchmark/bench_summary_family.csv (bench_analyze.py) | Fig. 5(b, d) | Table 2 | 초록, §3.4 | ✓ |
| 쌍대 bootstrap: 2D −0.15 [−0.22, −0.07], −0.26 [−0.36, −0.16]; 3D 0.00 [−0.09, +0.09], −0.21 [−0.31, −0.12] | benchmark/bench_paired.csv (bench_boot.py) | — | — | §3.4 | ✓ |
| 원판·구를 뺀 평균: 0.58 / 0.72 / 0.78 (2D), 0.35 / 0.34 (3D) | bench_summary_family.csv | — | — | §3.4 | ✓ |
| Gaussian 폭 선택 0.75 (2D) / 0.5 (3D), 보정 상수 | benchmark/bench_constants.json | — | — | §3.4, Methods | ✓ |
| flower: 최소 곡률반경 ≲ 1.7 px에서만 큰 오차, 그 밖에는 |오차| < 1.5% | bench_raw.csv | Fig. 5(a) | — | §3.4 | ✓ |
| 간격 sweep: 2 px 이상에서 ±0.2% 이내, 1.75 px에서 −0.46%, 1 px에서 −2.9% | benchmark/bench_gap.csv (bench_gap.py) | Fig. 5(e) | — | §3.4 | ✓ |
| near-touch 평균 0.54 (2D) / 0.44 (3D) | bench_summary_family.csv | Fig. 5(a, c) | — | §3.4 | ✓ |


## 최적화 반영 (2026-11-18)
| 값 | 데이터·스크립트 | 그림 | 표 | 본문·초록 | 일치 |
|---|---|---|---|---|---|
| stencil α에 따른 m(φ) 폭: 2D α = 1/4에서 0.0276 → 0.0140, 3D 최적 α ≈ 0.05 | benchmark/t20_kernel.csv (t20_kernel.py) | — | — | §2.3 | ✓ |
| 후보 설정의 train 오차 | benchmark/t22_*.csv, t23_*.csv, t25 / t26 출력 | — | — | Methods | ✓ |
| 선택된 2D 설정(h = 1.5, 방향 보정)의 test MARE: 0.54 / 0.58 / 0.81 | benchmark/bench_test2d_opt.csv (t27_test2d.py) | Fig. 5(a, b) | Table 2 | 초록, §3.4 | ✓ |
| 쌍대 차이: 대 Gaussian −0.15 [−0.22, −0.09], −0.13 [−0.20, −0.06], −0.25 [−0.40, −0.11]; 대 Crofton −0.26 [−0.38, −0.13]; 대 h = 2 판독: 0.00 [−0.04, +0.03], −0.14 [−0.20, −0.09] (1%) | benchmark/bench_paired_opt2d.csv (t28_post.py) | — | — | §3.4 | ✓ |
| 원판을 뺀 2D 평균 0.59; flower 1.6% (곡률반경 ≥ 1.7 px이면 < 1.7%) | bench_test2d_opt.csv | Fig. 5(a) | — | §3.4 | ✓ |
| 간격 sweep (선택된 판독): 1.75 px −0.59%, 1 px −3.9% | benchmark/bench_gap.csv, 열 e_opt | Fig. 5(e) | — | §3.4 | ✓ |

## rev8 추가 (래스터화 하한, 회색조·컬러) — 데이터: data/revision2/floor_gray/
| 원고 수치 | 출처 |
|---|---|
| 정보 하한 지수 −2.02, 계수 0.159 (식 infobound) | t31_infofloor.csv → t41_floor_analysis.py |
| c=0.050 (상관 0.77), Gaussian 0.057, 2/9π=0.071 | t39_pose_all.csv → t41_floor_decomp.csv |
| 원판 하한 대비 1.3–1.5배(R=4–6), 3.3배(R=26) | t41_disk_sweep_summary.csv vs t31 |
| 꽃잎형 제외 0.33 중 0.28 | t41_floor_decomp.csv (no_flower:PBGIA_bin) |
| 회색조 c=0.013, 원판 R≤8 하한 미만·이후 1.5배 이내 | t41_* (PBGIA_gray) |
| Table graycut | t43_gray_multicut.csv |
| marching sq. 대비 +0.10 [0.06,0.16], −0.67 [−0.76,−0.57], flip 0.2% 108 vs 0.30, 1.01 vs 0.34 | t42_gray_fresh.csv, t42_gray_fresh_paired.csv |
| Table colorext | t44_color_extent.csv |
| 컬러 방향장 41–46°, 1.48/1.23, 2.02/1.18, 0.30/0.40, 1.41/0.96; 사진 회전 오차 범위 | t45_color_director_synth.csv, t45_color_director_rot.csv |
| Table multithresh 'continuous' 열, 줄무늬 0.20° vs 0.32° (σ=8) | t46_gray_director_rot.csv, t46_strata_sigma.csv |

## rev9 추가 (연속 원천 텍스처 재분석) — data/revision2/floor_gray/t47_*
| 원고 수치 | 출처 |
|---|---|
| 국소 회전 오차 연속 0.1–1.9°, 컷 1.0–6.9°, ST 0.05–1.0° | t47_real_sigma.csv (local_rot_err, σ=2–16) |
| 줄무늬 0.16° (σ=6), 컷 1.9°, ST 0.15° | t47_real_sigma.csv (gt_err) |
| 나뭇결 결함 33/0 vs 389/24 | t47_real_sigma.csv (n_defects) |
| LIC: σ=8에서 12개, σ=10–14에서 4개(σ≤12 하나 매칭 반경 밖) | t47_lic_sigma.csv |
| 잔디 무작위장 비 0.70–1.14 / 0.74–1.01 / 0.66–0.92 | t47_real_sigma.csv |
| 다중 컷 수치가 rev8과 동일(검증) | t47 vs t16_real_sigma.csv 차이 0 |

## rev10 (혼합 방식 시험) — t48_*, t50_*, t49_lic_mech.py 출력
| 원고 수치 | 출처 |
|---|---|
| 부분 포화 β=1,3: σ=8 결함 10, 8; 평균 국소 회전 0.83°, 1.23° | t48_lic.csv, t48_real.csv |
| 층별 필터 후 재조합 K=8–32: 결함 수 = 연속, 회전 2.0–2.6° | t48_* |
| |∇s| 가중: σ=8–14 4개 모두 일치, 평균 0.65° (ST 0.50°), σ=4에서 93개 | t50_lic.csv, t50_real.csv |
| 9–32 컷: σ=4에서 16–18개 | t49_lic_mech.py 출력 |

## rev11 자기 도메인 실측 영상 — data/revision2/magnetic/
| 원고 수치 | 출처 |
|---|---|
| 스펙트럼 비등방 3.1, 격자 간격 29.8 px, 도메인 수 1067–1082 (분위 0.70–0.85), 1074 vs 1120, LoG 1144 | t51_ltem.json |
| 도메인당 Q = 1.0000 ± 0.0015, 최대 편차 0.026, 총 1075.2 | t51_ltem.json, t51_ltem.npz |
| Kerr SNR 0.72/0.78, σ = 2.5/2.0, 면적 ≤0.4%, 둘레 ≤1.8% (평균 0.8%) 일치, 3차 맞춤 잔차, median/Otsu 1.3–3.0배 | t52_kerr.csv, t52_kerr_smooth.csv |
| 미로: 주기 10.7 px, 끝점 489 / 분기점 615, σ=1.5·반경 2 px 일치율 42% (n=1148), 무작위 12%, 부호 교환 7%, 재현율 43%, ST 39%/43% | t53_labyrinth.json, t53_labyrinth_defects.csv, t54_lab_controls.csv |

## rev13 (NC 초안: 업데이트 횟수 N, 실행 시간, 보정 표현, 자성 영상 SI 이동)
| 수치 | 출처 | 위치 |
|---|---|---|
| λ = zJ/(zJ+h) = 0.727 (2D, h=1.5), 0.75 (3D, h=2) | 해석식(선형화) | 본문 Eq. (rate), SI 3.x |
| 벽 잔차 감소율 0.74 (N=12–30), 1.1e-4 (N=20), 6.4e-6 (N=30), float32 바닥 4e-8 (N≥100) | nsteps/t56_nsteps_aux.csv (t56_nsteps_aux.py) | 본문, SI Fig. nsteps(a) |
| 2D MARE vs N: N=8 0.546; N=12 0.545/0.587/0.881; N=20 0.544/0.580/0.817; N=30 0.544/0.580/0.814; N=0 2.51 | nsteps/t55_nsteps.csv (t55_nsteps.py) | 본문, SI |
| 3D MARE vs N (N=30 상수): N=12 0.400/0.470/0.655; N=20 0.355/0.366/0.499; N=30 0.350/0.362/0.492; N=0 8.58 | t55_nsteps.csv | 본문, SI |
| 3D 재도출(고정 상수에 앵커): N=8 0.348 (clean) | nsteps/t55b_nsteps3d.csv (t55b_nsteps3d.py) | 본문, SI |
| 벽 법선 rms 오차: 4.70° (clean), 4.95° (1%); Gaussian σ=1.5: 4.96°, 6.04° | t56_nsteps_aux.csv | SI |
| 실행 시간(1코어): 2D 갱신당 0.30 ms, 초기화 1.2, 판독 1.7, N=30 총 12 ms; 3D 갱신당 8 ms, 초기화 53, 판독 21, 총 0.32 s; Crofton 0.30, MS 0.19, G+MS 0.40, MC 7.9, G+MC 12 ms | nsteps/t57_runtime.json (t57_runtime.py) | 본문, Methods, SI |

## rev14 (벽 법선 오차 재검토, 세포 핵 영상)
| 수치 | 출처 | 위치 |
|---|---|---|
| 벽 법선 rms(최근접 경계점 정확 법선, 2 px 띠): 픽셀 3.9° (b≥8), 4.7° (b<8); 방향자 σ=2 2.5° (b≥8); 가우시안 σ=1/1.5/2: 5.1/3.1/2.4° | nsteps/t56b_angle_N.csv (t56b_angle_N.py) | SI Fig. nsteps(d), SI 3.x |
| 직선 벽 픽셀 오차 3.7°; σ=2 방향자 1.6→1.2° (편향 보정), σ=4 1.2→0.8° (가우시안 0.5°) | nsteps/t59_ellipse_normals.csv (t59_normal_bias.py) | SI |
| 방향 편향 δφ=Σ a_j sin4jφ: a=2.39°, 0.32°, 0.11° (잔차<0.1°); 예측 (k²/24) sin4φ, ⟨k²⟩=0.76 → 1.8° | nsteps/t59_bias.json | SI |
| 세포(BBBC038/StarDist, 테스트 50장): 표 수치 전부 | cells/t60_cells.csv, t60_cells_cfg.json (t60_cells.py) | 본문 Fig. 8, SI Table 3 |
| 조정된 비교 방법: σ=0.5, 최소 10 px (훈련 447장) | cells/t60_cells_cfg.json | 본문, Methods |
| 차수: 150장 모두 정수(5e-8); 디지털 χ와 일치 68–76%; 4/8 연결 χ 불일치 28%(이진) | cells/t60b, t60c, t60d | SI |
