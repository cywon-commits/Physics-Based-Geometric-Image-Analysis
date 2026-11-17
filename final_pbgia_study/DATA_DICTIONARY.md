# 데이터 파일과 관측량 정의

| 파일 | 내용 |
|---|---|
| manifest.csv / protocol.json | 444개 입력 recipe, split/prototype/pose/noise seed, 제한 선택 규칙 |
| train_validation_raw.csv / test_raw.csv | 같은 N30 field의 I_E, I_TV, s0, tensor 각도/gap, 법선 정렬, 단위길이, 시간 |
| *_conventional.csv | raw binary Crofton/marching/PCA. method별 value는 boundary 크기이며 PCA는 value=NaN |
| test_conventional_gaussian.csv | 고정 σ1 Gaussian+threshold의 descriptive traditional benchmark |
| calibration.json / alpha_validation.csv / model_freeze.* | train reference c, 다섯 α 후보, test 전 freeze 및 hash |
| geometry_errors_long.csv | 방법별 개별 case estimate, signed error_percent; raw/train_c 구분 |
| geometry_summary.csv / geometry_by_family.csv | prototype 균등 MARE와 bias, noise/family별 결과 |
| paired_bootstrap.csv | 6개 prototype cluster의 선택−energy MARE 차이 bootstrap; 일반화 보증 아님 |
| orientation_summary.csv | 목표를 구분한 축 지표; comparison_is_accuracy=False는 다른 관측량을 비교한 진단 |
| physics_trajectory.csv / wall_profiles.csv | N0–40 실제 H 및 I_s/I_v, r1/r2, 실측 폭과 ray profile |
| initialization_stability.csv | w_init별 N30 scalar RMS 차이; 최종 w_eff와 다름 |
| variational_comparison.csv / synchronous_counterexample.csv | 기본 모델과 별도 reciprocal descent, monotonicity 반례 |
| spacing_diagnostics.csv | physical-spacing 정책(D=1/a)의 원/구. pixel study와 다른 정책 |
| boundary_interactions.csv / interaction_profiles.csv | thin shell·가까운 경계, input/scalar connectivity 및 positive phase 손실 |
| spin_topology.csv / topology_convention.json | cap한 S³ endpoint degree/certificate와 별도 binary Euler/components |
| projection_case_manifest.csv / projection_tensors.npz | 6개 field recipe, 적분 후의 full/vector 2D tensors. key는 case_axisN_representation_weight |
| projection_summary.csv | 108개 full/vector/in-plane map의 gap/purity/정의 영역 및 후보수 |
| projection_inplane_maps.csv | axis2 wall in-plane map의 Q00,Q01,Q11와 gap. 전체 map은 NPZ에서 유도 가능 |
| projective_positive_controls.csv | RP¹/RP²의 8개 유효 loop, 2개 퇴화 loop 양성/유보 대조 |
| defect_candidates.csv | nontrivial coarse cell과 동일 square 64/128 sample 재판정. 원래 spin singularity GT 아님 |
| projection_loops.csv | map 중심의 radius3/6/10 loop 판정과 최소 sampled gap/trace |
| projection_scalar_dominance.csv | full tensor leading line의 scalar 성분 비중 |
| projection_tensor_perturbation.csv | PSD 혼합 tensor perturbation 안정성. 입력 noise 실험과 구별 |
| pixel_pairs.csv / pixel_orientation_pairs.csv / pixel_pathological.csv | 동일 mask의 다른 smooth truth/축/요철. shared mask hash, estimator, 두 endpoint 오차와 하한 |
| controlled_rotation_shift.csv | 회전과 shift를 독립 변경한 ellipse/ellipsoid, 16개씩 |
| isolated_runtime.csv / *_audit.json | 비용 분리, 시험/투영/pixel 실행과 검증 metadata |
| clean_reproduction_verification.json | 전체 별도 재현 단계별 비용 및 주요 column 일치 확인 |

경계 크기 GT는 2D perimeter 또는 3D surface area다. 원본 mask의 면적/부피와 혼동하지 않는다. I_E raw는 geometric gradient energy이며 Hamiltonian ℋ가 아니다. scalar_fraction=I_s/(I_s+I_v). I_TV는 ½∫|G4s|. I_s0는 생성된 scalar0의 raw contour/surface 크기다.

각도는 degrees, unoriented line 사이 acos(|a·b|)다. array axes x0,x1[,x2] 순서를 따른다. normal target은 inward boundary normal, normal distribution tensor 축과 spatial covariance 축은 정의가 다르다. circle/sphere/ring의 전역 axis는 NaN이다. pair_separation 목표에서 energy/normal tensor 각도는 연결축과의 비교 진단이며, ‘법선 정확도’가 아니다. gap은 각 tensor 정의에 따른 정규화값이며 spatial/normal/energy의 절대값을 직접 같은 threshold로 비교하지 않는다.

projective gap은 trace-normalized λmax−λsecond이고 purity는 tr(Qnorm²). trace0은 purity/방향 미정이다. `DEFINED`는 threshold와 sampled neighbor coherence를 통과한 수치 판정이다. `INCONCLUSIVE`를 0 winding 또는 무결함으로 바꾸지 않는다. `stable_candidates`는 bilinear projected map의 loop 재표본화 안정성만 뜻하며 실제 GT defect의 TP 수가 아니다. 생성 map의 FP/FN은 알려져 있지 않다.

pixel CSV의 `identical_prediction=True`는 같은 deterministic array 입력이라는 뜻이다. 모든 pair를 중복 독립 계산한 척하지 않으며, 2개 explicit duplicate pipeline 확인은 pixel_audit에 있다. lower bound는 두 truth만의 absolute minimax 하한이고 excess는 전체 error attribution이 아니다. 작은 curvature 제한을 가진 원/구와 무제한 curvature corrugation은 별도 파일이다.

독립 표본 수는 image 개수와 다르다. geometry test는 차원별 6 prototype이며 pose/noise 파생 72개다. tensor map 108개는 같은 6개 field의 대조여서 108개 독립 영상으로 세지 않는다. 모든 timing은 해당 환경의 wall seconds이고 dtype 및 transient memory 비용도 포함한다.
