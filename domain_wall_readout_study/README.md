# 고정 자화장에서의 기하 관측량 비교 (2D·3D)

동일한 G4 생성장에 gradient energy, scalar energy, TV, weighted TV, point/integrated coarea와 생성된 s=0 marching을 적용한다. 생성 커널·epsilon·N을 학습하거나 이 데이터에서 최적 방법을 선택하지 않는다. [한국어 보고서](output/RESEARCH_REPORT_KO.md), [PDF](output/RESEARCH_REPORT_KO.pdf), [전체 조건별 수치](output/condition_summary.csv)가 실제 결과다.

## 실제 계산과 결론

- 562조건(2D283/3D279),11관측량,6,182행. 이 중 calibration-only 원/구12조건을 제외한 평가조건은550개다. 본 실행은wall180.30초,CPU179.90초,peakRSS약594MiB였다.
- 같은 예약 참조로 보정한 clean 형상군:2D TV MARE0.733%,energy1.999%;3D energy1.943%,TV2.368%.3D 곡률군은energy0.632%,TV3.581%. **관측량을 coarea로 일괄 교체할 근거는 없다.** 형상군이 작아 일반적 우월성으로 해석하지 않는다.
- 좁은 delta의 point sampling은 subpixel/해상도에 매우 민감했다. CDF를 먼저 적분한 readout도 G4 ringing으로 overcount할 수 있다.
- 3D는 실제 volumetric field/면적/방향 readout 계산을 완료했다. 이전3D kernel optimization의 미완료를 새 학습 성공으로 부르는 연구는 아니다.
- Energy의 단위는length^(d−2),면적/둘레는length^(d−1)이다. Rawenergy MARE를coarea raw오차와 직접 비교하지 않는다. 모든 방법의 global reference보정과,기하단위 방법의raw오차를 따로 제공한다.

## 재현

Python3.12,CPU단일BLASthread,float64. NumPy/SciPy/Skimage/Matplotlib를 사용하며GPU와PyTorch는 필요 없다.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_all.py --prepare --tests
python run_all.py --run
python run_all.py --diagnostics --report --paired --audit
```

`--all`은 위 단계를 순서대로 실행한다. 새로운 재현은 기존output을 보존용 이름으로 옮긴 후 시작한다. raw.csv가 존재하면prepare와benchmark덮어쓰기를 거부한다. `--report/--paired`는 이미 계산한rawCSV만 분석한다. 큰field checkpoint는 저장/게시하지 않으며manifest의seed·GT정의와코드로 재생성한다.

## 비교 정의

기본 생성:NN sum2d,J1,D1,h2,current grad(s)의정확G4,N10,EDT/tanh latticewidth2.3,periodic,H/||H||,zeroH에서이전상태. f불변,div(v)없음. Oracle평면은별도분석용연속sech/tanh장에서표본화하며실제digital생성장과구별한다.

관측량11개는모두같은field를공유한다. `readout_metadata.json`에수식과단위가있다. 이론적으로같은pointdelta×|G4s|와|G4CDF(s)|도이산chain rule이성립하지않아수치적으로다르다. deltaepsilon은.20/.35/.50고정이며test에서최적폭을선택하지않았다.

평면평가창은중앙ROI다. density적분의cellbox와marching의sample-centerbox는L과L−a로달라,각방법에맞는정확plane-box교차GT를쓴다. Oracle의finitewindow기대값은독립1Dquadrature로구해geometrybias와samplingerror를분리했다. `continuum_radial.csv`는compacttheta의별도연속체1D적분이며pixelized벤치마크로세지않는다.

폭변수는oracle에서는물리profilescale,digital에서는초기EDT폭이다. N10뒤실측한wallwidth가아니다. Spacing suite는같은물리개체재표본화/물리초기폭고정/G4 derivative1/a/dVa^d를적용한다. J/h/N은고정이므로단일PDE의continuum convergence를증명한것이아니다. 실제domain span과격자크기를기록했다. `boundary_padding.csv`는같은spacing/ROI에서domain만키운별도probe다.

## 면적·방향·보정의 의미

예약원/구3radii×2shift/차원으로prototype-balanced global relativeLS c=Σq/Σq²를정하고,모든평가조건에고정전이했다. 개별참조반지름sweep도공개하며최저test오차참조를고르지않는다. 참조는split=reference로예약된calibration-only표준자이며optimizertraining이있었던것은아니다.

Normaltensor는공통G4(s)법선방향에각readout밀도를가중한다. Energytensor는모든n성분의gradient tensor를읽는공통baseline이다. Spatialcovariance는같은밀도의위치공분산이다. 평면법선/oblate법선/torus대칭축과prolate장축/pair분리축을구분해고유축convention을선언했다. 원/구에임의GT장축을부여하지않고,eigengap으로불리한사례를제외하지않는다. 두구의분리축은법선분포만으로식별되지않을수있음을해석적테스트와실제결과로분리해보였다.

Generated s=0 mesh는입력개체의GT가아니다. Coarea도생성장의등위면을읽으므로field의경계이동·근접벽·pixelization편향이남는다. Mask재구성·Dice/IoU·연결성·Euler·degree·벽너비·잔차를loss/gate/fallback에넣지않았다. 위상현상은공통자화장의중요한잠재응용이나이번목표의제약은아니다.

## 한계와 주요 파일

Synthetic형상과적은원형/rotation/shift/noise seed에한정된다. Bootstrap은원형단위이며calibration표준자를재적합한불확실성까지포함하지않는다. Paired비교는공통조건·공통GT에만적용한다(marching평면은다른footprint라제외). 측정시간은공유derivativepass와방법별처리를포함한구현비용이며최적화된single-head latency가아니다.

`raw.csv`,`manifest.csv`,`protocol.json`,`condition_summary.csv`,`reference_radius_sweep.csv`,`paired_readout_differences.csv`,`rotation_shift_variation.csv`,`energy_component_physics.csv`,`pair_direction_demonstration.csv`,`run_cost.json`,`artifact_audit.json`,`run_logs/`와그림을포함한다. 예전출력은변경하지않는다.
