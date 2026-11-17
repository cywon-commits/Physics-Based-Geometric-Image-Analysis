# Reference-invariant generator optimization

새 연구의 주 지표는 `q=I/P`의 원형 간 분산과 회전/shift 안정성이다. 특정 원의 `c=P/I`는 학습·선택에 쓰지 않는다. 사용자 의뢰서는 `CODEX_TASK_KO.md`, 결과는 `output/RESEARCH_REPORT_KO.md`에 있다. 과거 결과와 원본 노트북은 `reference/`의 읽기 전용 사본이다.

## 실제 결과

2D 학습14개가 모두 projected-gradient 수렴 기준으로 종료했다(separable2개는 active bounds2개). Train/validation/test 각각17원형이고, 별도 참조8개다. Noise-weight1 모델의 CV(q)는 train4.050%→1.389%,validation3.530%→1.531%,fresh clean test1.965%→1.566%였다. Test paired bootstrap CV차이95%구간은[−1.727,+1.212]pp로0을 포함한다.

Reference c의CV는1.042%→1.919%,최악참조MARE는2.707%→4.270%로 나빠졌다.1% noise paired 적분 drift도 악화됐다. **q분산 감소를 보편적 reference-invariance나일괄noise개선이라고 주장하지 않는다.**3D는 수치검증/gradient runtime probe까지만 완료했으며 학습·독립표면적시험은 미실행이다.

## 범위

2D: G4를 정확히 포함하는 비정규화·비분리형 5×5 6계수, 고정 NN 교환/고정 G4 관측자/N10. 네 초기값 × noise weight0,1,4의 L-BFGS-B와 separable moment-one 대조2개. Train/validation/test 각각17원형, 각3회전×2shift×4noise=408영상, train-only 별도참조8개. 학습 proxy는 모든 train 원형의 교차2pose, noise 설정에 따른 clean 또는4noise다. 각도 손실은 방향이 유일한 원형에만 적용한다.

3D: 12orbit 5³ 계열·G4 포함·대칭·3-step 자동미분을 검증한 **부분 구현**이다. 3D 최적화·독립 표면적 시험·sphere ruler sweep은 실행하지 않았으며 2D 수치를 대신 쓰지 않는다. `output/status_3d.json`에 비용 우선순위/남은 작업이 있다. 초기 프로토콜에서2D 완결과14 optimizer run에 예산을 우선 배정했다. 8GiB CPU 환경이3D를 원천적으로 불가능하게 한다는 주장은 아니다.

## 설치·재현

Python3.12, CPU1 thread, double precision. Torch는 CPU wheel을 사용한다.

```bash
python -m venv .venv
. .venv/bin/activate
pip install numpy==2.3.5 scipy==1.17.0 pandas==2.2.3 matplotlib==3.10.8 pytest==9.2.0 tabulate==0.10.0
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_all.py --smoke --tests
python run_all.py --train2d
python run_all.py --validate2d
python run_all.py --freeze2d
python run_all.py --test2d
python run_all.py --diagnostics --report2d
python run_all.py --smoke3d
```

새로 재현하려면 기존 output을 보존용 이름으로 옮긴 후 시작한다. 기존 모델/선택/test가 있으면 훈련·validation·freeze·test 덮어쓰기를 거부한다. `--train3d/--validate3d/--freeze3d/--test3d/--report3d`는 명시적으로 미완료 상태를 알리고 종료하며3D 계산을 가장하지 않는다. 보고서 재생성은 기존 raw CSV만 읽는다.

## 구현과 선택

`core.py`는 정확한 forward-mode Jacobian, `torch_engine.py`는 reverse autodiff다. 학습은 chunk8의 no-grad forward→소형 outer loss backward→chunk별 정확한 chain-rule backward로 메모리를 제한한다. 두 Jacobian과 유한차분을 smoke에서 비교했다. 무한소 도함수를 수치 차분으로 최적화하는 방식이 아니다. Normalization 분기 norm≤1e−12는 이전 상태, 분기 안의 도함수는 이전 상태 도함수다. Degenerate I≤1e−14나 nonfinite gradient는 수치 실패이며 불리한 형상을 필터링하지 않는다.

손실: 원형별 log mean(q)의 variance + 원형별 clean I CV² +0.0001×(angle/1°)² + w_noise×paired log(I_noisy/I_clean)². 진폭·선형 moment·고주파 gain을 정규화/penalize하지 않는다. 관측량은 벡터 성분 전부를 포함한 고정 G4 gradient tensor trace다. 위상·재구성·벽 너비·잔차 조건은 학습이나 선택에 없다.

각 noise 설정의 validation objective를 최소화하되 clean q분산 및 train/validation objective가 같은 G4 anchor보다 나쁜 학습 해는 채택하지 않는다. Primary N10과 별도 validation N선택은 분리한다. 참조8개는 frozen 결과의 c/MARE sensitivity 진단만 하며 최적 test ruler를 고르지 않는다. 7×7, 교환 변경, 관측자 재학습은 이번 예산에 포함하지 않았다.

## Provenance와 해석

기준 branch `codex/generator-first-20261116`, commit `54287cdc0b1e4ce3fefecb2df42467c190985355`. Notebook saved outputs와 cell 번호를 JSON으로 추출했다. 원본5×5 계수의 이식은 원본 full algorithm이 아니다. 기존 notebook은 raw Hadamard exchange/binary init/h1 등을 사용했지만 주 비교는 NN/EDT/h2다.

3D는 partial 상태이므로 `manifest_3d`, `model_freeze_3d`, `test_raw_3d`를 만들어 완료한 것처럼 보이지 않는다. 원형별 bootstrap CI는 적은 synthetic prototype에 대한 설명적 불확실성이다. 일반 이진 마스크에 고유한 참둘레를 부여하거나 완전한 회전 불변성/실제 영상 일반화를 주장하지 않는다. 위상 현상은 공통 자화장의 중요한 잠재 응용이나 이번 최적화 제약은 아니다.
