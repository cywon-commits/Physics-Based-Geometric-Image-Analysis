# 재현 명령과 provenance

## 저장소 및 기준

저장소: https://github.com/cywon-commits/Physics-Based-Geometric-Image-Analysis

새 독립 branch: `codex/final-pbgia-20261117`, base main commit `0ec340537c956bf4fe333e7268ce9495eebb9527`. 이 디렉터리만 추가한다. 원시 과거 결과를 수정하지 않으며 PR은 merge하지 않는다. 최종 변경 commit은 이 파일의 Git commit 자체로 식별한다(`git rev-parse HEAD`); self-referential commit SHA를 파일 안에 적어 hash를 바꾸지 않는다.

기존 자료는 다음 ref에서 실제 코드·설정·보고서를 확인했다.

| 대상 | commit |
|---|---|
| dynamic 및 fixed_iteration | `0ec340537c956bf4fe333e7268ce9495eebb9527` |
| generator-first primitives | `54287cdc0b1e4ce3fefecb2df42467c190985355` |
| reference-invariant | `cf4f99ab4e98bb5658b050eaa9420ed50efabb9c` |
| domain-wall readout | `0dcff4b61ff439bbfb56f5705f614c8853a2922d` |

[historical_sources.json](reference/historical_sources.json)과 [SOURCE_PROVENANCE.csv](reference/SOURCE_PROVENANCE.csv)에 원래 로컬 source 위치, 공개 ref 및 SHA256을 기록했다. vendor/baseline.py의 5개 함수와 초기화/gradient는 기존 generator-first 코드에서 AST로 추출했다. degree 코드는 기존 구현을 재사용했다. certificate는 기존 refined diagnostic 함수를 추출했다. source와 vendored bytes는 hash가 다를 수 있으며 transformation을 명시했다. unused reference snapshot은 주 실행에 쓰지 않는다.

## 환경

실행 Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-image 0.26.0, matplotlib 3.10.8, pytest 9.2.0. Linux CPU, OpenBLAS/OMP/MKL thread1, GPU 없음. dependency 버전은 [requirements.txt](requirements.txt)에 고정했다. 초기 세션의 환경 교체 후 scikit-image/pytest가 빠져 있어 writable target에 같은 버전을 복원했다. 최초 final-test import 실패는 field 계산 전에 발생했으며, 성공한 독립 시험의 audit를 별도로 저장했다. 시스템 경로나 credential을 재현에 요구하지 않는다.

```bash
git clone https://github.com/cywon-commits/Physics-Based-Geometric-Image-Analysis.git
cd Physics-Based-Geometric-Image-Analysis
git switch codex/final-pbgia-20261117
cd final_pbgia_study
python -m venv /tmp/pbgia-final-env
source /tmp/pbgia-final-env/bin/activate
python -m pip install -r requirements.txt
python code/reproduce.py --workdir /tmp/pbgia-final-new-run
```

driver는 기존 경로의 덮어쓰기를 거부한다. 위 명령은 배포 데이터를 보존하면서 새 디렉터리에 독립 재현을 만든다. 모든 명령은 `code/reproduce.py`에 있으며, 각 subprocess의 log와 wall time을 새 `output/run_logs`와 `output/reproduction_stage_times.json`에 저장한다.

新규 재현은 동일 frozen 과학 protocol의 **검증**이며, 시험 숫자를 보고 후보를 추가하는 두 번째 모델 선택이 아니다. model_freeze의 timestamp/hash 및 timing은 달라질 수 있지만 선택 파라미터는 같아야 한다. CSV 순서와 주요 수치도 비교한다. 새로운 원형 또는 kernel 후보를 추가하지 않는다.

## 실제 명령 순서

1. `python code/preflight.py`: 새 manifest/protocol 저장.
2. `python -m pytest -q tests -k 'not freeze_manifest_and_heldout_once_audit'`: 비용 큰 실행 전 smoke/numerical 검증.
3. `python code/geometry_study.py validate`: 300개 train/validation/reference, α 선택 및 freeze.
4. `python code/geometry_study.py test`: 144개 frozen-new-test를 한 번 실행. 동일 output 재실행 금지.
5. `python code/physics.py`: planar N0–40, 폭, 에너지, 변분 비교, spacing, checkerboard 반례.
6. `python code/interactions.py`: 얇은 shell/가까운 경계.
7. `python code/topology_study.py`: S³ positive controls, 4개 generated endpoint audit.
8. `python code/projections_study.py`: 6개 field, 3축, 2가중치, full/vector/in-plane tensor, loop 판정.
9. `python code/pixel_information.py`: 48개 동일 mask 쌍, ellipse orientation, corrugation.
10. `python code/stability_benchmark.py`: 독립 rotation/shift 대조, 고정 Gaussian 전통 대조, isolated timing.
11. `python code/analyze.py`: 요약·paired bootstrap·tensor perturbation·그림.
12. `python -m pytest -q tests`: 최종 수치 테스트.

최초 실행의 train/validation에는 이후 fixed s0/local-normal 관측을 별도 추가했다(`geometry_study.py supplement`). baseline I_E가 원래 CSV와 일치함을 검증했고 calibration/선택을 바꾸지 않았다. torus PCA는 test 전에 목표가 symmetry axis임을 확인해 covariance 최소축을 사용했다. 새 재현 경로는 처음부터 이 정의와 모든 고정 observer를 포함한다. 보조 conventional Gaussian 대조는 test 후 추가한 **descriptive comparator**로, σ=1을 고정했고 PBGIA 선택에 사용하지 않았다.

## 실행한 독립 재현 확인

별도 `/workspace/pbgia_final_clean_reproduction`에서 전체 순차 pipeline을 실제 실행했다. 단계 wall time의 합계는 **234.49 s**, 최대 child process RSS는 **242.98 MiB**였다. 이는 단일 환경의 측정이며 대규모 데이터의 비용 예측이 아니다.

| 단계 | wall s |
|---|---:|
| train/validation | 96.58 |
| frozen test | 47.79 |
| physics | 38.87 |
| interactions | 3.22 |
| S³ topology | 3.32 |
| projections | 7.64 |
| pixel information | 13.34 |
| controlled benchmarks | 15.15 |
| 분석·그림 | 4.72 |

preflight와 양쪽 테스트 시간도 234.49 s에 포함한다. 새 run에서 최종 **13개 테스트가 통과**했다. 두 run의 selected configuration이 정확히 같았고, test I_E/I_TV/s0/법선·축 지표, pixel GT/추정값, S³ degree/certificate, projection 정의 비율/purity/후보수의 비교 대상 **14개 column 모두 최대 차이가 0**이었다. [검증 JSON](output/clean_reproduction_verification.json)을 제공한다. 이 검증은 모든 CSV timing, timestamp 또는 모든 이미지 bytes의 일치를 뜻하지 않는다.

## 테스트와 데이터 보호

테스트는 과거 pinned state의 bitwise 재현, dynamic update·adjoint·f 불변·unit norm·zero-H 분기, 실제 Hamiltonian 미분 및 변분 descent, 에너지 증가 반례, split freshness, rank-one projector/혼합 tensor 반례, sign-gauge/loop controls, trace0 abstention, S³ degree 부호, 동일 mask와 minimax bound, freeze audit를 검사한다. Dice/IoU 또는 topology/reconstruction 결과로 최종 모델을 고르지 않는다.

새 raw 데이터와 기존 historical fixture는 구분한다. `historical_regression_2d.npz`/`3d.npz`는 옛 모델 정확 일치를 확인하는 소형 regression fixture이고, `projection_tensors.npz`는 축 적분 후의 2D tensor들이다. 새 full 3D spin checkpoint나 optimizer state는 게시하지 않는다. 원형은 코드·seed로 재생성한다.

## 파일 무결성과 재현의 한계

`python code/package_audit.py`는 필요한 산출물, local report links, freeze/metric consistency와 공개 파일 크기를 검사하고 `output/package_audit.json` 및 `SHA256SUMS.txt`를 만든다. `python code/package_audit.py --verify-hashes`는 배포된 bytes를 검사한다. hash 목록 자체와 캐시를 제외한 모든 공개 파일을 SHA256으로 식별한다. source snapshot audit에는 당시의 미사용 reference 파일과 나중에 추가한 diagnostic 함수 전 버전이 포함될 수 있다. 기본 inference/common/geometry 함수의 bytes는 독립 재현으로 다시 검증했다.

floating point의 bitwise 일치는 현재 명시한 CPU/library 환경의 관측이다. 다른 플랫폼에서는 numerical tolerance를 사용하되 sign/degree/gap 판정의 경계 사례를 재검토해야 한다. Euler/degree certificate도 interval arithmetic은 아니다. base spin 및 모델 선택의 과학적 한계는 최종 보고서에 따로 기록했다.

PDF는 optional `python -m pip install -r requirements-report.txt` 및 `python code/export_pdf.py`로 만든다. 실행 환경의 Noto Sans CJK KR 글꼴을 사용했다. PDF의 표·한글·첫 페이지 및 그림을 확인했으며, 이는 과학 실행용 dependencies와 별개다. scientific driver는 PDF 생성을 요구하지 않는다.
