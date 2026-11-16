# PBGIA fixed-iteration Codex 결과 비판적 검토 (2026-11-16)

**검토 수준:** GitHub의 실제 `main` 브랜치 보고서, 코드/결과 파일 목록, `selected_configuration.json`, `design.json`, `test_summary.csv`, `validation_summary.csv`, `selection_sensitivity.csv`, `refinement_margin_checks.csv`, `transfer_summary.csv`, `test_audit.json`, `tests.log`, `artifact_checks.log`를 읽고 상호 대조했다. **전체 고비용 계산을 독립 재실행한 결과는 아니다.** 원자료와 집계표의 모든 행별 일치도 완전 재집계하지 않았으며, '검증'은 현재까지 확인한 범위에 한정한다. 선행 dynamic-normal 장시간 연구는 [../RESEARCH_REPORT.md](../RESEARCH_REPORT.md)에, 새로운 고정 반복 실험은 [../fixed_iteration/output/REPORT_KO.md](../fixed_iteration/output/REPORT_KO.md)에 있다.

## 1. 알고리즘 일치

연구 지시대로 스칼라 입력 `f`는 고정, `grad(s)`는 매 단계 갱신, 교환은 positive axial-neighbor sum, direct `H/|H|` normalization, `div(v)` 미사용이다. 생성 커널 고정과 **측정 커널별 독립 학습**도 구분되어 있다. 비변분 정규화 반복을 해밀토니안의 경사 흐름/단조 감소로 주장하지 않으며, `trace(T)`를 교환에너지라고 하지 않는다.

단순 `J\Delta n`으로의 교체는 정규화 fixed-point를 바꾸므로 금지해야 한다. 경계는 periodic이므로 충분히 넓은 바깥쪽 일정 외부영역이 전제다.

## 2. 반복횟수 N=30의 선택 근거

`N = 5, 10, 20, 30, 40`을 검사했으며 train/validation/test는 geometry prototype 단위로 분리했다. `selected_configuration.json`에는 **N=30**, 두 차원의 서로 다른 3-parameter 미분 커널, 독립 reference 보정계수 및 선택 후 test를 해제할 수 없도록 남긴 SHA 기록이 있다. `test_audit.json`에는 1,320개 test 행과 `no_retuning=true`가 기록돼 있다.

Validation 규칙은 각 dimension × noise stratum에서 `N=40` 대비 geometry MARE 증가 ≤0.1 percentage point, orientation MAE 증가 ≤0.1°, 실패 증가 없음 및 시간 단축이다. `selection_sensitivity.csv`에 따르면 `N=20`은 특히 3D geometry +0.128~0.196 pp, orientation +0.23~0.28°로 탈락한다. `N=30`은 사전 기준을 통과했다.

**중요한 제한:** `N=20`의 탈락은 동역학 자체의 미수렴만으로 해석할 수 없다. N별 3변수 커널이 같은 270회 objective evaluation의 유한 예산으로 별도 학습되었기 때문이다. `kernel_by_iteration.json`의 optimiser success flag는 최대 반복 도달로 false이다. 탐색 시작점과 예산을 달리하면 선택이 바뀔 가능성을 아직 배제하지 못한다.

## 3. Test 결과 (동일한 최신 split의 수치끼리 비교)

| 조건 | PBGIA-30 2D 둘레 MARE | PBGIA-30 3D 면적 MARE |
|---|---:|---:|
| clean | 0.9972% | 1.4040% |
| salt-and-pepper 0.2% | 0.9145% | 1.3250% |
| salt-and-pepper 1% | 0.7749% | 1.3180% |

오차가 잡음률과 함께 약간 줄어든 것은 해당 합성 형상/보정에서의 결과다. 일반적인 noise-improvement 법칙으로 볼 수 없다. 원래 연결 구조의 소실 여부와 둘레 오차를 구분해야 한다.

비교 baseline도 **동일 split**만 사용해야 한다.
- Clean 2D: raw Crofton **0.4399%** < PBGIA-30 **0.9972%**, 따라서 정확도 우월성은 성립하지 않는다.
- Clean 3D: PBGIA-30 **1.4040%** < reference-calibrated marching cubes **1.7690%**, 이 합성 집합의 작은 우위다.
- 주축: 2D PBGIA 0.4103°, binary PCA 0.2256°; 3D PBGIA 0.4130°, binary PCA 0.2474°. 분리된 둥근 물체의 중심간 축은 경계 법선의 적분 텐서로 정의하기 곤란하다.
- 2D에서 `prior_learned` 커널 clean geometry **0.8332%**, Scharr **0.8572%**로 이번에 새로 학습한 **0.9972%**보다 작다. **현재 훈련의 learned 커널이 최고**라는 표현은 부정확하다.
- 기존 다른 연구 분할의 `2D 0.2999%`/`3D 1.3872%` 수치를 현재 split의 최종 결과와 비교하면 안 된다.

## 4. 실제 속도와 실패

배포 CPU 중앙값: N=30 2D(80²) **12.79 ms**, 3D(56³) **339.60 ms**; N=40 각각 **16.81 ms**, **444.3 ms**. 반복수 감소에 따른 약 1.31배 이점이 보고된다. 그러나 conventional baseline은 수백 마이크로초~수 밀리초 수준으로 실행돼 '기존 방법보다 빠름' 주장은 성립하지 않는다. Timings는 shared CPU 환경에서의 소규모 계측으로, GPU 및 큰 영상 성능을 대표하지 않는다.

3D test에서는 잡음률 계층당 **mask failure 2건**, clean minimum Dice 약 **0.8364**가 보고된다. 이는 정수 degree 또는 Euler의 일치만으로 실제 물체 치수를 보장할 수 없다는 반례다. h=4, 2D transfer에서 **unresolved wall 10건**이 나타났고, 해당 고정 계수를 새로운 h에서 사용할 때 경고·재보정이 필요하다.

## 5. 수학적 주의

- 초기 1D 스텝장 해석해는 원리의 기원이나, 현재 비변분 normalized map의 정확한 wall 해가 아니다. `sqrt(2A/h)`와 BPS 장력 등은 새로운 방법의 정리로 재인용하지 않는다.
- 3D 위상 degree의 정수성은 `S³` 사상의 **외부 collar compactification**, 사면체 사상의 비특이적 보간, 충분히 떨어진 정규값 검증에 의존한다. 보고된 정수는 컴팩트화된 장에 대한 값이다.
- `chi`의 연결 성분 판정은 `S³` degree와 서로 다른 관측량이다.
- `trace(T)` 기하량은 보정치다. 경계 위치와 반지름, 곡률과 벽 두께에 따라 오차가 있을 수 있다.
- 고정 N에서의 관측량 정확도와 field fixed-point residual, wall-resolution 및 eigen-gap은 별도 보고해야 한다.
- Grid-refinement의 spacing별 보정계수 재학습은 물리적 단일 해상도에서의 일관성 검증에 가깝다. 물리적 wall width를 실제 단위로 고정한 보편적인 연속극한 수렴 증명은 아니다.

## 6. 재현성 감사와 후속 연구 우선순위

`tests.log`에 **12 passed**, `artifact_checks.log`에 split/lock/normalization/candidate-selection **PASS** 기록이 있다. 이는 근거 추적에 도움이 되지만 코드를 새 환경에서 풀로 실행한 것과 동일하지 않다. 권장 재현 절차: `cd fixed_iteration && python -m pytest -q tests && python run_all.py --fresh`; 이때 기존 output 보존 및 test lock 규칙을 따른다.

우선순위: (1) **2D와 3D의 별도 N 선택** 및 전용 커널 비교, (2) 커널 다중 시작점/학습 예산을 늘린 재최적화, (3) 3D 형상 실패 사례 상세 분석, (4) voxel spacing/물리적 자구벽 폭/grid refinement, (5) 전처리/주축/위상수의 conventional baseline과 실제 영상 비교. 실험 설계는 새 split으로 사전 등록하고 기존 test에서 재튜닝하지 않는다.

## 잠정 결론

현재 저장소 자료는 **PBGIA-30을 이번 synthetic benchmark에 대한 유효한 고정 연산량의 prototype**으로 뒷받침한다. 다만 이는 보편적 최적 알고리즘이나 기존 알고리즘 대비 일괄 우월성의 증거가 아니다. 특히 원시 Crofton의 2D 정확도, 형상 보존 실패 및 kernel-search 예산의 영향을 다음 단계에서 반드시 확인할 필요가 있다.
