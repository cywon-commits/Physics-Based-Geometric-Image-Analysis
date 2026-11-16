# Generator-first PBGIA 연구 (2D/3D)

생성 미분 G와 양의 exchange KJ의 효과를 **같은 관측자** 아래 먼저 비교하고, 독립적인 기하/방향 관측자 재학습의 기여를 분리한다. 기존 결과는 변경하지 않는다. 새 설계에서는 형상 복원·위상 관련 후보 탈락 조건이 없다.

`CODEX_TASK_KO.md`가 사용자 제공 의뢰서다. 실제 결과와 한계는 [한국어 연구 보고서](output/RESEARCH_REPORT_KO.md), 선택 메타데이터는 [pretest_selection.json](output/pretest_selection.json)에 있다.

## 실제 실행 결과

- 120 generator 후보 평가, 독립 head의 DE 총24,000평가. 신규 test192영상/16독립 원형, 수치·split·동결·CLI 테스트9개 통과.
- Validation 선택: 2D/3D 모두 **고정4차 중심차분 G1, N=10**, 새 독립 관측 head. 학습형 생성기의 기하 정확도 우월성은 입증되지 않았다.
- 선택 모델 test MARE:2D2.326%,3D1.111%; frozen PBGIA-30:2D2.096%,3D1.336%. 방향·속도와 trade-off가 있어 일괄적 개선으로 부르지 않는다.
- [한국어 PDF](output/RESEARCH_REPORT_KO.pdf), [선택 계수](output/selected_coefficients.csv), [test 요약](output/test_summary.csv).
- 전체 train 약1,013초, validation1,506초, test270초(실측CPU 단일 thread 환경). 작은 train/validation 규모, G3 및 독립 walltime optimizer 미실행 등 한계를 보고서에 기록했다.

## 재현

Python 3.12, CPU 단일 BLAS thread. GPU는 필요 없다.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
python run_all.py --smoke --tests
python run_all.py --train
python run_all.py --validate
python run_all.py --test
python run_all.py --diagnostics --report
```

각 단계는 명시한 예산만 실행한다. 기존 `output`을 보존하려면 폴더를 다른 이름으로 옮긴 뒤 새 실행을 시작한다. 고정 선택이나 test 파일이 존재하면 훈련/선택/재시험을 거부한다. 보고서 재생성은 기존 CSV만 읽는다. `--all`은 위 단계를 순서대로 한 번 실행한다. 대용량 상태 checkpoint는 저장하거나 게시하지 않는다. manifest의 seed, 모델 계수와 소스로 다시 생성한다.

```bash
python code/infer.py mask.npy --preset geometry_priority --out prediction.json
python code/infer.py mask.npy --preset orientation_priority --spacing 1 --out prediction.json
```

마스크는 2D/3D 배열이며 양수 내부를 뜻한다. validation 규칙을 만족하는 공동 default가 없으면 우선순위 preset을 명시해야 한다. 추론은 정확히 고정 N회 실행하며 모델을 자동 교체하지 않는다. 임의 spacing은 생성 미분에 1/a, 관측 tensor에 물리적 미분과 부피요소(a^(d−2))를 반영한다. 등방 spacing만 지원하며 spacing 외삽 정확도는 결과의 검증 범위에 제한된다.

## 연산자와 통제

- G0 중심차분; G1 고정 4차; G2 separable 5-point (`5×5`, `5×5×5` 최대 support). 새 계수를 학습한다.
- E0 axial NN; E1 Hadamard `sum(G_i**2)`; E2 self/nearest/second/diagonal의 양의 mass. 기본 exchange 합은 2d. Raw E1과 h,D 재스케일 동치 통제는 별도다.
- 고정 f, 매 단계 grad(s), synchronous H normalization, zero-H에서 이전 상태, periodic. div(v) 없음.
- 기본 binary EDT initialization, J=1,h=2,D=1. Binary 초기화는 별도 통제.
- Train 참조 원/구로 모든 모델을 보정한다. 기존 PBGIA-30은 원래 frozen c와 동일 train 재보정 c 모두 비교한다. Conventional도 raw와 train 보정을 함께 제공한다.
- Train 8원형, validation 8원형, primary test 16원형(각4회전×3noise). Train 참조2개. 원형 단위 split이며 과거 시험의 (차원,종류,R)와 분리했다. 작은 구 stress는 과거에 본 크기 범위이므로 신규 독립 시험으로 주장하지 않는다.

## 예산과 한계

차원/공유 generator family당12후보, 두 시작점. N=5,10,30에서 axial generator 각각6후보 추가. 고정 G의 free exchange6후보. 관측자별 geometry 및 orientation DE120평가/N를 독립 배정한다. 계수·실패 상태·실측 시간은 output에 남긴다. 이 유한 예산의 탐색을 전역 최적화로 부르지 않는다. 동일 walltime 민감도는 실제 탐색 prefix 비교이며 별도의 동일시간 optimizer 실험이 아니다. 더 넓은 G3, GPU autodiff, 실제 영상은 이번 실행 범위 밖이다.

`reference/`는 과거 baseline·split·원본 Colab 코드의 읽기 전용 복사본이다. 그 안의 기존 결정 규칙은 실행하거나 새 선택에 적용하지 않는다. GitHub 원본 검토 SHA와 추출 함수 provenance를 포함한다.
