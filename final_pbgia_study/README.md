# PBGIA 최종 통합 검증 / final integrated study

최종 한국어 보고서는 [FINAL_RESEARCH_REPORT_KO.md](FINAL_RESEARCH_REPORT_KO.md)다. 실제 계산·테스트·새 디렉터리의 전체 재현을 완료했다. 큰 checkpoint와 기존 결과의 수정은 포함하지 않는다.

이 패키지의 권위 있는 범위는 2026-11-17 사용자의 최종 통합 검증 의뢰다. 기존 `CODEX_TASK_KO.md` 및 과거 branch는 모델·결과의 출처이며, 과거 최적화 목표를 새로 무제한 반복하는 지시가 아니다.

- 생성: current grad(s), positive NN sum, normalized synchronous alignment, N30, scalar에 div(v) 없음.
- 고정 관측자: 과거 PBGIA-30 learned 5×5 / 5×5×5 observer. TV에는 고정 G4.
- 제한 선택: train-only 보정, validation α 5개만 비교. 2D α=0 / 3D α=.5. frozen test 후 추가 후보 없음.
- clean test MARE: 선택 readout 0.893% / 0.732%; 같은 기준자 보정 전통 방법 0.738% / 0.563%.
- 주 결론: 공통장으로서 제한적 유용성은 확인했고, 추가 대규모 최적화는 종료한다.

[이론](MODEL_AND_PROJECTION_THEORY.md) · [픽셀 정보 한계](PIXEL_INFORMATION_LIMIT.md) · [결과 matrix](RESULTS_MATRIX.csv) · [종료 판단](FINAL_CONCLUSIONS.md) · [데이터 사전](DATA_DICTIONARY.md) · [재현](REPRODUCE.md).

그림을 포함한 12쪽 [PDF 보고서](output/FINAL_RESEARCH_REPORT_KO.pdf)도 제공한다.

```bash
python -m venv /tmp/pbgia-final-env
source /tmp/pbgia-final-env/bin/activate
python -m pip install -r requirements.txt
python code/reproduce.py --workdir /tmp/pbgia-final-new-run
```

`--workdir`는 존재하지 않는 경로여야 한다. 제공된 `output/`은 보존한다. 재현 driver는 새 디렉터리에 코드를 복사해 smoke→validation→freeze→test→진단→분석→최종 테스트를 실행하고 주요 수치를 현재 산출물과 비교한다. CPU one-thread 기준 약 4분이었으며 환경에 따라 달라진다.

원시 데이터는 [output](output), 그림은 [figures](output/figures), 설정과 smoke fixture는 [reference](reference)에 있다. 3D full spin checkpoint를 새로 저장하지 않았다. 작은 과거 regression fixture와 2D 투영 tensor만 압축 보관한다.
