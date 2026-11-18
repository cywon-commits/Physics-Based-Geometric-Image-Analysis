# Nature Communications 투고 전 저자 확인 목록

원고 안에서 빨간 테두리·노란 바탕의 박스("AUTHOR TO COMPLETE / CONFIRM")와 빨간 굵은 글씨(▶ [AUTHOR: …])로 표시한 곳입니다.

1. **공저자:** 공저자, 소속, ORCID를 추가합니다. 교신저자는 ORCID가 필수입니다.
2. **소속:** 주소의 우편번호(02447)와 교신 이메일(cywon@khu.ac.kr)을 확인합니다.
3. **AI 사용 기술 (Methods 마지막 절):** 실제로 쓴 도구, 버전, 사용 기간, 역할을 확인합니다.
   - Claude Opus 5.5: 사용 날짜
   - OpenAI Codex/GPT: 모델명과 날짜, 이전 계산 단계에서 쓴 경우
   - Google Gemini: 버전과 날짜, 초기 원고 초안
   - 해당 없는 항목은 지우고, 빠진 도구는 추가합니다.
4. **데이터 공개:** 저장소 릴리스를 Zenodo에 보관하고 DOI를 넣습니다. NC가 요구하는 Source Data 파일을 data/revision2/의 CSV로 만듭니다.
5. **제3자 데이터 사용 허가:** KerrPy의 Kerr 영상은 GPL-3.0이므로 그림 사용 가능 여부를 저널에 확인하거나 저자 허락을 받습니다. Argonne 영상(MIT)과 TU Wien 영상(CC-BY 4.0)은 출처 표기로 충분합니다.
6. **사사:** NRF 과제의 부처(MSIT 여부)를 확인하고, 다른 지원과 감사 대상을 추가합니다.
7. **저자 기여, 이해충돌:** 저자 기여(Author contributions)를 작성하고, 이해충돌(Competing interests) 문구를 확인합니다.
8. **참고문헌:** McCray et al.(2024)과 Zhao et al.(TU Wien 데이터셋)의 저자 목록 전체를 채웁니다. 현재는 "and others"로 되어 있습니다.
9. **그림 패널 표기:** NC 스타일은 굵은 소문자(a, b, …)를 씁니다. 현재 그림 파일은 "(a)" 형식이며, 최종본에서 style.py의 label() 함수 하나만 바꾸면 일괄 변경됩니다.
10. **투고 시스템 형식:** 초기 투고는 형식이 자유이고, 그림은 본문 끝에 모여 있습니다. 채택 후 Springer Nature LaTeX 템플릿으로 옮기면 됩니다.
