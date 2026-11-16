# 스모크 테스트 결과 (2026-11-16)

- `python -m pytest -q tests`: **4 passed**.
- `python code/fixed_iteration_checkpoints.py --dim 2 --size 40 --out smoke_2d`: 정상. N=5,10,20,30,40 checkpoint 기록.
- `python code/fixed_iteration_checkpoints.py --dim 3 --size 24 --out smoke_3d`: 정상. N=5,10,20,30,40 checkpoint 기록.
- 2D RMS 변화량: N=5 0.015845, 10 0.001845, 20 3.106e-5, 30 8.844e-7, 40 4.518e-8.
- 3D RMS 변화량: N=5 0.019126, 10 0.004132, 20 2.105e-4, 30 1.248e-5, 40 1.086e-6.
- **주의:** 이 두 synthetic sphere/circle와 소형 격자에 관한 코드 스모크일 뿐이다. 이 결과는 최적 N, 표면적/둘레 정확도, 커널 최적화 성공을 검증하지 않는다. 최종 결과로 사용 금지.
