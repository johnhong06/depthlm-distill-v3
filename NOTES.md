# depthlm-distill-v3 — 진행 기록

기록 규칙: 기능 하나·실험 하나마다 체크리스트와 실행 로그를 갱신한다. 결정은 "결정 기록"(V-번호)으로 남긴다.
설계 결정의 원 근거는 `paper/design_v3_corrections.md` (C-1~C-4) 와 옛 저장소 `depthlm-distill/NOTES.md` D-28~D-32.

## 체크리스트

- [x] 저장소 등록: 옛 저장소에서 필요한 것만 복사 (풀 v5·교사 라벨·라이브러리·실험 코드·공식 유틸·ref) — 2026-09-30
- [x] `03_build_cells_v3.py`: (B,k) 격자 9+2셀 + 반복 12셀 생성, 3개 풀 모두 **새 교사 질의 0 검증** — 2026-09-30
- [ ] `31_grid.py`·`32_decide.py` 등예산 비교 목록을 v3 행 단위(각 B 의 k arm 쌍)로 갱신
- [ ] 반복 셀 학습·집계 스크립트 (잡음 바닥 보고 전용, 본 격자와 분리)
- [ ] 저예산 1행(B=400: 3셀 + 반복)부터 학습·평가 — 시드 ≥3
- [ ] W1 파일럿: 값 공간(log-depth) 수식·대상 셀 확정 → 한 셀 실행
- [ ] gt+pseudo arm: λ·GT 픽셀 출처(NYUv2 gt 열 외 SUN RGB-D·KITTI 추출 여부)·대상 셀 확정
- [ ] hard 1자리 vs 2자리 ablation: 부분집합 2자리 재라벨(teacher_full) → `--decimals 2` 비교
- [ ] `run.sh` 를 cells_v3.json 기준으로 갱신 (그 전에는 사용 금지)
- [ ] 레시피-혼합: per-row 조건 학습 코드 (게이트: 도메인 격자 주장 ≥1건)

## 결정 기록

### V-1 (2026-09-30) 저장소 재등록 — 무엇을 가져오고 무엇을 버렸나 (사용자 지시)
- 가져온 것: `depthlm_uncertainty/`(라이브러리), `third_party/DepthLM_Official/`(공식 유틸, FAIR NC — LICENSE·NOTICE 유지),
  `pools/{indoor,outdoor,mixed}` (v5 순서 + 교사 라벨 44,800 px/풀), `ref/`(평가 픽셀·분포), `configs/`,
  실험 코드(01·02 풀 구축 이력용, 07 dedup, 11 라벨링, 12 교사 평가, 20 학습, 21 평가, 31·32 표·판정, 41·43 평가셋 구축), `run.sh`(v3 갱신 전 사용 금지).
- 버린 것: 옛 격자 rows_N*·arms.json 의 지위(파일은 풀 폴더에 남아 있으나 v3 는 cells_v3.json·rows_B* 를 쓴다),
  results_*·smoke·기존 진행 상태 전부. 옛 저장소는 참조 전용.
- 새로 만든 것: `03_build_cells_v3.py` — C-2 격자(B ∈ {400,1600,6400} × k ∈ {1,4,16}, N=B/k, 확장 25,600 k∈{4,16})
  + B=400 반복 셀 12개((25,16)×블록4, (100,4)×블록4, (400,1)×픽셀 인덱스 4). 실행 로그: 3개 풀 모두
  본 셀 11개·반복 12개 전부 기존 라벨 안 → **새 질의 0** (C-2 의 핵심 주장을 코드로 검증).

## 실행 로그

- 2026-09-30: `03_build_cells_v3.py --pool {mixed,indoor,outdoor}` — 각 풀 rows_B*_k*.parquet 11개,
  rep_*.parquet 12개, cells_v3.json 저장. "기존 라벨 밖 0" 전 셀 확인.
