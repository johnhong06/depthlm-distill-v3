# depthlm-distill-v3 — 진행 기록

기록 규칙: 기능 하나·실험 하나마다 체크리스트와 실행 로그를 갱신한다. 결정은 "결정 기록"(V-번호)으로 남긴다.
설계 결정의 원 근거는 `paper/design_v3_corrections.md` (C-1~C-4) 와 옛 저장소 `depthlm-distill/NOTES.md` D-28~D-32.

## 체크리스트

- [x] 저장소 등록: 옛 저장소에서 필요한 것만 복사 (풀 v5·교사 라벨·라이브러리·실험 코드·공식 유틸·ref) — 2026-09-30
- [x] `03_build_cells_v3.py`: (B,k) 격자 9+2셀 + 반복 12셀 생성, 3개 풀 모두 **새 교사 질의 0 검증** — 2026-09-30
- [x] README 에 채울 격자 스켈레톤 생성 (3풀 × soft/hard 22행 + teacher·zero-shot 베이스라인 + 잡음 바닥 표 + 부수 실험 표 + 진행 순서) — 2026-09-30, 커밋 76c4ce2
- [x] `31_grid.py`·`32_decide.py` v3 갱신 — cells_v3.json 읽기, 각 예산 행 3쌍 배분 비교 + 9셀 손실 비교, `--floor` 잡음 바닥 게이트 — 2026-09-30
- [x] `04_extract_gt.py` 풀 픽셀 GT 추출 + 교사 대조 검증 (V-5: SUN RGB-D 스케일 수정 포함) — 2026-09-30
- [x] `20_train_student.py` 에 gthard·gtsoft 조건 (L = (1−λ)·L_pseudo + λ·CE(GT), λ=0.1, GT 있는 행만) — 2026-09-30
- [x] `21_eval_student.py` `--greedy_only` (반복 셀·추가 시드용, 트리 생략) — 2026-09-30
- [x] `33_noise_floor.py` 반복 셀 집계 → noise_floor_<pool>.json (32_decide --floor 입력) — 2026-09-30
- [x] `run.sh` v3 갱신 — cells_v3.json, 단위 = 셀×SEEDS(+REPLICATES 반복 셀), 시드>0·반복은 greedy 전용 평가 — 2026-09-30
- [x] 스모크: gtsoft·gthard GPU 학습, `--greedy_only` 평가, `bash run.sh smoke` 전체 파이프라인(30스텝+3px, 파싱 100 %) — 2026-09-30
- [ ] indoor B=400 행부터 H200 제출 (A1: soft SEEDS="0 1 2" REPLICATES=1 / A2: hard 동일 / A3: gtsoft / A4: gthard)
- [ ] W1 파일럿: 값 공간(log-depth) 수식·대상 셀 확정 → 한 셀 실행
- [ ] hard 1자리 vs 2자리 ablation: 부분집합 2자리 재라벨(teacher_full) → `--decimals 2` 비교
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

### V-2 (2026-09-30) H200 실행 계획 확정 (사용자 지시 + 분할 설계)
- **전 실험 H200**(베이스라인 포함). 풀 순서 **indoor → driving(outdoor) → mixed**. **B=25,600 행은 보류**(표에 deferred 표기, 비워둠).
- **작업 분할**: 1 작업 = (풀, 예산 행, 손실) — 풀당 격자 6작업 + 베이스라인 1작업, 각 ≈7–12 h (한 번에 20 h+ 돌리지 않음).
- **평가 절약(사전등록)**: 본 셀(시드 0)은 대형 세트(`ref/dist_*`), 추가 시드·반복 셀 12개는 산포만 필요하므로 소형 픽셀 세트(`ref/tree_px_*`)로 평가 — B=400 작업이 평가에 지배되는 것을 방지. 이 선택은 결과 확인 전에 고정.
- **표 정리**: 본 격자 soft/hard = 둘 다 Loss_pseudo 임을 README 에 명시(사용자 질문). GT arm(gt 단독, gt+pseudo)은 C-3b 대로 본 격자와 순위 비교하지 않되, 채울 칸이 보이도록 전용 표("GT arms — budget-external reference")로 승격. 대상 셀 = 각 풀의 격자 최적 셀(확정 후), λ·주행 GT 출처는 실행 전 고정.

### V-3 (2026-09-30) 소형 평가 철회, gt+pseudo 본 격자 병행, indoor 우선 빠른 채움 (사용자 지시·질문)
- **소형 평가 철회 (사용자 지적이 맞음)**: 소형 세트(~300 px/세트)의 반복당 표본 잡음 ~±0.03 δ1 은 판정 대상 배분 효과(D-27: ≤0.03)와 같은 크기라 잡음 바닥을 부풀린다. 옛 프로젝트도 같은 이유로 대형 세트를 승인받아 옮겨갔다. **모든 평가는 대형 세트**로 하되, 절약은 계산 내용에서: 본 셀(시드 0) = 전체 평가(CoV 트리 포함), 추가 시드·반복 셀 = **greedy 전용**(픽셀당 forward 1회, ≈6배 저렴 — 산포 추정에는 충분). `21_eval_student.py` 에 greedy 전용 플래그 필요(체크리스트).
- **gt+pseudo 를 본 격자와 병행 (사용자 의도 확인)**: "최적 셀에서 나중에"가 아니라 soft/hard 와 같은 셀에서 같이 진행. 표에는 각 셀의 세 번째 행(† 표시)으로 채우되, C-3b 의 지위는 유지 — 예산 밖 추가 정보이므로 **등예산 순위 비교에서는 제외**(soft/hard 만 판정). gt 단독(CE GT)은 부수 실험 표의 참조 행으로.
- **GT 커버리지 실측**: 현재 세 풀의 라벨 parquet 모두 gt>0 이 **0/44,800** — 옛 gt 조건은 풀 v1 라벨 기준이었다. 풀 픽셀 GT 추출 스크립트가 gt+pseudo 의 선행 조건(로컬 ~/data 의 SUN RGB-D·NYUv2·KITTI 깊이맵에서 추출, 커버리지 풀별 보고. 희소 GT 라 무라벨 픽셀 존재 — L_gt 는 GT 있는 행에만).
- **진행 방식**: indoor 부터, 싼 행부터 제출해 표를 빠르게 채움. 1작업 = (풀, 예산 행, 손실 ∈ {soft, hard, gt+pseudo}) — 풀당 9작업 + 베이스라인 1작업, 각 ≈7–12 h. λ 는 실행 전 고정(기본 후보: 합산 후 평균 정규화).

### V-4 (2026-09-30) λ = 0.1 고정, gt 병행 arm 을 gt+hard·gt+soft 로 분리, B=25,600 삭제 (사용자 지시)
- **λ**: 고전 KD(Hinton) 의 "정답(true-label) 항에 상당히 낮은 가중치가 최선" [arXiv:1503.02531]을 따라
  L = (1−λ)·L_pseudo + λ·L_gt, **λ = 0.1** 사전 고정(논문에 보편 수치는 없어 관행 수치 채택). 가중 평균 형태라 유효 학습률 유지.
- **gt 병행 arm 분리 (사용자 지적)**: 의사라벨 항이 hard(CE)/soft(KL) 두 형태이므로 합산 arm 도 **gt+hard, gt+soft** 두 조건.
  표의 † 행 분리(셀당 4행: soft/hard/gt+soft/gt+hard), 작업 = (풀, 예산 행, 손실 4종) → 풀당 12작업.
- **B=25,600 삭제**: deferred 가 아니라 격자에서 제거. `03_build_cells_v3.py` 에서 확장 제거 후 3개 풀 재생성(본 9셀,
  새 질의 0 재확인), rows_B25600_* 삭제. 필요해지면 C-2 고정 규칙으로 재추가.

### V-5 (2026-09-30) GT 추출 실행·검증 — SUN RGB-D 스케일 수정, 주행 커버리지 4.3 % 한계
- `04_extract_gt.py` 로 세 풀의 라벨 parquet gt 열을 채움. **커버리지**: indoor 37,423/44,800 (84 %), mixed 20,001 (45 %),
  outdoor **1,928 (4.3 %)** — kitti_raw(주행 풀의 71 %)는 velodyne 투영 미구현으로 GT 0, 나머지 KITTI 도 희소(11–17 %).
  → **주행 풀의 † arm 은 GT 신호가 행의 4 %뿐**이라 순수 arm 과 사실상 같을 수 있음을 보고서에 명시할 것.
- **검증(교사 라벨 대조)이 스케일 오류를 잡음**: 공식 curate_sunRGBD.py 의 /10000 으로 추출하면 SUN RGB-D 4개 서브셋 모두
  log(교사/GT) ≈ +0.20 (=1.25배, δ1 0.53–0.63) — 툴박스 규약(bitshift(v,−3)|bitshift(v,13) 후 /1000 = /8000 상당)과의 비율과 일치.
  툴박스 규약으로 수정 후 δ1 0.81–0.90, 편향 −0.02~−0.06 — NYUv2(0.875, −0.03)와 정합. **공식 저장소 코드가 항상 옳지는 않다.**
- KITTI 의 −0.17~−0.20 은 직접 깊이(dsel)와 disparity 변환(2012/2015)에서 동일 → 스케일이 아니라 교사의 원거리 과소예측(D-75·D-76 과 일치).
- 근사 기록: KITTI 2012/2015 depth = fx·0.54/disp (표준 기선, 프레임별 calib 미사용), GT 는 _10 프레임만. λ·규약은 gt_coverage.md 와 docstring 에.

## 실행 로그

- 2026-09-30: `03_build_cells_v3.py --pool {mixed,indoor,outdoor}` — 각 풀 rows_B*_k*.parquet (25,600 삭제 후 9개),
  rep_*.parquet 12개, cells_v3.json 저장. "기존 라벨 밖 0" 전 셀 확인 (V-4 재생성 포함).
- 2026-09-30: `04_extract_gt.py` 3개 풀 실행 — 커버리지 indoor 84 % / mixed 45 % / outdoor 4.3 %.
  교사 대조 검증으로 SUN RGB-D 스케일 오류 발견·수정 (V-5): 수정 후 SUN RGB-D 교사 δ1 0.81–0.90 (편향 −0.02~−0.06).
- 2026-09-30: v3 선행 구현 스모크 (로컬 GPU, 사용자 GPU 비움) —
  `20_train --cond gtsoft --steps 6` loss 1.458 저장 OK / `--cond gthard --steps 4` loss 2.060 OK
  (GT 있는 행 2 + 없는 행 1 로 두 경로 모두 통과; `r.gt`→`r["gt"]` pandas 메서드 충돌 버그 수정, 기존 gt 조건에도 있던 잠재 버그).
  `21_eval --greedy_only` 3세트 × 2px OK — greedy 복호 0.15–0.38 s/px (트리 포함 ≈1.5 s/px 대비 ≈6배 저렴, V-3 가정 실측 확인).
  run.sh v3: bash -n OK, UNITS 생성 검증 (B=400 행 + SEEDS "0 1 2" + REPLICATES=1 → 21단위).
