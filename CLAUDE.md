# depthlm-distill-v3 — 프로젝트 규칙

상위 `Jihyuck/CLAUDE.md` 의 공통 규칙이 그대로 적용된다. 이 프로젝트는 `depthlm-distill/`·`depthlm-distill-h200/` 의
**설계 재점검(2026-09-30, C-1~C-4)에 따른 재등록**이다. 옛 저장소는 참조 전용(코드·라벨·결과를 읽되 수정하지 않는다).
설계 결정의 근거 전문은 `paper/design_v3_corrections.md` (C-1 축, C-2 격자, C-3 학습 신호, C-4 데이터·평가).

## 연구 질문

> 교사 질의 예산 B 가 고정될 때, VLM 깊이 모델(DepthLM 12B)의 절대 깊이 능력을 GT 0장으로 소형 학생(≤3B, LoRA)에
> 옮기려면 예산을 **어떻게 배분**하고(서로 다른 이미지 수 N × 이미지당 픽셀 k, N = B/k) **어떤 신호**로
> (hard CE / soft 자릿수 분포 KL) 학습해야 하는가?
> DepthLM Finding 4("1픽셀/이미지, 다양성 > 밀도")는 스케일링 곡선(Fig. 4c)의 해석이지 고정 예산 대조 실험이 아니며
> [arXiv:2509.25413], 그 검정은 원문에 없다 — 우리가 원문이 안 본 조건(고정 예산·의사라벨·소형 학생)에서 검정한다.

## 절대 규칙 (C-1~C-4 반영)

1. **GT 는 채점과 교사 신뢰도 점검에만. 본 격자 학습에 쓰지 않는다.** GT 합산 손실(Loss_gt+Loss_pseudo)은
   명시된 별도 arm 에서만 — 예산 밖 추가 정보로 표기하고 본 격자와 순위 비교하지 않는다 (C-3b).
2. **ETH3D 는 완전 held-out.** 풀에 없고, 어떤 선택에도 쓰지 않는다.
3. **공정 비교**: 조건 간 다른 것은 정확히 하나. 같은 픽셀 목록·같은 교사 라벨·같은 스텝·초기화·셔플 시드·
   라벨 형식(소수 첫째)·초점 750·중간값 복호.
4. **격자 = (예산 B × 배분 k), N = B/k, 중첩(nested)**: B ∈ {400, 1600, 6400} × k ∈ {1, 4, 16} 9셀.
   B=25,600 확장은 삭제(V-4) — 더 큰 예산이 필요해지면 고정 규칙 "B 에서 k ∈ {1,4,16} 중 N ≤ 6,400 인 arm 전부"(C-2)로
   추가. 이미지 순서 고정(v5), 픽셀 인덱스 0..k−1 공유.
5. **배분 축은 서로 다른 이미지 수** — "서로 다름"의 기준은 v5 dedup(dHash 9×8 해밍 ≤ 10 & 32×32 회색조 상관 > 0.9
   쌍이 앞 N 장에 없음). 장면 수·도메인 비율은 arm 별 기술 통계로만 보고하고, **장면 다양성을 인과로 주장하지 않는다** (C-1).
6. **잡음 바닥**: 등예산 배분 차이는 B=400 의 두 반복 — 부분집합(이미지 블록·픽셀 인덱스, `pools/<pool>/rep_*.parquet`,
   b0·p0 은 본 셀과 같아 따로 안 돌림)과 시드(0·1·2, 초기화·드롭아웃·데이터 순서) — 중 큰 산포를 넘을 때만 주장한다
   (`33_noise_floor.py` → `32_decide.py --floor`). 본 셀은 접두사(중첩 유지), 반복은 잡음 추정 전용 (C-2, V-7).
7. **사전등록**: 판정 규칙·평가 픽셀·체크포인트를 결과 전에 NOTES 에 적는다. CI 가 0 을 제외할 때만 주장.
   부정 결과 전부 보고. 인용은 검증된 것만(arXiv id), 수치는 로그에서만.
8. **레시피-혼합 게이트**: 균일 mixed vs 도메인별 최적 레시피 혼합 비교는 indoor·driving 격자에서
   32_decide '주장'이 최소 1건 나올 때만 실행 — 없으면 실행하지 않고 그 사실을 보고 (C-4d).

## 학습 신호 (C-3)

| 조건 | 정의 | 지위 |
|---|---|---|
| hard | CE(교사 greedy1 "x.d" + EOS) | 본 격자 |
| soft | 최종 로짓의 {0–9, ".", EOS} 지지집합 KL, 교사 greedy 경로 teacher-forcing | 본 격자 |
| gt+hard | (1−λ)·CE(교사 답) + λ·CE(GT), **λ = 0.1 사전 고정** — 고전 KD 는 정답(GT) 항에 상당히 낮은 가중치가 최선 [arXiv:1503.02531], 보편 수치는 없어 관행값 0.1 | † 병행 arm (규칙 1) — 같은 셀에서 학습·보고하되 등예산 순위 제외 |
| gt+soft | (1−λ)·KL(교사 분포) + λ·CE(GT), λ = 0.1 동일 | † 병행 arm (규칙 1) — 위와 동일 |
| W1 | soft 와 같은 최종 분포, 거리 척도만 값 공간(log-depth) Wasserstein | **한 셀 파일럿만** → 승격 여부 결정 |
| gt | CE(GT) 단독 | 참조 상한선 (부수 실험) |

GT 는 L_gt 항에서 GT 가 있는 행에만 적용 — 커버리지 indoor 84 % / mixed 45 % / outdoor 4.3 % (`pools/<pool>/gt_coverage.md`).
GT 는 **카메라까지의 유클리드 거리**(공식 curate·평가 세트와 같은 정의, 깊이맵 z 를 풀 intrinsics 로 변환, V-12)이고,
교사 라벨처럼 첫째 자리 **절사**(`trunc1`, float32 오차 보정 1e-4). 가중 평균 (1−λ)/λ 형태라 유효 학습률이
순수 arm 과 비교 가능하게 유지된다. SUN RGB-D 깊이는 툴박스 규약(비트시프트, /1000) — 공식 curate 의 /10000 은 1.25배 틀림 (V-5).

중간 레이어 feature 증류는 하지 않는다: 이종 교사·학생(Pixtral-12B ↔ Qwen2.5-VL-3B) 층 대응이 임의적이고,
학습 중 12B forward 가 필요해 예산 프레임이 깨지며, 원문 "no regression or regularization loss" 와도 어긋난다 (C-3a).

라벨 자릿수는 소수 첫째 + 중간값 복호 유지 (D-75: 상한 손실 ≤0.006, 둘째 자리 분포는 균등=잡음, 트리 3–4배 비용).
hard 1자리 vs 2자리 소형 ablation 은 실행해 공백을 닫는다 (`20_train_student.py --decimals 2`, 부분집합 재라벨 필요) (C-3c).

## 고정값

| 항목 | 값 |
|---|---|
| 교사 | `facebook/DepthLM` (Pixtral-12B), 초점 750, 마커, 템플릿 prefill, 소수 첫째 자리 트리(질량 ≈0.99) |
| 학생 | Qwen2.5-VL-3B-Instruct, LoRA r16 α32 q/k/v/o, AdamW lr 1e-4 cosine, bs 1, accum 8, 2 epoch, 시드 고정, 입력 초점 750 |
| 학생 입출력 | **교사와 같은 DepthLM 공식 형식** (V-8): 공식 질의 → 템플릿 `<think> The point is around ` 채움 → 숫자(소수 첫째) + ` meters`. 손실은 숫자·종료 토큰에만. 학습·평가 모두 `mm_token_type_ids` 를 넘겨 이미지 3차원 위치(M-RoPE) 사용 (V-9 — 빠지면 조용히 1차원 위치로 떨어진다) |
| 평가 지표 | δ1 = max(pred/gt, gt/pred) < 1.25 (공식 `third_party` metrics 와 동일 확인, C-4a) / AbsRel 보조 |
| 평가 세트 | iBims-1 · NYUv2 (실내), DDAD · nuScenes mini (주행), ETH3D (held-out). 원문 8개의 진부분집합, 사유는 C-4b |
| 평가 원칙 | 각 풀은 자기 도메인 세트로만 (교차 도메인은 도메인 갭을 측정하므로 계산하지 않음) |
| 통계 | 픽셀 쌍대 차이의 클러스터 부트스트랩 2,000회 95 % CI — 장면 클러스터(DDAD·nuScenes, **`ref/scenes_*.parquet` 에서만** — 모르는 이미지면 중단) / 이미지 클러스터(iBims-1·NYUv2) (V-12) |
| 채점 | **모든 평가 픽셀** — 파싱 실패·"0.0" 답은 0.05 m 답(=오답)으로. 학생·교사·zero-shot 같은 규칙. 예측 = 첫째 자리 절사 + 0.05 (V-12) |
| 완료 판정 | 어댑터 폴더의 `DONE`(마지막 스텝 뒤에만 기록)만 믿는다 — 중간 저장 없음, DONE 없는 셀은 평가·집계하지 않는다 (V-12) |

## 데이터

- **풀 (v5, 옛 저장소에서 그대로 복사, 재구축 금지)**: indoor(SUN RGB-D·NYUv2) / outdoor(KITTI) / mixed(50/50),
  각 6,400 장. 교사 라벨 44,800 px/풀 (+ gt 열) — **v3 격자(본 9셀 + 반복 12셀)는 전부 이 라벨 안, 새 질의 0**
  (`experiments/03_build_cells_v3.py` 실행 로그로 검증됨, `pools/<pool>/cells_v3.json`).
- **풀 좌표 = 라벨 좌표** (V-10): 학생 학습 픽셀(pool.jsonl)과 교사가 본 픽셀(라벨의 pixel_x/y_orig)이 같아야 한다. mixed 는 41,096 행이
  달라서 `08_align_pool_coords.py` 로 맞췄고, `20_train` 이 시작 전에 확인하고 다르면 중단한다. 풀을 다시 만들면 반드시 이 검사를 거칠 것.
- **서로 다른 이미지 수** (V-11, `tables/cell_stats.md`): N ≤ 1,600 은 전부 서로 다름, N=6,400 은 outdoor 5,035 · mixed 5,651 — **사전등록 확정**: 그 셀은 "6,400 (서로 다른 5,035 / 5,651)"로 표기하고 배분 주장은 서로 다른 이미지 수로 서술.
- **옛(v2) 학생 결과는 본 격자에 쓰지 않는다** (V-8·V-9: 다른 질의 문장 + 학습 중 1차원 이미지 위치). `05_import_legacy.py` 는
  `results/v2_legacy/` 에만 넣고, README 의 v2 비교표(두 수정의 합친 효과)에만 쓴다. **교사 쪽 자산(라벨·자릿수 분포·교사 행)은 그대로 유효.**
- **교사 행**: `ref/dist_*` 의 교사 기록에서 계산(`34_baselines.py`), 학생과 같은 픽셀·절사+0.05·유클리드 GT. H200 주행 팩과 픽셀 일치 확인 (V-7).
- 풀 소스는 교사 학습 소스(Argoverse2·Waymo·nuScenes·ScanNet++·Taskonomy·HM3D·Matterport3D)와 무교집합 —
  "교사가 본 적 없는 이미지에서 질의만으로 증류" 성립 (C-4b).
- mixed = 원문 정합 조건(DepthLM 은 혼합 학습, 주행 ≈49 %), indoor/outdoor 분리 = 도메인별 레시피 진단 조건 (C-4c).
- 이미지 실체는 `$DATA_ROOT` (로컬 `~/data`, H200 `/app/data`) — 저장소에는 좌표·라벨만 둔다.

## 실행

모든 실험은 H200 (이슈 한 줄 명령, GPU 할당량 7). 제출 순서·명령은 README "Order of work" 가 정본이다.
```bash
bash run.sh smoke                                   # 새 저장소 점검
bash run.sh baseline mixed                          # zero-shot 행 (네 세트, greedy) + 교사 행
bash run.sh grid indoor soft CELLS=B400_k1,B400_k4,B400_k16 SEEDS=0,1,2 REPLICATES=1
bash run.sh grid indoor gtsoft CELLS=B400_k1,B400_k4,B400_k16      # † arm
```
결과 zip 을 받으면 로컬에서: `results/` 에 풀기 → 표 채움 → `33_noise_floor.py` → `32_decide.py --floor`.
옛(v2) 저장소 zip 은 `05_import_legacy.py` 로 `results/v2_legacy/` 에만 (비교표 전용).
재현용 표는 `tables/` 에 복사해 커밋한다 (`results/` 는 git 밖).

## 남은 일 (NOTES 체크리스트와 동기)

H200 제출(README 순서), W1 파일럿 수식·셀 사전등록, 2자리 ablation 재라벨, 레시피-혼합 per-row 조건 학습(게이트 통과 시).
