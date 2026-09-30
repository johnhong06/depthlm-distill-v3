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
4. **격자 = (예산 B × 배분 k), N = B/k, 중첩(nested)**: B ∈ {400, 1600, 6400} × k ∈ {1, 4, 16} 9셀
   + 확장 B=25,600 은 k ∈ {4,16} (풀 한계로 k=1 결측 — 결측으로 보고). 이미지 순서 고정(v5), 픽셀 인덱스 0..k−1 공유.
   확장 규칙: "B 에서 k ∈ {1,4,16} 중 N ≤ 6,400 인 arm 전부" (C-2).
5. **배분 축은 서로 다른 이미지 수** — "서로 다름"의 기준은 v5 dedup(dHash 9×8 해밍 ≤ 10 & 32×32 회색조 상관 > 0.9
   쌍이 앞 N 장에 없음). 장면 수·도메인 비율은 arm 별 기술 통계로만 보고하고, **장면 다양성을 인과로 주장하지 않는다** (C-1).
6. **잡음 바닥**: 등예산 배분 차이는 B=400 반복 셀(이미지 블록·픽셀 인덱스, `pools/<pool>/rep_*.parquet`)의
   산포를 넘을 때만 주장한다. 본 셀은 접두사(중첩 유지), 반복 셀은 잡음 추정 전용 (C-2).
7. **사전등록**: 판정 규칙·평가 픽셀·체크포인트를 결과 전에 NOTES 에 적는다. CI 가 0 을 제외할 때만 주장.
   부정 결과 전부 보고. 인용은 검증된 것만(arXiv id), 수치는 로그에서만.
8. **레시피-혼합 게이트**: 균일 mixed vs 도메인별 최적 레시피 혼합 비교는 indoor·driving 격자에서
   32_decide '주장'이 최소 1건 나올 때만 실행 — 없으면 실행하지 않고 그 사실을 보고 (C-4d).

## 학습 신호 (C-3)

| 조건 | 정의 | 지위 |
|---|---|---|
| hard | CE(교사 greedy1 "x.d" + EOS) | 본 격자 |
| soft | 최종 로짓의 {0–9, ".", EOS} 지지집합 KL, 교사 greedy 경로 teacher-forcing | 본 격자 |
| W1 | soft 와 같은 최종 분포, 거리 척도만 값 공간(log-depth) Wasserstein | **한 셀 파일럿만** → 승격 여부 결정 |
| gt+pseudo | Loss_gt + Loss_pseudo 합산 | 별도 arm (규칙 1) — λ·GT 출처·대상 셀 구현 전 확정 |
| gt | CE(GT) 단독 | 참조 상한선 |

중간 레이어 feature 증류는 하지 않는다: 이종 교사·학생(Pixtral-12B ↔ Qwen2.5-VL-3B) 층 대응이 임의적이고,
학습 중 12B forward 가 필요해 예산 프레임이 깨지며, 원문 "no regression or regularization loss" 와도 어긋난다 (C-3a).

라벨 자릿수는 소수 첫째 + 중간값 복호 유지 (D-75: 상한 손실 ≤0.006, 둘째 자리 분포는 균등=잡음, 트리 3–4배 비용).
hard 1자리 vs 2자리 소형 ablation 은 실행해 공백을 닫는다 (`20_train_student.py --decimals 2`, 부분집합 재라벨 필요) (C-3c).

## 고정값

| 항목 | 값 |
|---|---|
| 교사 | `facebook/DepthLM` (Pixtral-12B), 초점 750, 마커, 템플릿 prefill, 소수 첫째 자리 트리(질량 ≈0.99) |
| 학생 | Qwen2.5-VL-3B-Instruct, LoRA r16 α32 q/k/v/o, AdamW lr 1e-4 cosine, bs 1, accum 8, 2 epoch, 시드 고정, 입력 초점 750 |
| 평가 지표 | δ1 = max(pred/gt, gt/pred) < 1.25 (공식 `third_party` metrics 와 동일 확인, C-4a) / AbsRel 보조 |
| 평가 세트 | iBims-1 · NYUv2 (실내), DDAD · nuScenes mini (주행), ETH3D (held-out). 원문 8개의 진부분집합, 사유는 C-4b |
| 평가 원칙 | 각 풀은 자기 도메인 세트로만 (교차 도메인은 도메인 갭을 측정하므로 계산하지 않음) |
| 통계 | 픽셀 쌍대 차이의 클러스터 부트스트랩 2,000회 95 % CI — 장면 클러스터(DDAD·nuScenes) / 이미지 클러스터(iBims-1·NYUv2) |

## 데이터

- **풀 (v5, 옛 저장소에서 그대로 복사, 재구축 금지)**: indoor(SUN RGB-D·NYUv2) / outdoor(KITTI) / mixed(50/50),
  각 6,400 장. 교사 라벨 44,800 px/풀 — **v3 격자(본 11셀 + 반복 12셀)는 전부 이 라벨 안, 새 질의 0**
  (`experiments/03_build_cells_v3.py` 실행 로그로 검증됨, `pools/<pool>/cells_v3.json`).
- 풀 소스는 교사 학습 소스(Argoverse2·Waymo·nuScenes·ScanNet++·Taskonomy·HM3D·Matterport3D)와 무교집합 —
  "교사가 본 적 없는 이미지에서 질의만으로 증류" 성립 (C-4b).
- mixed = 원문 정합 조건(DepthLM 은 혼합 학습, 주행 ≈49 %), indoor/outdoor 분리 = 도메인별 레시피 진단 조건 (C-4c).
- 이미지 실체는 `$DATA_ROOT` (로컬 `~/data`, H200 `/app/data`) — 저장소에는 좌표·라벨만 둔다.

## 실행

```bash
source ~/venv/main/bin/activate
python experiments/03_build_cells_v3.py --pool mixed   # (B,k) 격자·반복 셀 생성 + 라벨 커버 검증 (완료됨)
python experiments/20_train_student.py --cond soft --rows pools/mixed/rows_B400_k4.parquet --labels pools/mixed/teacher_labels.parquet --pools configs/pool_mixed.yaml --tag _B400_k4
python experiments/21_eval_student.py ...              # 셀별 평가
```

`run.sh` 는 옛 격자(arms.json, rows_N*) 기준이라 **v3 격자(cells_v3.json, rows_B*)로 갱신 전에는 쓰지 말 것** (NOTES 체크리스트).

## 남은 구현 (NOTES 체크리스트와 동기)

31_grid/32_decide 의 등예산 비교 목록을 행 단위로 갱신, 반복 셀 학습·집계 스크립트, W1 파일럿 수식·셀 확정,
gt+pseudo arm 의 λ·GT 추출, 2자리 ablation 재라벨, run.sh v3 갱신, 레시피-혼합 per-row 조건 학습.
