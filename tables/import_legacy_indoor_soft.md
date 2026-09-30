# 옛 격자 결과 들여옴 — indoor soft (05_import_legacy.py, V-6)

출처: `/home/aims/Downloads/results_soft_indoor.zip` (sha256 앞 16: 047e159547e7d301)
매핑: {'N400_k1': 'B400_k1', 'N1600_k1': 'B1600_k1', 'N400_k4': 'B1600_k4', 'N6400_k1': 'B6400_k1', 'N1600_k4': 'B6400_k4', 'N400_k16': 'B6400_k16'}
제외(삭제된 예산 25,600): ['N1600_k16', 'N6400_k4']

들여온 파일:
- checkpoints/soft_N400_k1_indoor_f750/adapter_model.safetensors → checkpoints/soft_B400_k1_indoor_f750/
- eval/eval_soft_N400_k1_indoor_f750_large.parquet → eval/eval_soft_B400_k1_indoor_f750_large.parquet
- checkpoints/soft_N1600_k1_indoor_f750/adapter_model.safetensors → checkpoints/soft_B1600_k1_indoor_f750/
- eval/eval_soft_N1600_k1_indoor_f750_large.parquet → eval/eval_soft_B1600_k1_indoor_f750_large.parquet
- checkpoints/soft_N400_k4_indoor_f750/adapter_model.safetensors → checkpoints/soft_B1600_k4_indoor_f750/
- eval/eval_soft_N400_k4_indoor_f750_large.parquet → eval/eval_soft_B1600_k4_indoor_f750_large.parquet
- checkpoints/soft_N6400_k1_indoor_f750/adapter_model.safetensors → checkpoints/soft_B6400_k1_indoor_f750/
- eval/eval_soft_N6400_k1_indoor_f750_large.parquet → eval/eval_soft_B6400_k1_indoor_f750_large.parquet
- checkpoints/soft_N1600_k4_indoor_f750/adapter_model.safetensors → checkpoints/soft_B6400_k4_indoor_f750/
- eval/eval_soft_N1600_k4_indoor_f750_large.parquet → eval/eval_soft_B6400_k4_indoor_f750_large.parquet
- checkpoints/soft_N400_k16_indoor_f750/adapter_model.safetensors → checkpoints/soft_B6400_k16_indoor_f750/
- eval/eval_soft_N400_k16_indoor_f750_large.parquet → eval/eval_soft_B6400_k16_indoor_f750_large.parquet
