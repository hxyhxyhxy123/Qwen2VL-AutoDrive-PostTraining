# Ablation Results

This document records the comparison experiments used to select the final
post-training strategy. All models are evaluated on the same 3,000-sample
strict scene/frame/image no-leak DriveLM QA split.

## Compared Methods

| Method | Start Point | Purpose | Script |
| --- | --- | --- | --- |
| Base | Qwen2.5-VL-3B-Instruct | Zero/few-shot baseline under the project prompt | `scripts/run_strict_eval_3000.sh` |
| SFT | Base + LoRA-SFT | Adapt to DriveLM QA style and driving-language distribution | `scripts/run_p1_sft_after_data.sh` |
| SFT->DPO | SFT adapter | Pairwise preference optimization ablation | `scripts/run_p3_dpo_after_sft.sh` |
| SFT->ORPO | SFT adapter | Main preference-aligned model | `scripts/run_p2_orpo_after_sft.sh` |
| SFT->GSPO-style | SFT adapter | Small-scale RLVR-style ablation | `scripts/run_p5_gspo_from_sft_ablation.sh` |

## Main Table

| Model | N | Exact | Token-F1 | Token-F1 95% CI | F1>=0.5 | Avg Latency |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Base | 3000 | 0.138 | 0.210 | [0.198, 0.221] | 0.160 | 0.38 s |
| SFT | 3000 | 0.460 | 0.637 | [0.623, 0.651] | 0.664 | 0.59 s |
| SFT->DPO | 3000 | 0.382 | 0.552 | [0.537, 0.567] | 0.553 | 0.66 s |
| SFT->ORPO | 3000 | 0.471 | 0.647 | [0.632, 0.660] | 0.675 | 0.57 s |
| SFT->GSPO-style | 3000 | 0.463 | 0.640 | [0.626, 0.654] | 0.666 | 0.58 s |

## Interpretation

- LoRA-SFT provides the largest improvement. The main gain should be
  interpreted as DriveLM domain, answer-style, and task-format adaptation.
- ORPO is the best final model in this setup. It improves over SFT by
  +0.009 Token-F1 and +0.011 Exact Match.
- DPO underperforms SFT. This suggests that the synthetic rejected responses
  and the small preference set are not strong enough for stable DPO gains.
- SFT->GSPO-style slightly improves over SFT but does not surpass ORPO. It is
  retained as an ablation rather than the main result.

## Result Files

- `results/strict_scene_3000_metrics.csv`
- `results/strict_scene_3000_leaderboard.md`

