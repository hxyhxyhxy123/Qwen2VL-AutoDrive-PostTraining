# AutoDrive-VLM-RLBench Final Results

## Main Evaluation

Final internal evaluation uses `strict_scene_3000_all_models_20260615_200358`, a 3,000-sample strict scene/frame/image no-leak DriveLM QA split.

| Model | N | Exact | Token-F1 | Token-F1 95% CI | F1>=0.5 |
| --- | ---: | ---: | ---: | --- | ---: |
| Base | 3000 | 0.138 | 0.210 | [0.198, 0.221] | 0.160 |
| SFT | 3000 | 0.460 | 0.637 | [0.623, 0.651] | 0.664 |
| SFT->DPO | 3000 | 0.382 | 0.552 | [0.537, 0.567] | 0.553 |
| SFT->ORPO | 3000 | 0.471 | 0.647 | [0.632, 0.660] | 0.675 |
| SFT->GSPO-style | 3000 | 0.463 | 0.640 | [0.626, 0.654] | 0.666 |

Best internal model: SFT->ORPO.

## Official Submission

- Hugging Face repo: `Xuran188/AutoDrive-Qwen2.5VL-ORPO`
- Official submission summary: `results/official_submission_summary.json`
- Official validation predictions: 15,480 DriveLM QA outputs
- Team name: `NUECS`
- Current official Space status at last check: submitted and queued

## Lightweight Demo Metrics

Cloud demo was kept as code, with heavy video outputs removed during cleanup.

- FastAPI service: `src/demo/cloud_event_service.py`
- YOLO event demo: `src/demo/yolo_video_event_demo.py`
- VLM keyframe analyzer: `src/demo/vlm_frame_analyzer.py`
- Real nuScenes clip: 12 FPS, 120 frames
- YOLOv8n average detection latency: about 15.54 ms/frame
- CV processing speed: about 44.95 FPS
- ORPO-VLM keyframe analysis: about 4.4 s/frame
- Peak VLM memory in demo: about 8.3 GB
