# 自动驾驶场景 VLM 后训练与视频事件分析系统

基于 Qwen2.5-VL-3B 的自动驾驶场景理解与多模态后训练项目。

本项目面向 DriveLM / nuScenes 多视角驾驶问答任务，构建了从数据处理、LoRA-SFT、偏好对齐、对比实验、严格无泄漏评测到官方验证集提交的完整 VLM 后训练 pipeline。项目主线选择 `SFT -> ORPO`，同时保留 `SFT -> DPO` 和 `SFT -> GSPO-style` 作为对比实验。

> 项目名沿用 Qwen2VL-AutoDrive；实际实验基座为 `Qwen/Qwen2.5-VL-3B-Instruct`。

## 技术栈

- 基础模型：Qwen2.5-VL-3B-Instruct
- 训练方法：LoRA-SFT、ORPO、DPO、GSPO-style RLVR
- 训练框架：PyTorch、Transformers、PEFT、ms-swift
- 数据集：DriveLM-v1.1、nuScenes 多视角相机图像
- 评测指标：Exact Match、Token-F1、F1>=0.5、bootstrap 95% CI
- Demo：FastAPI、YOLOv8n、真实 nuScenes 视频事件触发

## 项目亮点

1. **完整后训练流程**：跑通 Base -> LoRA-SFT -> SFT->ORPO 主线，并保留 SFT->DPO、SFT->GSPO-style 作为 ablation。
2. **严格无泄漏评测**：构建 3,000 条 strict scene/frame/image no-leak 评测集，避免同一场景、关键帧或图像在训练和评测间重叠。
3. **多算法对比**：在同一 3,000 条评测集上对比 Base、SFT、DPO、ORPO、GSPO-style，最终选择 ORPO 作为稳定主模型。
4. **官方提交闭环**：生成 15,480 条 DriveLM 官方验证集问答推理输出，并提交到官方 Hugging Face Space。
5. **轻量系统 Demo**：保留云端 FastAPI、YOLOv8n 事件触发与 VLM 关键帧分析代码，用于展示视频事件分析原型。

## 数据构建

| 数据项 | 数量 / 说明 |
| --- | --- |
| 原始图像 | 约 4.3 万张 DriveLM / nuScenes 多视角道路图像 |
| SFT 训练集 | 3,000 条 |
| SFT 验证集 | 500 条 |
| 偏好训练集 | 1,000 对 chosen / rejected |
| 偏好验证集 | 200 对 |
| 严格评测集 | 3,000 条 strict scene/frame/image no-leak QA |
| 官方验证集输出 | 15,480 条 DriveLM QA predictions |

## 实验结果

最终结果文件：

```text
results/strict_scene_3000_metrics.csv
results/strict_scene_3000_leaderboard.md
docs/ablation_results.md
```

| Model | N | Exact | Token-F1 | Token-F1 95% CI | F1>=0.5 |
| --- | ---: | ---: | ---: | --- | ---: |
| Base | 3000 | 0.138 | 0.210 | [0.198, 0.221] | 0.160 |
| SFT | 3000 | 0.460 | 0.637 | [0.623, 0.651] | 0.664 |
| SFT->DPO | 3000 | 0.382 | 0.552 | [0.537, 0.567] | 0.553 |
| SFT->ORPO | 3000 | 0.471 | 0.647 | [0.632, 0.660] | 0.675 |
| SFT->GSPO-style | 3000 | 0.463 | 0.640 | [0.626, 0.654] | 0.666 |

结论：

- LoRA-SFT 是主要收益来源，显著提升模型对 DriveLM 问答格式、领域术语和答案分布的适配能力。
- ORPO 在 SFT 基础上进一步小幅提升，是本项目最终主模型。
- DPO 在当前偏好数据构造下退化，说明小规模 synthetic preference 对 DPO 不够稳定。
- GSPO-style 略优于 SFT，但未超过 ORPO，因此作为对比实验保留。

## 项目结构

```text
Qwen2VL-AutoDrive-PostTraining/
├── README.md
├── index.html
├── requirements.txt
├── configs/
├── docs/
│   ├── ablation_results.md
│   ├── final_results_20260616.md
│   └── github_upload_guide.md
├── results/
│   ├── official_submission_summary.json
│   ├── strict_scene_3000_leaderboard.md
│   └── strict_scene_3000_metrics.csv
├── scripts/
│   ├── run_p1_sft_after_data.sh
│   ├── run_p2_orpo_after_sft.sh
│   ├── run_p3_dpo_after_sft.sh
│   ├── run_p5_gspo_from_sft_ablation.sh
│   ├── run_strict_eval_3000.sh
│   ├── run_strict_eval_pref_models_after_current.sh
│   └── run_official_best_infer.sh
└── src/
    ├── data/
    ├── train/
    ├── eval/
    └── demo/
```

## 运行步骤

### 1. 环境配置

```bash
conda create -n autodrive-vlm python=3.11 -y
conda activate autodrive-vlm
pip install -r requirements.txt
pip install ms-swift qwen-vl-utils
```

### 2. SFT 训练

```bash
bash scripts/run_p1_sft_after_data.sh
```

### 3. ORPO 主实验

```bash
bash scripts/run_p2_orpo_after_sft.sh
```

### 4. DPO / GSPO-style 对比实验

```bash
bash scripts/run_p3_dpo_after_sft.sh
bash scripts/run_p5_gspo_from_sft_ablation.sh
```

### 5. 严格 3,000 条评测

```bash
bash scripts/run_strict_eval_3000.sh
bash scripts/run_strict_eval_pref_models_after_current.sh
```

### 6. 官方验证集推理

```bash
bash scripts/run_official_best_infer.sh
python -m src.eval.prepare_drivelm_submission
```

## 指标解释

DriveLM 是开放式驾驶问答任务，Token-F1 对答案格式、关键词覆盖和模板风格比较敏感。因此，本项目将大幅提升解释为：

```text
模型对 DriveLM 领域问答格式、答案分布和自动驾驶语言模式的适配显著提升。
```

而不是简单等价为视觉理解能力同等幅度提升。更稳妥的表述是：SFT 带来主要领域适配收益，ORPO 进一步提升偏好对齐稳定性，DPO/GSPO-style 用作对比实验帮助选择最终方法。


## 参考

- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- DriveLM: https://github.com/OpenDriveLab/DriveLM
- DriveLM Dataset: https://huggingface.co/datasets/OpenDriveLab/DriveLM
- Official Driving-with-Language Space: https://huggingface.co/spaces/AGC2024/driving-with-language-official
- nuScenes: https://www.nuscenes.org/

