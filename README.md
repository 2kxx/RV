# Auto-Rubric as Reward - MVP实验指南

<p align="center">
  <a href="https://arxiv.org/abs/2605.08354v1"><img src="https://img.shields.io/badge/arXiv-ARR-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/OpenEnvisionLab/Auto-Rubric-as-Reward"><img src="https://img.shields.io/badge/HuggingFace-ARR-FFD21E?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
  <a href="https://openenvision.github.io/AutoRubric-as-Reward/"><img src="https://img.shields.io/badge/Project-Website-2F6FED?logo=googlechrome&logoColor=white" alt="Project Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <strong>Auto-Rubric as Reward: 从隐式偏好到显式多模态生成标准</strong>
</p>

## 🚀 快速开始 - MVP实验

### 1. 环境设置

```bash
# 克隆项目
cd /path/to/RewardVerse

# 创建conda环境
conda create -n rewardverse python=3.10 -y
conda activate rewardverse

# 安装依赖
bash env_setup.sh

# 或者手动安装
pip install -r requirements.txt
pip install -e .

# 创建必要的目录
mkdir -p data rubric_pipeline/rubrics runs/v_arr_mvp
```

### 2. 获取Gemini API密钥

MVP实验使用Gemini 2.5 Pro作为视觉语言模型评判器。您需要：

1. 访问 [Google AI Studio](https://aistudio.google.com/)
2. 创建API密钥
3. 设置环境变量：

```bash
export OPENAI_API_KEY="your_gemini_api_key_here"
```

### 3. 准备数据

确保您的视频数据位于正确的位置。示例数据格式：

```json
{
  "id": "varr_seed_001",
  "prompt": "A red sports car drifts around a mountain bend with dust trailing the rear tires.",
  "win_video": "data/v_arr_mvp/videos/varr_seed_001_win.mp4",
  "lose_video": "data/v_arr_mvp/videos/varr_seed_001_lose.mp4",
  "source": "v_arr_mvp",
  "split": "seed"
}
```

将视频文件放置在 `data/v_arr_mvp/videos/` 目录下。

### 4. 运行MVP实验管道

```bash
# 运行完整的MVP实验管道
bash scripts/v_arr_mvp/run_mvp_pipeline.sh

# 或者手动设置参数运行
CONFIG_PATH="rubric_pipeline/config/v_arr_mvp_t2v.yaml" \
BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
MODEL="gemini-2.5-pro" \
CONCURRENCY="2" \
SEED_DATASET="examples/v_arr_mvp_seed_pairwise.json" \
TEST_DATASET="examples/v_arr_mvp_test_pairwise.json" \
RUBRICS_FILE="rubric_pipeline/rubrics/v_arr_mvp_t2v_v1.txt" \
ARTIFACT_DIR="runs/v_arr_mvp" \
bash scripts/v_arr_mvp/run_mvp_pipeline.sh
```

### 5. 实验步骤说明

MVP实验管道包含5个主要步骤：

1. **Auto-Rubric生成 + 成对评估 (RQ1)** - 自动生成评分标准并进行评估
2. **零样本基线 (RQ4)** - 运行零样本基准测试
3. **位置偏差分析 (RQ3)** - 分析模型的位置偏差
4. **评分标准分类 (RQ2)** - 对生成的评分标准进行分类
5. **构建MVP决策报告** - 生成完整的实验报告

### 6. 构建 Visual-Quality 4 变体数据集（一键脚本）

为对比四种不同输入配置（`Prompt+y⁺` / `Prompt+y⁺+y⁻` / `y⁺` / `y⁺+y⁻`），仓库内置了一键构建脚本：

```bash
bash scripts/v_arr_mvp/oneclick_build_vq_mvp.sh
```

脚本要点：

- **数据源**：自动扫描根目录下所有 `merged_evaluate_results_with_path_part*.jsonl`（part1–part5）。
- **筛选规则（严格 dominance）**：同一 prompt 下选取一对样本，要求 winner 在 `Aesthetics-Visual_Quality` 的 **Physics / Rendering_Quality / Temporal_Consistency** 三个子维度上**全部严格优于** loser。
- **样本数**：默认 30 对（`N_PAIRS` 环境变量可改）。
- **prompt 翻译**：使用离线缓存 [`data/v_arr_mvp/visual_quality_mvp/_prompt_translations.json`](data/v_arr_mvp/visual_quality_mvp/_prompt_translations.json)（zh→en，已预先填好 30 条），**全程无需任何翻译 API**。
- **输出（4 份 JSON，与测试时格式完全一致）**：

  | 文件 | 输入构成 | 模式 |
  | --- | --- | --- |
  | `v1_prompt_ywin.json` | Prompt + y⁺ | pointwise（Exemplar Rubric 配置） |
  | `v2_prompt_pair.json` | Prompt + y⁺ + y⁻ | pairwise（ARR 原版配置） |
  | `v3_ywin_only.json`   | 仅 y⁺ | pointwise（无 prompt） |
  | `v4_pair_only.json`   | y⁺ + y⁻ | pairwise（无 prompt） |

  以上文件统一写入 `data/v_arr_mvp/visual_quality_mvp/`，并附带 `manifest.json` 记录元信息。

- **自检**：脚本最后一步会校验严格三维 dominance、score-gap 分布、task_type 分布与 top-5 winner→loser 方法对，任何缺失/违例都会显式报错。

可用环境变量（全部可选）：

```bash
N_PAIRS=50 \                                      # 改成 50 对
OUT_DIR=data/v_arr_mvp/vq_mvp_v2 \                # 换输出目录
INPUT_GLOB='merged_evaluate_results_with_path_part[12].jsonl' \
TRANSLATION_CACHE=my_zh2en.json \                 # 自带翻译缓存
bash scripts/v_arr_mvp/oneclick_build_vq_mvp.sh
```

> 若 `N_PAIRS` 调大后挖出的中文 prompt 不在缓存里，脚本会精确打印缺失项并退出，把它们追加到 `_prompt_translations.json` 后重跑即可。

### 7. 查看结果

实验完成后，查看生成的报告：

```bash
cat runs/v_arr_mvp/mvp_report.md
```

报告包含：
- 自动生成的评分标准文本
- 评估结果和统计分析
- 位置偏差分析
- 评分标准分类结果
- 实验结论和建议

## 📊 MVP实验配置

主要的配置文件：`rubric_pipeline/config/v_arr_mvp_t2v.yaml`

关键配置参数：
- `base_url`: Gemini API端点
- `model_name`: 使用的模型（Gemini 2.5 Pro）
- `task_description`: 任务描述和评分标准要求
- `query_specific_generate_number`: 每个查询生成的评分标准数量
- `categories_number`: 分类数量

## 🎯 自定义实验

### 使用自定义数据集

1. 准备JSON格式的数据集文件
2. 更新配置文件中的路径
3. 运行实验管道

### 使用不同的VLM

修改配置文件中的 `base_url` 和 `model_name`：

```yaml
# 使用OpenAI GPT
base_url: "https://api.openai.com/v1"
model_name: "gpt-4-vision-preview"

# 使用本地Qwen3-VL
base_url: "http://localhost:8000/v1"
model_name: "Qwen3-VL-8B-Instruct"
```

## 🔧 故障排除

### 常见问题

1. **API密钥错误**：确保 `OPENAI_API_KEY` 环境变量已设置
2. **视频文件缺失**：检查视频文件路径是否正确
3. **依赖问题**：重新运行 `bash env_setup.sh`
4. **内存不足**：减少 `CONCURRENCY` 参数值

### 日志查看

实验日志保存在 `runs/v_arr_mvp/` 目录下：
- `rubric_judge.log` - 评分标准生成日志
- 各个步骤的JSON输出文件

## 📁 项目结构

```
RewardVerse/
├── rubric_pipeline/       # 自动评分标准管道
│   ├── config/           # 配置文件
│   ├── rubrics/          # 生成的评分标准文件
│   └── ...
├── scripts/v_arr_mvp/    # MVP实验脚本
│   ├── run_mvp_pipeline.sh                # 端到端 MVP 评估流水线
│   ├── oneclick_build_vq_mvp.sh           # 一键构建 VQ 4 变体数据集
│   └── build_visual_quality_mvp_dataset.py
├── examples/             # 示例数据
├── data/
│   └── v_arr_mvp/
│       └── visual_quality_mvp/            # 4 变体数据集 + 翻译缓存
├── runs/                 # 实验运行结果
└── docs/                 # 文档
```

## 📚 相关资源

- [论文](https://arxiv.org/abs/2605.08354v1)
- [项目网站](https://openenvision.github.io/AutoRubric-as-Reward/)
- [Hugging Face模型](https://huggingface.co/OpenEnvisionLab/Auto-Rubric-as-Reward)

## 🙏 致谢

本项目基于以下工作构建：
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [diffusers](https://github.com/huggingface/diffusers)
- [OpenJudge](https://github.com/agentscope-ai/OpenJudge)

## 📄 引用

```bibtex
@misc{tian2026autorubricrewardimplicitpreferences,
      title={Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria}, 
      author={Juanxi Tian and Fengyuan Liu and Jiaming Han and Yilei Jiang and Yongliang Wu and Yesheng Liu and Haodong Li and Furong Xu and Wanhua Li},
      year={2026},
      eprint={2605.08354},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.08354}, 
}
```
