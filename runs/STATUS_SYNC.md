# RewardVerse · 全景同步文档（Status Sync）

> 文档时间戳：**2026-06-01 (v4)**　·　LLM 后端：`gemini-3.1-pro`（Tencent Venus llmproxy）
> 维护人：jacuzzitang　·　位置：[runs/STATUS_SYNC.md](./STATUS_SYNC.md)
>
> 本文目的是给协作者一份**单文件同步清单**：当前已实现 / 未实现的方法、所有 benchmark、baseline 设置、最新一轮结果。
>
> **v4 主要变更（vs v3）**：
> - 🗑️ 物理删除 `rewardmodel_7d.py` / `build_sota_table.py` / `sota_manifest.example.json` / `sota_runbook.md` / 旧版 `aggregation_VQ.json` 等无用产物（18→7 hierarchical judge 路线整组下线 —— 后续其他 benchmark 更需要子维度信号，所以不再保留这条 hierarchical 表格路径）。
> - 🗑️ 移除 v3 §8 "VGRB 上 B1abc 系统跑（E5）" 的工作项（不再做外部 VGRB 上的 zero-shot Gemini 系统跑；脚本仅保留作为内部 18 维 RQ4 对照）。
> - 🔁 **UnifiedReward-7B 状态翻转**：v3 ❌ 已弃用 → v4 🟡 **待重启**（见 §4.2 / §8.3）。
> - 🔁 **VGRB FULL 状态翻转**：v3 "Gemini 阶段暂缓" → v4 明确归位为 "**未来在自家 rubric MLLM reward model 上跑**"（见 §5.2.2 / §8.3 M1）。
> - 🧹 移除所有 `min / median / votes` 决策细节，仅在 §6 留一行历史脚注。
> - 🧹 移除 VideoFeedback-200 相关全部段落（更早已下架）。
>
> 详细数字仍以以下"事实之源"为准：
> - [runs/ALL_RESULTS_SUMMARY.md](./ALL_RESULTS_SUMMARY.md)（18 维内部 pointwise 结果）
> - [runs/v_arr_mvp_dim/all_dims_summary.md](./v_arr_mvp_dim/all_dims_summary.md)（自动刷新的 17 维子表）

---

## 0. TL;DR（一页摘要）

| 领域 | 状态 |
|---|---|
| **18 维 pointwise rubric**（内部 v_arr_mvp test set） | ✅ **18/18 跑通**，平均 PLCC ≈ 0.65，全面优于 VideoScore-1.1 / VideoScore-2 |
| **VideoGen-RewardBench (200 对 smoke, 4 axis × listwise+per_rubric+aggregation)** | ✅ 已完成：TA acc=0.49 / Overall acc=0.445 / VQ ~0.22（被 GT same=55% 主导） |
| **VideoGen-RewardBench (FULL ~25.2k 对)** | 🟡 **当前 Gemini 阶段不跑**；**未来训完自家 rubric MLLM reward model 后必跑（主战场 benchmark）**（详见 §5.2.2 / §8.3 M1） |
| **3 种 aggregation 公式**（保留版） | ✅ `mean / weighted_mean / lp4` + `listwise_direct` baseline；`min / median / votes` 已实证证伪并删除（见 §6） |
| **UnifiedReward-7B baseline** | 🟡 **待重启**（推理脚本已删；模型权重仍在；重跑需先选定方案，见 §4.2 + §8.3） |
| **VS-v1.1 / VS-2 在 VGRB 上的 pairwise wrapper** | 🟡 待补（pointwise→pairwise 包装器，详见 §8.1 E2） |

---

## 1. 项目结构（与本文相关的目录）

```
RewardVerse/
├── rubric_pipeline/           # rubric 生成 + 评分 (graders / generator / models)
│   ├── config/v_arr_mvp_<dim>_t2v.yaml      # 18 维生产配置
│   └── rubrics/v_arr_mvp_<dim>_v2.txt       # 18 维生产 rubric
├── data/v_arr_mvp/<dim>_mvp/                # 18 维 MVP test set
├── scripts/v_arr_mvp/                       # 主线脚本
├── runs/
│   ├── v_arr_mvp_dim/<dim>/pointwise.json   # 内部 18 维 pointwise 结果
│   ├── baselines/{videoscore,videoscore2_aligned}/   # ⚠️ unifiedreward/ 已删（见 §4.2）
│   ├── external_benchmarks/videogen_rewardbench/
│   │   ├── rewardverse_pw/pairwise/                  # 旧版直接 pairwise（VQ/MQ/TA, 200 对）
│   │   ├── rewardverse_pw/pairwise_listwise_perrubric/  # 主线 listwise+per_rubric（4 axis × 200 对）
│   │   ├── videoscore_v11/, videoscore2/             # VS pointwise 历史产物
│   │   └── _vq_headtohead/                           # τ-sweep head-to-head 表（已收尾）
│   ├── ALL_RESULTS_SUMMARY.md
│   └── STATUS_SYNC.md          # ← 本文
└── docs/                       # 设计文档
```

---

## 2. 方法（Methods）

### 2.1 已实现 ✅

| Key | 全名 | 实现脚本 | 适用范围 | 状态 |
|---|---|---|---|---|
| `rubric-pointwise` | RewardVerse 18 维 rubric pointwise | [`scripts/v_arr_mvp/pointwise_rubric_judge.py`](../scripts/v_arr_mvp/pointwise_rubric_judge.py) | 内部 v_arr_mvp | ✅ 全 18 维已跑通 |
| `rubric-pairwise-direct` | LLM 直接 A/B/same 判决 | [`scripts/v_arr_mvp/pairwise_rubric_judge.py`](../scripts/v_arr_mvp/pairwise_rubric_judge.py) | VGRB 4 axis | ✅ VQ/MQ/TA 各 200 对（200 对 smoke 历史） |
| **`rubric-listwise-perrubric`** | **listwise 同时输出每条 rubric 的 rank（项目最佳方法）** | 同上（listwise mode + per_rubric 输出） | VGRB 4 axis | ✅ 200 对 smoke 已完成；🟡 全集 25.2k 待自家 MLLM RM 训完后跑（§8.3 M1） |
| `aggregation: mean` | 5 条 rubric 评分的均值 → A/B/same | [`scripts/v_arr_mvp/aggregation_listwise_eval.py`](../scripts/v_arr_mvp/aggregation_listwise_eval.py) | listwise 输出 | ✅ |
| `aggregation: weighted_mean` | LLM 给的 per-rubric weight 加权均值 | 同上 | listwise 输出 | ✅ |
| `aggregation: lp4` | (TOP-v) 上的 4-norm soft-min（弱项更重） | 同上 | listwise 输出 | ✅ |
| `listwise_direct` baseline | 直接拿 LLM 自己输出的 `decision` 字段 | 同上 | listwise 输出 | ✅（作为 sanity baseline） |
| `head-to-head τ sweep` | pointwise 分数差 + τ 阈值 → A/B/same | [`scripts/v_arr_mvp/vgrb_headtohead.py`](../scripts/v_arr_mvp/vgrb_headtohead.py) | VGRB 200 对 | ✅ 已收尾（`_vq_headtohead/table_v2.json`） |
| `B1a / B1b / B1c zero-shot Gemini` | 同一个 Gemini，不同 prompt（VQ-only / 单维 / 7-group） | [`scripts/v_arr_mvp/baseline_zero_shot_18d.py`](../scripts/v_arr_mvp/baseline_zero_shot_18d.py) | **仅作内部 18 维 RQ4 对照** | ✅（在 18 维上已与 rubric pointwise 对照过） |

### 2.2 已删除 / 实证证伪 ❌

> 经过 4 个 axis × 200 对 listwise 数据的实证 ablation，下列聚合方法**已从主线代码中删除**：

| 删除项 | 失败原因 |
|---|---|
| `min` 聚合 | 在 1.0/1.5/2.0 这种粗粒度尺度上产生 22%–44% `pa==pb` ties；4 个 axis 全劣于 mean |
| `median` 聚合 | K=5 太小，位置敏感；全部 axis 劣于 mean |
| `votes` 聚合 | 2 档信号（rank 1 / rank 2）下投票区分度过低，相对 mean 无 headroom |

### 2.3 未实现 / 待实现 🟡

| 编号 | 项目 | 难度 | 备注 |
|---|---|---|---|
| **U2** | VideoScore-2 / VS-v1.1 的 **pairwise wrapper**（pointwise→pair winner） | 低 | 当前 VGRB pairwise 上**还没有这两家的 acc 数字**；只是 τ-sweep 包装，~50 行代码 |
| **U5** | `pacing` 维度 rubric 与 baseline 的 test 版本对齐 | 低 | 当前 rubric 主版用旧 `pacing` (n=30, PLCC 0.501)，baseline 用 `pacing_v2`，标 `(*)`。要消掉这个 `*` 需 rubric 在 `pacing_v2` 上重跑（已有 `pointwise_v2.json` = 0.496/0.481/0.831 备选） |
| **U6** | logic 维度（所有方法的共同薄弱项） | 中 | rubric_v2 PLCC=0.144；test 集本身定义可能不稳，需要补强 ground truth |

> ⚠️ **状态变化记录**：
> - **U0 (VGRB 全集 25.2k)** → 🟡 **未来 rubric-MLLM RM 上跑**（决策记录见 §5.2.2 / §8.3 M1）
> - **U1 (`rewardmodel_7d` 18→7 hierarchical judge)** → ❌ **已下线**：脚本物理删除（其他 benchmark 更需要子维度信号，hierarchical 表格路径不再保留）
> - **U3 (UnifiedReward 重跑)** → 🟡 **待重启**（见 §4.2 现状评估 + §8.3 三个备选方案）
> - **U4 (6 维 fixed-head 不对齐 baseline)** → 不可解，永久搁置（见 §9）

---

## 3. Benchmarks

### 3.1 内部 benchmark：v_arr_mvp（18 维分层 30-band 抽样）

- 18 维：`character / scene / consistency / action / expression / composition / lens / pacing / chromaticity / materiality / lighting / grounding / progression / logic / rhythm / vocal / soundscape / visual_quality`
- 每维一个 `data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>(_v2).json`，n=30
- `visual_quality` 复用全维共享 `shared_test_pointwise.json`
- GT：人工 ordinal label，预测：rubric 加权打分

### 3.2 外部 benchmark：VideoGen-RewardBench（pairwise）

- **来源**：`KlingAIResearch/VideoGen-RewardBench`（也接受 `KwaiVGI/VideoGen-RewardBench`）
- **本地 snapshot**：`/group/40063/jacuzzitang/datasets/VideoGen-RewardBench/`（13.4 GB，含 `videogen-rewardbench.csv` + `videos.zip`）
- **总规模**：CSV 共 25234 行（≈ 25.2k 对，用户口径"~26.5k"）
- **抽样脚本**：[`scripts/v_arr_mvp/fetch_external_benchmarks.py videogen_rewardbench`](../scripts/v_arr_mvp/fetch_external_benchmarks.py)
  - 默认：**emit 全集** （v2 修改后；不再有 `--num_pairs 200` 默认值）
  - Smoke 模式：`--num_pairs 200 --seed 0`
- **4 个 axis**：`VQ`（visual quality）/ `MQ`（motion quality）/ `TA`（text alignment）/ `Overall`，标签 ∈ {A, B, same}
- **GT 分布**（VQ on 200 对 smoke）：`same: 111, A: 46, B: 43`，`gt_same_rate=0.555`，**trivial baseline `all_same_acc=0.555`**

---

## 4. Baselines

### 4.1 已跑通 ✅（仅在内部 v_arr_mvp 18 维上）

| 名称 | HF 仓库 | 类型 | 实现 |
|---|---|---|---|
| **VideoScore-v1.1** | `TIGER-Lab/VideoScore-v1.1` | fixed multi-head | [`baseline_videoscore.py`](../scripts/v_arr_mvp/baseline_videoscore.py)（fp16, 单 GPU ≥16G） |
| **VideoScore-2 (aligned, bf16)** | `TIGER-AI-Lab/VideoScore2` | fixed multi-head | [`baseline_videoscore2.py`](../scripts/v_arr_mvp/baseline_videoscore2.py) |

> 内部结果落点：`runs/baselines/{videoscore, videoscore2_aligned}/_runall_summary.tsv`

### 4.2 待重启 🟡 — UnifiedReward-7B

| 项 | 现状 |
|---|---|
| HF 仓库 | `CodeGoat24/UnifiedReward-7B` |
| 模型权重 | ✅ 仍在本地：`/group/40063/jacuzzitang/models/UnifiedReward-7B/` (~16 GB) |
| 推理脚本 | ❌ 已删（`scripts/v_arr_mvp/baseline_unifiedreward.py`） |
| 工具子目录 | ❌ 已删（`scripts/v_arr_mvp/utils_unified_reward/`） |
| 历史运行产物 | ❌ 已删（`runs/baselines/unifiedreward/`） |
| 历史弃用根因 | H20 GPU + 老 LLaVA-NeXT 推理栈 SIGFPE（rc=136）；多次重装 / 调 dtype / 切 attention impl 均无效 |
| **本次决议** | 🟡 **待重启 / 待选方案**：见 §8.3 三个备选方案与可行性评估 |

### 4.3 trivial baseline（用于挡 floor）

| 名称 | 含义 |
|---|---|
| `all_same_acc` | VGRB 上预测全 same 的 acc（VQ on 200 对 smoke = 0.555） |
| `majority_no_tie_acc` | 排除 same 后的多数类 acc |
| `listwise_direct` | LLM 自己直接输出的 A/B/same，不做任何聚合 |

---

## 5. 当前结果

### 5.1 内部 18 维 pointwise（v_arr_mvp，n=30/dim）

> 完整数字见 [`runs/ALL_RESULTS_SUMMARY.md`](./ALL_RESULTS_SUMMARY.md)。下面给汇总均值（PLCC / SRCC / MAE_zs）：

| 方法 | 维度数 | mean PLCC | mean SRCC | mean MAE_zs |
|---|---:|---:|---:|---:|
| **RewardVerse (rubric)** — 12 个 baseline 已覆盖维度 | 12 | **0.640** | 0.612 | 0.626 |
| VideoScore-v1.1 | 12 | -0.157 | -0.085 | 1.194 |
| VideoScore-2 aligned | 12 | 0.473 | 0.426 | 0.787 |
| **RewardVerse (rubric)** — 全 18 维 | 18 | **0.652** | 0.634 | 0.628 |

**关键结论**：
1. RewardVerse 在 12 维平均 PLCC ≈ 0.64，**+0.18 PLCC** 高于 VideoScore-2 aligned（0.47）。
2. 仅 `lighting` 与 `pacing(v2)` 两维 VS-2 高于 RewardVerse；其余 10 维 RewardVerse 全部领先。
3. `logic` 是所有方法的共同薄弱项。

### 5.2 外部 benchmark：VGRB

#### 5.2.1 主结果：200 对 smoke × 4 axis × listwise+per_rubric+aggregation

> 数据源：`runs/external_benchmarks/videogen_rewardbench/rewardverse_pw/pairwise_listwise_perrubric/aggregation_*.json`

| Axis | listwise_direct | mean | weighted_mean | lp4 |
|---|---|---|---|---|
| | acc / acc_no_tie / macro | acc / no_tie / macro / τ\* | acc / no_tie / macro / τ\* | acc / no_tie / macro / τ\* |
| **VQ** | 0.220 / 0.494 / 0.328 | 0.220 / 0.494 / 0.328 / 0.0 | 0.220 / 0.494 / 0.328 / 0.0 | 0.220 / 0.494 / 0.328 / 0.0 |
| **MQ** | 0.200 / 0.597 / 0.402 | 0.200 / 0.597 / 0.402 / 0.0 | **0.220** / 0.597 / 0.412 / 0.25 | **0.235** / 0.597 / 0.419 / 0.10 |
| **TA** | 0.480 / 0.662 / 0.441 | **0.490** / **0.676** / 0.448 / 0.0 | 0.485 / 0.669 / 0.445 / 0.0 | **0.490** / **0.676** / 0.448 / 0.0 |
| **Overall** | **0.445** / **0.664** / **0.442** | 0.440 / 0.657 / 0.437 / 0.0 | **0.445** / **0.664** / **0.442** / 0.0 | 0.440 / 0.657 / 0.437 / 0.0 |

> τ\* = 在 `tau_grid=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]` 中按 `(acc_no_tie, acc, -tau)` 取最优。
> **观察**：所有方法都被 GT same=55.5% 主导，没有任何方法显著超过 trivial `all_same_acc=0.555` —— 这是 VGRB 的本质难点（pairwise 标签里 same 比例太高），**仅靠"放大样本量"无法绕过**，故当前 Gemini 阶段不跑全集（详见 §5.2.2），改投替代实验（详见 §8）。

#### 5.2.2 全集 ~25.2k 对：🟡 当前 Gemini 阶段不跑，**未来在自家 rubric MLLM reward model 上跑**

> **决策性质**：路线切换（不是"暂缓"也不是"弃跑"）　**决策依据**：成本 vs headroom 不匹配 + benchmark 本质归属

| 维度 | Gemini 路线下的估算（仅供参考） |
|---|---|
| LLM 调用 | 25.2k × 4 axis ≈ **101k** Gemini 调用 |
| 墙钟（concurrency=4） | **~70 h**（≈ 3 天） |
| 费用估算（公开价口径） | **~$5,000 USD** |
| **预期 headroom（Gemini 直接打分）** | **接近 0**：200 对 smoke 已显示所有方法被 GT same=55.5% 主导 |

**为什么不在 Gemini 上跑全集**：在"直接用 Gemini 当 judge"的范式下，~$5k + 数十小时换一个几乎确定不会改变结论的 4 维 acc 表，性价比不值。

**未来必跑 / 主战场归属**（见 §8.3 M1）：
- 🎯 **基于 Gemini 蒸馏 / 训完自家 rubric-based MLLM reward model 之后**，VGRB 全集是该模型的**主战场 benchmark**，必跑全 25.2k；
- 届时 LLM 调用成本归零（本地 GPU 推理），约束只剩 GPU 时间；
- 200 对 smoke 的结果届时可作为 "Gemini-judge 上限" 参考线。

**当前 Gemini 阶段把预算投到**（见 §8）：能真正改变 acc 或填补 baseline 空洞的实验（如 E1 same-prompt ablation、E2 VS pairwise wrapper）。

**保留资产**（届时秒级复用）：
- 全套代码 100% 就绪：`bash scripts/v_arr_mvp/run_external_vgrb.sh` 一键发起
- 200 对 smoke 结果完整保留在 `pairwise_listwise_perrubric/` 目录

#### 5.2.3 旧版直接 pairwise（无 per_rubric，sanity 参考）

> 数据源：`runs/external_benchmarks/videogen_rewardbench/rewardverse_pw/pairwise/{VQ,MQ,TA}.json`

| Axis | acc | acc_no_tie | n_no_tie |
|---|---:|---:|---:|
| VQ | 0.225 | 0.506 | 89 |
| MQ | 0.185 | 0.552 | 67 |
| TA | 0.455 | 0.628 | 145 |

#### 5.2.4 VQ head-to-head（pointwise + τ sweep, 已收尾）

> 数据源：`runs/external_benchmarks/videogen_rewardbench/_vq_headtohead/table_v2.json`
> trivial baselines: **all_same=0.555**, majority_no_tie=0.5169

| 方法 | best τ | acc | acc_no_tie | pred_same_rate |
|---|---:|---:|---:|---:|
| RewardVerse (rubric-pointwise) | 0.5 | 0.350 | 0.247 | 0.445 |
| VS-v1.1 (default head) | 0.5 | 0.315 | 0.225 | 0.425 |
| VS-v1.1 (VQ_subscore head) | 0.3 | 0.345 | 0.315 | 0.355 |
| VS-2 (default head) | 0.5 | 0.400 | 0.348 | 0.455 |
| VS-2 (VQ_subscore head) | 0.5 | 0.360 | 0.405 | 0.365 |

---

## 6. Aggregation 公式 ablation 备忘（已收尾）

> 4 axis × 200 pair listwise 数据实证后，**最终保留 3 种** + `listwise_direct` baseline：

| 公式 | 状态 | 一句话理由 |
|---|---|---|
| `mean` | ✅ 留 | 最简最稳，0 tie，TA 上 acc_no_tie 最优 |
| `weighted_mean` | ✅ 留 | LLM 给非均匀 weight 的 axis 上有差异化（MQ best τ=0.25） |
| `lp4` | ✅ 留 | 1~2 二元尺度上和 mean 几乎等价，但保留 τ-soft-min 旋钮 |
| `listwise_direct` | ✅ 留（baseline） | LLM 自己直接输出，sanity 用 |
| `min` / `median` / `votes` | ❌ 已实证证伪并删除 | 历史脚注；详见 [`aggregation_listwise_eval.py` 顶部 docstring](../scripts/v_arr_mvp/aggregation_listwise_eval.py) |

---

## 7. 代码与数据入口（速查）

### 7.1 主线脚本

| 用途 | 脚本 |
|---|---|
| 18 维 pointwise 评分 | [`scripts/v_arr_mvp/pointwise_rubric_judge.py`](../scripts/v_arr_mvp/pointwise_rubric_judge.py) |
| pairwise/listwise 评分（带 per_rubric） | [`scripts/v_arr_mvp/pairwise_rubric_judge.py`](../scripts/v_arr_mvp/pairwise_rubric_judge.py) |
| listwise → 3 种 aggregation 评估 | [`scripts/v_arr_mvp/aggregation_listwise_eval.py`](../scripts/v_arr_mvp/aggregation_listwise_eval.py) |
| **VGRB 一键全流程** | [`scripts/v_arr_mvp/run_external_vgrb.sh`](../scripts/v_arr_mvp/run_external_vgrb.sh) |
| 外部 benchmark 拉取（仅 VGRB） | [`scripts/v_arr_mvp/fetch_external_benchmarks.py`](../scripts/v_arr_mvp/fetch_external_benchmarks.py) |
| VGRB 4 axis listwise view 切分 | [`scripts/v_arr_mvp/prep_vgrb_pairwise_view.py`](../scripts/v_arr_mvp/prep_vgrb_pairwise_view.py) |
| Zero-shot Gemini baseline (B1abc, **仅内部 18 维 RQ4 对照**) | [`scripts/v_arr_mvp/baseline_zero_shot_18d.py`](../scripts/v_arr_mvp/baseline_zero_shot_18d.py) |
| VideoScore baseline | [`scripts/v_arr_mvp/baseline_videoscore.py`](../scripts/v_arr_mvp/baseline_videoscore.py) |
| VideoScore2 baseline | [`scripts/v_arr_mvp/baseline_videoscore2.py`](../scripts/v_arr_mvp/baseline_videoscore2.py) |
| VGRB head-to-head（已收尾） | [`scripts/v_arr_mvp/vgrb_headtohead.py`](../scripts/v_arr_mvp/vgrb_headtohead.py) |

### 7.2 关键产物

| 产物 | 路径 |
|---|---|
| 18 维 pointwise 主结果 | `runs/v_arr_mvp_dim/<dim>/pointwise.json` |
| 18 维汇总 markdown | [`runs/v_arr_mvp_dim/all_dims_summary.md`](./v_arr_mvp_dim/all_dims_summary.md) |
| VGRB listwise 原始输出（200 对 smoke） | `runs/external_benchmarks/videogen_rewardbench/rewardverse_pw/pairwise_listwise_perrubric/{VQ,MQ,TA,Overall}.json` |
| VGRB aggregation 评估 | `runs/external_benchmarks/videogen_rewardbench/rewardverse_pw/pairwise_listwise_perrubric/aggregation_*.json` |
| 全体结果总览 | [`runs/ALL_RESULTS_SUMMARY.md`](./ALL_RESULTS_SUMMARY.md) |

### 7.3 复现入口

```bash
# 内部 18 维 pointwise（已有缓存可秒过）
bash scripts/v_arr_mvp/run_rubric_all_dims.sh

# 外部 VGRB 200 对 smoke（最佳方法 = listwise + per_rubric → aggregation）
# 当前阶段默认只跑 smoke；未来 rubric-MLLM RM 训完后再去掉 VGRB_LIMIT 跑全集
VGRB_LIMIT=200 bash scripts/v_arr_mvp/run_external_vgrb.sh

# Aggregation 重新评估（4 个 axis）
python scripts/v_arr_mvp/aggregation_listwise_eval.py \
    --listwise_pred runs/external_benchmarks/videogen_rewardbench/rewardverse_pw/pairwise_listwise_perrubric/Overall.json \
    --axis Overall \
    --output runs/external_benchmarks/videogen_rewardbench/rewardverse_pw/pairwise_listwise_perrubric/aggregation_Overall.json
```

---

## 8. 下一步可补的实验（v4 替代清单）

> 投入排序：从"低成本 + 高 ROI"到"高成本 + 不确定 ROI"。

### 8.1 🔥 高优先（论文级，必须补）

| 编号 | 项目 | 估算成本 | 预期产出 | 说明 |
|---|---|---|---|---|
| **E1** | **VGRB 200 对 smoke 上加 `same-prompt` ablation** | ~200 × 4 axis × 2 prompt = 1.6k Gemini 调用，**~$50** / ~30 min | 验证"显式鼓励输出 same"能否打破 trivial all_same=0.555 floor | 当前 LLM 几乎从不输出 same（pred_same_rate ~5%），这是被 trivial baseline 主导的根因；如果改 prompt 后 acc 显著上升，就值得再考虑 Gemini 阶段全集 |
| **E2** | **U2: VS-v1.1 / VS-2 pairwise wrapper（VGRB 200 对 smoke）** | 0 LLM 调用（本地 GPU 推理） / 几小时墙钟 | VGRB 200 对 4 axis 上 baseline 列填齐，**消除"对比表空洞"问题** | pointwise→pair winner 的 τ-sweep 包装，~50 行代码 |

### 8.2 🟡 中优先（增益型）

| 编号 | 项目 | 估算成本 | 预期产出 | 说明 |
|---|---|---|---|---|
| **E4** | **VGRB 5k 分层抽样跑 listwise+per_rubric** | ~20k 调用 / ~$1000 / ~14h | 全集结论的"95% 置信区间"代理 | 如果 reviewer 一定要"complete eval split"以外的数字，5k 比 200 强且比 25.2k 便宜 5×；分层 by GT label + axis |
| **U5** | `pacing` rubric 用 `pacing_v2` 重跑 | ~30 LLM 调用 / ~$1 | 消掉 ALL_RESULTS 表里的 `(*)` 标注 | 已有 `pointwise_v2.json` 数字 = 0.496/0.481/0.831 备选 |
| **U6** | logic 维度 GT 重审 | 人工审核成本 | 修复"三个方法都为负"的疑似不稳 GT | 需要人工复核 30 个 logic test sample |

### 8.3 🔵 低优先 / 高风险

| 编号 | 项目 | 估算成本 | 风险 | 说明 |
|---|---|---|---|---|
| **E6** | **UnifiedReward-7B 重跑** | 工程时间 ~2-3 天 + 0 API 费用 | **高**：原弃用根因（H20 + 老 LLaVA-NeXT SIGFPE rc=136）未消除 | 现状：模型权重还在 `/group/40063/jacuzzitang/models/UnifiedReward-7B/`（16GB）；但推理脚本 + 工具子目录 + 历史产物都已删。重跑需要：1）重新实现推理脚本；2）解决原 SIGFPE。**详见下方三个备选方案** |
| **M1** 🎯 | VGRB 全集 25.2k（**自家 rubric MLLM reward model 路线**） | 0 API 费用 + GPU 时间 | 低（本地推理无 quota 风险） | **未来里程碑 / 必跑**：基于 Gemini 蒸馏/训完 rubric-MLLM RM 后跑；VGRB 全集是该模型的主战场 benchmark；相同代码 `run_external_vgrb.sh` 直接换 backend 即可 |

#### UnifiedReward-7B 重跑的三个备选方案

> ⚠️ **请先拍板再动手**：当年弃用的根因（SIGFPE）没消除前就重写脚本，大概率重蹈覆辙。

**方案 A — 换 GPU 卡型（最便宜先试）**
- 在非 H20 卡（A100 / V100 / 4090）上重跑试一次
- 工程量：从 git 历史恢复旧脚本 + 重新部署，~半天
- 风险：依赖卡型可用性
- 推荐度：⭐⭐⭐ **首选**（成本最低、风险可控）

**方案 B — 升级推理栈**
- 把老 LLaVA-NeXT 替换为 transformers ≥4.45 的官方 `LlavaNextForConditionalGeneration`
- 工程量：~1-2 天，需要重新对齐 prompt 模板
- 风险：UnifiedReward-7B 的 reward head 与新 transformers 版本是否兼容存疑
- 推荐度：⭐⭐ 备选（如果方案 A 仍 SIGFPE 再试）

**方案 C — 用 vLLM 重写**
- 直接走 vLLM 的 LLaVA-NeXT 后端，绕开原推理栈
- 工程量：~2-3 天
- 风险：需要适配 reward head；社区对 UnifiedReward-7B 的 vLLM 支持度未知
- 推荐度：⭐ 最后备选

**最终建议**：先做 §8.1 的 E1 + E2 两件低成本高 ROI 的事；**UnifiedReward 重跑放最后**，且必须先选定方案 A/B/C 之一。

---

## 9. 已弃用 / 永久搁置清单

| 项 | 状态 | 备注 |
|---|---|---|
| `rewardmodel_7d` (18→7 hierarchical judge) | ❌ 已下线（脚本物理删除） | 其他 benchmark 更需要子维度信号；hierarchical 表格路径不再保留 |
| `build_sota_table.py` + `sota_runbook.md` + `sota_manifest.example.json` | ❌ 已物理删除 | 围绕 `rewardmodel_7d` 的 SOTA 表构建链路一并下线 |
| VGRB 上 B1abc Zero-shot Gemini 系统跑 | ❌ 不做 | zero_shot 脚本仅保留作为内部 18 维 RQ4 对照 |
| VideoFeedback-200 整组实验 | 🗑️ 早已下架 | 5 axis 上 RV zero-shot 难以与 in-distribution VS-v1.1 拉开差距，且 VF 维度无法体现 18 维优势 |
| 6 维 fixed-head 不对齐 baseline | ❌ 永久搁置 | composition / lens / expression / progression / vocal / soundscape，VS 系列 head 不对齐，无解 |
| `min` / `median` / `votes` aggregation | ❌ 已实证证伪并删除 | 见 §6 |

> ℹ️ **VGRB 全集 25.2k 不在本清单中**：当前 Gemini 阶段不跑（见 §5.2.2 / §8.3 M1），未来训完自家 rubric MLLM reward model 后必跑（主战场 benchmark）。
>
> ℹ️ **UnifiedReward-7B 不在本清单中**：状态为 🟡 待重启（见 §4.2 / §8.3）。

---

> **🔄 文档维护**：每当某项 status 翻转（🟡→✅ 或 ❌→✅），请同步更新 §0 TL;DR + 对应小节 + `runs/ALL_RESULTS_SUMMARY.md`。
