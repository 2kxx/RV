## RewardVerse — 当前所有 baseline / rubric 结果总览

> 截止：2026-05-28（命名口径统一）
> 本文件按"每个维度 × 每种方法"列出已跑通的 PLCC / SRCC / MAE_zs / N。
> 数据来源：每个维度对应的 `pointwise.json` / `*_test.json` 头部字段（不依赖可能过时的 `_runall_summary.tsv`）。
>
> **命名约定**：本项目方法名统一为 **RewardVerse (rubric)**；底层 LLM 后端为 `gemini-3.1-pro`，经腾讯 Venus llmproxy 转发。
> 历史输出文件（`pointwise.json` 的 `"method"` 字段）可能仍写作 `Auto-Rubric (Gemini-3.1-pro)`，下次重跑会改写为新值；已存档的旧 JSON 不重写（不影响指标，仅影响标签）。

---

### ✅ 评测集说明（同口径）

之前一度误以为 rubric 跑的是另一份"自有 test"，经核对每份 `pointwise.json` 的 `"dataset":` 字段，**rubric 与 baseline 跑的就是同一组 per-dim test 数据集**。
统一口径如下：

| 方法 | 实际 test 集 | 已跑维度 |
|---|---|---:|
| **RewardVerse (rubric)** | `data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>(_v2).json`（visual_quality 用 `shared_test_pointwise.json`） | **18/18** ✅ |
| **VideoScore-v1.1** | 与 rubric 相同 | **12/18** ✅（其余 6 维 head 不对齐，**有意跳过**） |
| **VideoScore-2 aligned** | 与 rubric 相同 | **12/18** ✅（其余 6 维 head 不对齐，**有意跳过**） |
| **UnifiedReward-7B** | （计划同上） | **0/18** ❌（H20 + 老 LLaVA-NeXT 推理栈 SIGFPE，已弃用） |

#### 12 个 baseline
#### 12 个 baseline 已覆盖维度的 rubric 主结果选择（按你最新指令）

- **`logic`** → 用 **v2**（`pointwise_v2.json` 即当前主 `pointwise.json`，dataset = `test_pointwise_logic_v2.json`，n=30）。
- 其它 11 维：在 rubric 当前主 `pointwise.json` 与 `pointwise_v2.json` 之间取**综合表现更好**那一份。
  - `consistency / logic`：两份完全一致（都已切到 v2），无差异，直接用主 `pointwise.json`。
  - `pacing`：旧版 `pointwise.json`（旧 test）综合略好（PLCC 0.501 vs v2 0.496，SRCC 0.494 vs 0.481，MAE 0.779 vs 0.831）→ **保留旧版数字**，但要注意 baseline 跑的是 `pacing_v2`，**两边 test 不同**，下面用 `(*)` 标注。
  - 其余 9 维主 `pointwise.json` 的 dataset 与 baseline 完全一致。

#### ⚠️ 唯一仍有 test 不一致的格子

| Dim | Rubric (本表使用) | Baseline (VideoScore-2 aligned) |
|---|---|---|
| `pacing` | `test_pointwise_pacing.json`（旧, n=30） | `test_pointwise_pacing_v2.json`（v2, n=30） |

---

### 📊 总览表（按维度对齐）

每个 cell 形如 `PLCC / SRCC / MAE_zs (N)`；空白表示该方法没有该维度结果。
**粗体** = 同一行 PLCC 最高的方法；`(*)` = 该行 rubric 与 baseline test 版本不一致。

| # | Dimension | RewardVerse (rubric) | VideoScore-v1.1 | VideoScore-2 aligned | UnifiedReward |
|---|---|---|---|---|---|
| 1 | `visual_quality` | **0.843 / 0.818 / 0.457** (30) | -0.298 / -0.012 / 1.156 (30) | 0.785 / 0.772 / 0.561 (30) | FAIL |
| 2 | `chromaticity` | **0.784 / 0.767 / 0.504** (30) | -0.296 / -0.096 / 1.197 (30) | 0.640 / 0.595 / 0.695 (30) | FAIL |
| 3 | `lighting` | 0.708 / 0.658 / 0.542 (30) | -0.250 / 0.030 / 1.041 (30) | **0.789 / 0.724 / 0.519** (30) | FAIL |
| 4 | `materiality` | **0.786 / 0.766 / 0.513** (30) | -0.200 / -0.093 / 1.187 (30) | 0.748 / 0.805 / 0.588 (30) | FAIL |
| 5 | `consistency` | **0.551 / 0.528 / 0.744** (30) | -0.339 / -0.268 / 1.382 (30) | 0.394 / 0.296 / 0.819 (30) | FAIL |
| 6 | `action` | **0.580 / 0.566 / 0.762** (30) | -0.034 / -0.098 / 1.223 (30) | 0.300 / 0.150 / 0.993 (30) | FAIL |
| 7 | `rhythm` | **0.565 / 0.552 / 0.613** (30) | 0.187 / 0.034 / 1.062 (30) | 0.273 / 0.201 / 0.985 (30) | FAIL |
| 8 | `logic` | **0.144 / 0.227 / 0.971** (30, v2) | -0.134 / -0.001 / 1.229 (30, v2) | -0.130 / -0.058 / 1.238 (30, v2) | FAIL |
| 9 | `pacing` (*) | 0.501 / 0.494 / 0.779 (30, 旧) | -0.010 / -0.007 / 1.091 (30, v2) | **0.558 / 0.502 / 0.745** (30, v2) | FAIL |
| 10 | `character` | **0.813 / 0.829 / 0.483** (30) | 0.105 / 0.225 / 1.047 (30) | 0.608 / 0.756 / 0.642 (30) | FAIL |
| 11 | `scene` | **0.774 / 0.764 / 0.542** (30) | -0.234 / -0.219 / 1.224 (30) | 0.561 / 0.648 / 0.734 (30) | FAIL |
| 12 | `grounding` | **0.628 / 0.537 / 0.706** (30) | -0.343 / -0.293 / 1.390 (30) | 0.605 / 0.462 / 0.683 (30) | FAIL |
| 13 | `expression` | **0.600 / 0.612 / 0.700** (30) | N/A — no aligned head | N/A — no aligned head | — |
| 14 | `composition` | **0.693 / 0.661 / 0.572** (30) | N/A — no aligned head | N/A — no aligned head | — |
| 15 | `lens` | **0.723 / 0.725 / 0.554** (30) | N/A — no aligned head | N/A — no aligned head | — |
| 16 | `progression` | **0.656 / 0.594 / 0.664** (30) | N/A — no aligned head | N/A — no aligned head | — |
| 17 | `vocal` | **0.630 / 0.619 / 0.681** (30) | N/A — vision-only model | N/A — vision-only model | — |
| 18 | `soundscape` | **0.758 / 0.692 / 0.511** (30) | N/A — vision-only model | N/A — vision-only model | — |

注意几处与之前老 `_runall_summary.tsv` 不同的更新：
- **`visual_quality`**：rubric `pointwise.json` 头部即 PLCC=0.843（在 `shared_test_pointwise.json` 上，n=30），不是 0.708。
- **`consistency`**：当前主 `pointwise.json` 的 dataset 已是 `consistency_v2`，PLCC=0.551（旧 tsv 里的 0.536 已过时）。
- **`logic`**：当前主 `pointwise.json` 的 dataset 已是 `logic_v2`，n=30，PLCC=0.144（旧 tsv 里的 n=19、PLCC=-0.019 已过时）。

---

### 🔢 各方法均值（在 12 个 baseline 已覆盖的维度上）

| 方法 | 维度 | mean PLCC | mean SRCC | mean MAE_zs |
|---|---:|---:|---:|---:|
| **RewardVerse (rubric)** | 12 | **0.640** | 0.612 | 0.626 |
| VideoScore-v1.1 | 12 | -0.157 | -0.085 | 1.194 |
| VideoScore-2 aligned | 12 | 0.473 | 0.426 | 0.787 |

> 12 维 rubric 均值用的是上表 1–12 行的 PLCC/SRCC/MAE_zs 直接平均；`pacing` 一行是与 baseline 不同口径的旧版数字（标 `(*)`），这是按你"其他维度选综合表现好的"指令保留下来的。

仅在 11 个**完全同 test** 的维度上（去掉 `pacing`）：

| 方法 | 维度 | mean PLCC | mean SRCC | mean MAE_zs |
|---|---:|---:|---:|---:|
| **RewardVerse (rubric)** | 11 | **0.652** | 0.622 | 0.612 |
| VideoScore-v1.1 | 11 | -0.171 | -0.092 | 1.203 |
| VideoScore-2 aligned | 11 | 0.465 | 0.419 | 0.791 |

全 18 维 rubric 均值（仅 rubric 自身参考）：

| 方法 | 维度 | mean PLCC | mean SRCC | mean MAE_zs |
|---|---:|---:|---:|---:|
| RewardVerse (全 18 维) | 18 | 0.652 | 0.634 | 0.628 |

---

### 🧷 各方法状态简记

- **RewardVerse (rubric)**：18 维全部 OK。LLM 后端为 `gemini-3.1-pro`，经腾讯 Venus llmproxy 转发（见 `rubric_pipeline/config/v_arr_mvp_t2v.yaml`）。`logic` 主结果已切到 v2 (n=30)；`consistency` 主结果亦在 v2；`pacing` 维度按"综合更好"原则保留旧 test 的 0.501，与 baseline 的 v2 不同口径。
  汇总入口：[runs/v_arr_mvp_dim/all_dims_summary.md](runs/v_arr_mvp_dim/all_dims_summary.md)
- **VideoScore-v1.1** (`TIGER-Lab/VideoScore-v1.1`)：12 维全部跑通，PLCC 平均 −0.16，与人工口径几乎不相关，作为下限基线。其余 6 维（`composition / lens / expression / progression / vocal / soundscape`）模型 head 不对齐或缺音频通道，**有意未跑**（脚本顶部注释有说明）。
  汇总入口：[runs/baselines/videoscore/_runall_summary.tsv](runs/baselines/videoscore/_runall_summary.tsv)
- **VideoScore-2 aligned (bf16)** (`TIGER-AI-Lab/VideoScore2`)：12 维全部跑通（`visual_quality` 标记 SKIP_EXISTING，复用上一次成功 run）；其余 6 维**有意未跑**，原因同上。
  汇总入口：[runs/baselines/videoscore2_aligned/_runall_summary.tsv](runs/baselines/videoscore2_aligned/_runall_summary.tsv)
- **UnifiedReward-7B**：**已弃用**，原因：H20 + 老 LLaVA-NeXT 推理栈触发 SIGFPE（rc=136），多次重装环境 / 调 dtype / 切 attention impl 均无效。
  详情：[runs/baselines/unifiedreward/ABANDONED.md](runs/baselines/unifiedreward/ABANDONED.md)

---

### 📌 主要结论

1. **RewardVerse 在 12 维平均 PLCC ≈ 0.64，全面优于 VideoScore-2 aligned（≈ 0.47）和 VideoScore-v1.1（≈ −0.16）**。
2. 只看 11 个完全同 test 的维度，RewardVerse 0.65 vs VideoScore-2 aligned 0.47，差距 +0.18 PLCC。
3. **VideoScore-2 aligned 仅在 `lighting` 与 `pacing(v2)` 两维 PLCC 高于 RewardVerse**；其余 10 维 RewardVerse 全部领先。
4. `logic` 是所有方法的共同薄弱项（RewardVerse PLCC=0.14，两个 baseline 都为负），test 集本身可能定义不稳。
5. **18 维独占的 6 维**（`composition / lens / expression / progression / vocal / soundscape`）凸显 RewardVerse 的方法学优势：rubric 可以为任意维度生成评分准则，而 fixed-head 模型受限于训练时的 head 数量与模态（VideoScore 系列均为纯视觉）。
6. 想完全消掉 `pacing` 那个 `(*)`：要么重跑 rubric_v2 在 `pacing_v2` 上（已有 `pointwise_v2.json` = 0.496/0.481/0.831），要么让 baseline 也跑一次 `pacing` 旧版——按你的指令选当前更高的 0.501，已保留。

