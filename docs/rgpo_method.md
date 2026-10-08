# RGPO 方法（Method）— 论文精简稿

> 本文件用于论文 Method 章节，对 **RGPO**（Rubric-Guided / Rubric-Generative Policy Optimization）的三阶段流程做提纲式精简介绍：
> ① Seed Rubric 自进化 → ② Seed Rubric Warm-up（增强 score 能力）→ ③ Query 定制化 Rubric + Score 联合训练。

---

## 概览（One paragraph）

给定查询 $q$ 与候选生成 $y$，奖励模型需输出可解释的评分 $S(y\!\mid\!q)$。RGPO 把这一映射拆为两步：先由模型对 $q$ 生成查询定制化 rubric $\hat R$，再以 $\hat R$ 为评分参照打分 $S(y\!\mid\!q,\hat R)$。三阶段依次解决三件事：**(i)** 用少量种子样本由 teacher VLM 自进化出高质量 seed rubric 集合 $R^\star$ 作为 anchor；**(ii)** 在 $R^\star$ 固定的条件下 warm-up 学生模型的打分头，使其先具备稳定的 rubric-conditioned 评分能力；**(iii)** 在策略梯度框架下放开 rubric 生成，用 $R^\star$ 做 alignment anchor，对 rubric 与 score 做联合优化。

代码位置：`rubric_pipeline/generator/iterative_rubric/`、`rm_training/stage1_teacher/`、`rm_training/stage2_grpo/`。

---

## ① Seed Rubric 自进化

**目标**：在不依赖人工 rubric 标注的前提下，仅用少量带有偏好标签的种子样本，自动产出每个评测维度 $d$ 的高质量 seed rubric 集合 $R^\star_d=\{r^\star_{d,1},\ldots,r^\star_{d,K}\}$（默认 $K\!=\!5$）。

**输入**：种子 pair 集合 $\mathcal{D}_\text{seed}=\{(q_i,y_i^+,y_i^-)\}$ 与 EvalVerse 维度分类法 $\mathcal{T}$。

**核心循环（Propose–Evaluate–Revise）**：在 teacher VLM（如 Gemini）上对每个维度迭代

1. **Propose**：基于 $q$、当前已有 rubric 与维度上下文 $\mathcal{T}_d$ 提议候选 rubric $\tilde R$；
2. **Evaluate**：用 $\tilde R$ 在 $\mathcal{D}_\text{seed}$ 上对 $(y^+,y^-)$ 做 pointwise 打分，统计偏好恢复率 $\mathrm{acc}(\tilde R)$；
3. **Revise**：若 $\mathrm{acc}<\tau$ 或信息增量 $<\epsilon$，将失败 case 与差异维度回灌到下一轮 prompt，触发改写；超过 patience 次低增量则早停。

**约束与去重**：rubric theme 强制对齐 EvalVerse 子维度分类法以抑制语义漂移；同维内 Jaccard 阈值 0.65、跨维 0.85 做去重；最终用 MCR² 选出最具区分度的 $K$ 条作为 $R^\star_d$。

**输出**：每维度一份 seed rubric 文本（`rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt`），下游打包为 `data/rm_train/gold_rubrics_18d.jsonl`，每条带 `{theme, weight=1/K, tips}`。

**性质**：纯离线、teacher-only，不更新任何学生参数；产出的 $R^\star$ 既是 ② 的监督信号，也是 ③ 的 alignment anchor。

---

## ② Seed Rubric Warm-up（增强 score 能力）

**目标**：在 $R^\star$ 固定的条件下，先训练学生 $\pi_\theta$ 的**打分能力**，避免在 ③ 一开始 rubric 与 score 同时不可用导致的优化噪声。

**做法**：在 stage2 训练器（`train_grpo.py`）中开启 `--gold_rubric_warmup_epochs > 0` 进入 warm-up 窗口（默认 1 epoch，约 500 steps）：

- **rubric 端冻结**：每个 rollout 直接喂入对应维度的 gold rubric $R^\star_d$，跳过学生的 rubric 生成与对应策略梯度（`rubric_metas=[None]*G`，rubric-PG 链路关闭）；
- **score 端打开**：学生在 $R^\star_d$ 条件下分别对 $y^+$、$y^-$ 做 pointwise 打分（`score_method=soft_logits`，对位置 logits 在 $\{1,\dots,5\}$ 上取期望，输出连续分），策略梯度仅经由 score token 反传；
- **强信号**：warm-up 期内 `score_pg_alpha` 强制升至 `score_pg_alpha_max`，并把 Bradley-Terry 偏好项 $r_\text{pref}=\sigma\!\big(S(y^+\!\mid\!R^\star)-S(y^-\!\mid\!R^\star)\big)$ 作为主奖励；
- **格式压制**：`β_format` 在早期取较高值（默认 $0.5$）以约束 score scaffold 与权重和等 schema 合规性。

**训练目标**（warm-up 期）：
$$
\mathcal{L}_\text{warm} \;=\; -\,\mathbb{E}\big[\,r_\text{pref}\!+\!\beta\,r_\text{format}\,\big] \;+\; \beta_\text{KL}\,\mathrm{KL}\big(\pi_\theta\,\|\,\pi_\text{ref}\big)
$$
其中 $\pi_\text{ref}$ 为关闭 LoRA 的同一基座（`disable_adapter()`，零额外显存）。

**作用**：把"能在 rubric 条件下稳定打出与偏好一致分数"这一能力**先构建出来**，让 ③ 阶段的 rubric 生成有可信赖的下游评估信号，否则 score 端噪声会污染 rubric-PG 的优势估计。

---

## ③ 联合训练：Query 定制化 Rubric + Score（GRPO）

**目标**：放开 rubric 生成，让学生对每个 $q$ 直接产出查询定制化 rubric $\hat R$ 并打分，**rubric 与 score 共同优化**。

**Rollout（每对 $(q,y^+,y^-)$ 采 $G\!=\!8$ 条）**：
1. 仅给定 caption/$q$，从 $\pi_\theta$ 采 $\hat R_g$（rubric-gen，policy gradient 经由 rubric token 反传）；
2. 在 $\hat R_g$ 下对 $y^+$、$y^-$ 各做一次 pointwise score forward（**与部署一致**：单视频，不让模型同时看到两个候选）；
3. 计算每个 rollout 的奖励 $r_g$，在 group 内做 GRPO 优势 $A_g\!=\!(r_g-\bar r)/\sigma_r$ 后回传。

**奖励**：
$$
r(\hat R_g) \;=\; \underbrace{\sigma\!\big(S(y^+\!\mid\!\hat R_g)\!-\!S(y^-\!\mid\!\hat R_g)\big)}_{r_\text{pref}}
\;+\;\lambda(t)\!\cdot\!\underbrace{\tfrac{1}{n}\sum_j\max_i\cos(d_i,d_j^\star)}_{r_\text{align}}
\;+\;\beta(t)\!\cdot\! r_\text{format}
$$

- $r_\text{pref}\in(0,1)$：soft Bradley-Terry，对齐 pairwise GT；
- $r_\text{align}\in[0,1]$：$\hat R_g$ 与 seed $R^\star$ 的 **theme** 嵌入（BGE-M3）做二部图 Hungarian 余弦匹配；采用 **warm-start anchor**：当 $\mathrm{sim}\!\ge\!0.7$ 即饱和到 $1.0$ 不再贡献方差，$<0.7$ 线性衰减到 0——把 $R^\star$ 当作"不要漂离 teacher 子维度分类法"的锚而非要逐字复刻的答案；
- $r_\text{format}\in[0,1]$：JSON 合法性 + scaffold 合法性 + 5 条 + 权重和为 1 + 分数 $\in[1,5]$；
- $\lambda(t)\!=\!\lambda_0\!\cdot\!(1-\overline{\text{valid\_ratio}})$：当学生自发产出**有效 rubric**（$|S^+\!-\!S^-|>\epsilon$）的比例上升时，alignment 项自然衰减；
- $\beta(t)$：从 $0.5$ 线性/cosine 衰减到 $0.05$（默认 500 步），早期高 $\beta$ 把策略锚到 schema 合规，末期保留小幅压力防止格式漂回。

**梯度路径**（关键工程取舍）：
- **rubric-PG**：经 rubric 生成 token，是策略学习的主路径；
- **score 前向**：默认 greedy + `no_grad`，**不**反传——若让 score forward 也带梯度，会让模型学会**伪造分数**而非改进 rubric；
- **score-PG（可选 curriculum）**：在 `--score_pg_start_step` 之后线性升起 $\alpha\in[0,\alpha_\text{max}]$，仅在确有 score-token 元数据时启用；
- **KL**：参考策略为关闭 LoRA 的同基座，rubric 与 score 两条链路共享 $\beta_\text{KL}$，score 链路按当前 $\alpha$ 缩放，避免 warm-up 退场时 KL 计数的不连续。

**总损失**：
$$
\mathcal{L} \;=\; -\,\mathbb{E}_g\!\big[A_g\,(\log\pi_\theta(\hat R_g\!\mid\!q) + \alpha(t)\!\cdot\!\log\pi_\theta(s_g\!\mid\!\hat R_g,y))\big] \;+\; \beta_\text{KL}\,\mathrm{KL}(\pi_\theta\,\|\,\pi_\text{ref})
$$

**部署一致性**：推理期与 ② / ③ 的 score 前向严格同构——单视频 pointwise，先 `generate_rubric(q)` 再 `score_video(y,\hat R)`，pairwise 评测时对 $y_a,y_b$ 各打一次再比较。

---

## 三阶段衔接（Why this order）

| 阶段 | 谁更新 | rubric 来源 | score 来源 | 主奖励 |
|---|---|---|---|---|
| ① 自进化 | teacher VLM（offline） | self-evolved | teacher pointwise | 偏好恢复率 |
| ② Warm-up | student score 头 | **fixed** $R^\star$ | student pointwise | $r_\text{pref}+\beta\,r_\text{format}$ |
| ③ Joint  | student rubric + score | $\pi_\theta(\hat R\!\mid\!q)$ | student pointwise | $r_\text{pref}+\lambda\,r_\text{align}+\beta\,r_\text{format}$ |

直观地：① 提供 anchor，② 让 anchor 可被"用得起来"，③ 在不丢 anchor 的前提下让 rubric 自由适配每条 query。三阶段共享同一 score 前向接口与 EvalVerse 维度上下文，保证 rubric 与 score 的 prompt 在三阶段不发生漂移。

---

## 关键超参（默认）

| 名称 | 默认 | 说明 |
|---|---|---|
| $K$（每维 rubric 数） | 5 | seed 与 student 输出统一 |
| $G$（rollout 数） | 8 | GRPO group size |
| `score_method` | `soft_logits` | $\{1..5\}$ logits 期望，连续 |
| `gold_rubric_warmup_epochs` | 1 | warm-up 窗口长度 |
| `score_pg_alpha_max` | 1.0 | warm-up 期强制取此值 |
| `λ₀` | 0.3 | alignment 初始权重 |
| `align_floor` | 0.7 | warm-start saturation 阈值 |
| $\beta_0\!\to\!\beta_\text{min}$ | $0.5\!\to\!0.05$ | 默认 500 步线性衰减 |
| $\beta_\text{KL}$ | 0.04 | 参考策略 = LoRA-off 同基座 |
| LR / step | $5e\!-\!6$ / 1000 | LoRA(attn) on Qwen2.5-VL-7B-Instruct |
