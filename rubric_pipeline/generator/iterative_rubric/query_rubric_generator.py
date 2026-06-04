# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""Query-specific rubric generation for vision Auto-Rubric.

This module keeps an iterative Auto-Rubric loop and specializes the
prompts and data handling for visual generation tasks:

- Text-to-image: pointwise scoring and listwise ranking. Pairwise is the
  two-image special case of listwise.
- Image editing: pointwise scoring and listwise ranking, with an optional
  original/base image followed by one or more edited outputs.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Mapping

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from rubric_pipeline.graders.schema import (
    GraderMode,
    GraderRankCallback,
    GraderScoreCallback,
)
from rubric_pipeline.models.base_chat_model import BaseChatModel
from rubric_pipeline.models.schema.oai.message import ChatMessage
from rubric_pipeline.models.schema.prompt_template import LanguageEnum, PromptTemplate
from rubric_pipeline.utils.media import (
    TASK_IMAGE_EDIT,
    TASK_R2V,
    TASK_T2I,
    TASK_T2V,
    TASK_T2V_MULTISHOT,
    TASK_T2V_SOUND,
    attach_media_to_messages,
    count_outputs,
    extract_media_paths,
    extract_reference_images,
    has_base_media,
    is_image_edit_task,
    is_video_edit_task,
    is_video_task,
    normalize_task_type,
)

# EvalVerse taxonomy import (single source of truth in evalverse_adapter).
# Used to build the optional `evalverse_dim_block` injected into T2V prompts
# when the caller targets a specific EvalVerse main dimension (e.g. visual_quality).
from rubric_pipeline.datasets.evalverse_adapter import (
    EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS,
    EVALVERSE_MAIN_DIMENSIONS,
    EVALVERSE_MAIN_TO_SUB_DIMENSIONS,
)

# pylint: disable=line-too-long


# ========== Text-to-Image Prompts ==========

T2I_POINTWISE_GENERATION_PROMPT_ZH = """
请基于输入的文本提示、生成图片和标注分数，生成**恰好 {generate_number} 个**用于单图评分的视觉评估 rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: Pointwise，对一张生成图片独立评分，分数范围为 {min_score}-{max_score}，输出必须是整数
- 标准应覆盖文本提示遵循度、主体/属性/关系完整性、视觉质量、伪影控制和整体美感
- 标准必须具体、可操作，并能区分不同分数档位
- 只输出评分标准，不要直接给当前图片打分
- 必须严格生成 {generate_number} 个标准

## 输出格式
{{
    "rubrics": [
        "第一个评分标准的详细描述",
        "... 共 {generate_number} 个"
    ],
    "reason": "生成这些评分标准的原因"
}}
"""

T2I_POINTWISE_GENERATION_PROMPT_EN = """
Based on the text prompt, the generated image, and the labeled score, generate **exactly {generate_number}** visual evaluation rubrics for pointwise scoring.
{task_description_section}
{sample_content}

## Requirements
- Evaluation mode: Pointwise. Score one generated image independently on an integer scale from {min_score} to {max_score}
- Cover prompt adherence, subject/attribute/relation completeness, visual quality, artifact control, and overall aesthetics
- Each rubric must be concrete, actionable, and able to distinguish score levels
- Do not score the current image; only output reusable scoring criteria
- Generate exactly {generate_number} rubrics

## Output Format
{{
    "rubrics": [
        "Detailed description of the first scoring rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were generated"
}}
"""

T2I_LISTWISE_GENERATION_PROMPT_ZH = """
请基于输入的文本提示、{num_outputs} 张生成图片及标注排序，生成**恰好 {generate_number} 个**用于图片排序的视觉评估 rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: {ranking_mode}，对多张文本生成图片进行相对质量排序；rank=1 表示最好
- 分析高质量与低质量图片在提示遵循度、细节完整性、真实感、构图、伪影和美学上的差异
- 标准应能对所有候选图片建立清晰质量梯度，避免位置偏见
- 只输出排序标准，不要直接给当前图片排序
- 必须严格生成 {generate_number} 个标准

## 输出格式
{{
    "rubrics": [
        "第一个排序标准的详细描述",
        "... 共 {generate_number} 个"
    ],
    "reason": "生成这些排序标准的原因"
}}
"""

T2I_LISTWISE_GENERATION_PROMPT_EN = """
Based on the text prompt, {num_outputs} generated images, and the labeled ranking, generate **exactly {generate_number}** visual rubrics for ranking image quality.
{task_description_section}
{sample_content}

## Requirements
- Evaluation mode: {ranking_mode}. Rank multiple text-to-image outputs by relative quality; rank=1 means best
- Analyze differences between strong and weak images in prompt adherence, detail completeness, realism, composition, artifacts, and aesthetics
- The criteria must establish a clear quality gradient across all candidate images and avoid position bias
- Do not rank the current images; only output reusable ranking criteria
- Generate exactly {generate_number} rubrics

## Output Format
{{
    "rubrics": [
        "Detailed description of the first ranking rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were generated"
}}
"""

T2I_POINTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准，对输入的生成图片进行 pointwise 评分。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

评分范围: {min_score} 到 {max_score}，必须输出整数。

## 输出格式
{{
    "score": 分数值,
    "reason": "评分依据，说明图片如何符合或不符合各项标准"
}}
"""

T2I_POINTWISE_EVALUATION_PROMPT_EN = """
Score the generated image using the evaluation rubrics.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

Score range: {min_score} to {max_score}. The score must be an integer.

## Output Format
{{
    "score": score_value,
    "reason": "Reasoning explaining how the image satisfies or violates the rubrics"
}}
"""

T2I_LISTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准，对输入的 {num_outputs} 张生成图片进行 {ranking_mode} 排序。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

## 任务要求
- 只对候选生成图片排序，rank=1 表示质量最好，rank={num_outputs} 表示最差
- rank 数组必须恰好包含 {num_outputs} 个值，并按图片原始顺序对应 Image 1、Image 2 ...
- rank 值必须是从 1 到 {num_outputs} 的连续整数，不能并列
- 判断时重点关注文本提示遵循度、细节完整性、真实感、伪影控制、构图和美学
- 不要受图片排列顺序影响

## 输出格式
{{
    "rank": [每张候选图片对应的rank值],
    "reason": "详细说明每张图片的质量判断和排序依据"
}}
"""

T2I_LISTWISE_EVALUATION_PROMPT_EN = """
Rank the {num_outputs} generated images using the evaluation rubrics.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

## Requirements
- Rank only the candidate generated images. rank=1 is best and rank={num_outputs} is worst
- The rank array must contain exactly {num_outputs} values, corresponding to Image 1, Image 2, ... in the original order
- Rank values must be consecutive integers from 1 to {num_outputs}; ties are not allowed
- Focus on prompt adherence, detail completeness, realism, artifact control, composition, and aesthetics
- Do not let image order influence your judgment

## Output Format
{{
    "rank": [rank_value_for_each_candidate_image],
    "reason": "Detailed explanation of the quality assessment and ranking"
}}
"""

T2I_POINTWISE_REVISION_PROMPT_ZH = """
之前生成的 pointwise 评分标准验证失败。请基于输入图片、标注分数和反馈，生成**恰好 {generate_number} 个**改进后的评分标准。
{task_description_section}
{sample_content}

## 之前的评分标准
{rubrics}

## 失败反馈
{feedback}

## 改进要求
- 明确为什么旧标准无法给出正确分数
- 强化提示遵循、主体/属性/关系、视觉自然度、伪影和美学的可判别性
- 标准要能稳定区分 {min_score}-{max_score} 的整数分数档位

## 输出格式
{{
    "rubrics": [
        "改进后的第一个评分标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "改进原因"
}}
"""

T2I_POINTWISE_REVISION_PROMPT_EN = """
The previous pointwise scoring rubrics failed validation. Generate **exactly {generate_number}** improved scoring rubrics based on the image, labeled score, and feedback.
{task_description_section}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback
{feedback}

## Improvement Requirements
- Identify why the previous rubrics did not lead to the correct score
- Improve discriminative power for prompt adherence, subjects/attributes/relations, visual naturalness, artifacts, and aesthetics
- The rubrics must reliably distinguish integer score levels from {min_score} to {max_score}

## Output Format
{{
    "rubrics": [
        "First improved scoring rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were improved"
}}
"""

T2I_LISTWISE_REVISION_PROMPT_ZH = """
之前生成的排序标准验证失败。请基于输入的 {num_outputs} 张图片、标注排序和反馈，生成**恰好 {generate_number} 个**改进后的排序标准。
{task_description_section}
{sample_content}

## 之前的排序标准
{rubrics}

## 失败反馈
{feedback}

## 改进要求
- 分析期望排序与实际排序的差异
- 强化能够区分所有候选图片质量层次的标准
- 标准应减少位置偏见和主观歧义，并覆盖提示遵循、真实感、伪影、构图、美学等维度

## 输出格式
{{
    "rubrics": [
        "改进后的第一个排序标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "改进原因"
}}
"""

T2I_LISTWISE_REVISION_PROMPT_EN = """
The previous ranking rubrics failed validation. Generate **exactly {generate_number}** improved ranking rubrics based on the {num_outputs} images, labeled ranking, and feedback.
{task_description_section}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback
{feedback}

## Improvement Requirements
- Analyze the gap between the expected and actual ranking
- Strengthen criteria that distinguish all candidate image quality levels
- Reduce position bias and ambiguity while covering prompt adherence, realism, artifacts, composition, and aesthetics

## Output Format
{{
    "rubrics": [
        "First improved ranking rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were improved"
}}
"""


# ========== EvalVerse Text-to-Video Prompts ==========
# Aligned with the SIGGRAPH Asia 2026 EvalVerse taxonomy:
#   7 cinematic aspects -> 18 main dims -> 45 sub dims -> 196 rationales.
# Reward is computed under expert-calibrated Chain-of-Thought with
# Self-Reflection and Context-Aware Gating (Sec. 6 of the paper).

_EVALVERSE_ASPECTS_LINE_ZH = (
    "视觉概念设计(Visual Concept Design)、表演(Acting)、摄影(Cinematography)、"
    "美学(Aesthetics)、情感(Affectivity)、多镜头(Multi-Shot)、声音设计(Sound Design)"
)
_EVALVERSE_ASPECTS_LINE_EN = (
    "Visual Concept Design, Acting, Cinematography, Aesthetics, Affectivity, "
    "Multi-Shot, Sound Design"
)


def _build_evalverse_dim_block(language: LanguageEnum, target_main_dim: str | None) -> str:
    """Build the per-main-dim instruction block injected into T2V prompts.

    When ``target_main_dim`` is one of the 18 canonical EvalVerse main
    dimensions, the returned block enumerates that dim's sub-dimensions and
    instructs the LLM to focus rubrics there. Otherwise (e.g. v_arr_mvp's
    "let rubrics emerge from data" setting), an empty string is returned and
    the prompt falls back to the task_description-driven path.
    """
    if not target_main_dim:
        return ""
    canonical = str(target_main_dim).strip().lower().replace(" ", "_").replace("-", "_")
    if canonical not in EVALVERSE_MAIN_DIMENSIONS:
        # Unknown dimension -> behave like v_arr_mvp (no forced taxonomy).
        return ""
    sub_dims = EVALVERSE_MAIN_TO_SUB_DIMENSIONS.get(canonical, ())
    # Locate the parent aspect for human-readable framing.
    parent_aspect = next(
        (
            aspect
            for aspect, mains in EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS.items()
            if canonical in mains
        ),
        "unknown",
    )
    sub_dims_pretty = ", ".join(s.replace("_", " ") for s in sub_dims) if sub_dims else "(no sub-dimensions)"
    if language == LanguageEnum.ZH:
        return textwrap.dedent(
            f"""
            ## EvalVerse 目标维度（必须严格遵守）
            - 父级 aspect: `{parent_aspect}`
            - 当前 main dim: `{canonical}`
            - 该 main dim 下的 sub-dimensions: {sub_dims_pretty}

            生成的 rubric 必须**全部聚焦于该 main dim**：
            - 每个 sub-dim 至少需要 1 条 rubric 显式覆盖
            - 重要的 sub-dim 可写 2 条（不同观察角度）
            - 不要为该 main dim 之外的 EvalVerse 维度生成 rubric
            - 标注路径写为 `[<aspect> > {canonical} > <sub_dim>]` 的形式
            """
        ).strip()
    return textwrap.dedent(
        f"""
        ## EvalVerse Target Dimension (strict)
        - Parent aspect: `{parent_aspect}`
        - Current main dim: `{canonical}`
        - Sub-dimensions under this main dim: {sub_dims_pretty}

        All generated rubrics MUST focus on this main dimension:
        - Cover EACH sub-dim with at least one rubric
        - Important sub-dims may have up to 2 rubrics (different observation angles)
        - Do NOT generate rubrics that target other EvalVerse main dims
        - Use the path tag `[<aspect> > {canonical} > <sub_dim>]`
        """
    ).strip()


T2V_POINTWISE_GENERATION_PROMPT_ZH = """
请你扮演资深电影级视频评估专家。基于文本提示、生成视频和专家标注分数，生成至少 {generate_number} 条用于单视频 pointwise 评分的 rubric。
{task_description_section}
{evalverse_dim_block}
{sample_content}

## 第 0 步：根本失败模式总结（必须先做）
在写 rubric 前，先**抽象出 3-5 条根本性失败模式（root failure modes）**：
- 描述"什么样的根本性问题会让这一类视频从 5 分掉到 1 分"
- 失败模式应当是**质变性的**，不是细枝末节（例如"主体 ID 在镜头中漂移"而非"第 2 帧噪点偏多"）
- 失败模式将驱动 rubric 设计，使 rubric 抽象自数据但不过拟合单样本

## 任务要求
- 评估模式: Pointwise，对一段生成视频独立评分；最终得分由各 rubric 整数判定按权重加权得到，可保留一位小数
- rubric 数量: **至少 {generate_number} 条，无上限**；按上下文复杂度自决数量；优先保证覆盖度而非简洁
- 每条 rubric 都要写成可观察、可操作的具体细则；标明所属维度路径（如 `[Aesthetics > Visual Quality > Temporal Consistency]`，若 task_description 要求"无 benchmark 标签"则可省路径）
- **必须自带固定权重**：每条 rubric 末尾以 `(weight=0.NN)` 给出权重，**所有权重之和=1.0**（保留两位小数允许微小舍入）
- 严格执行 Context-Aware Gating：当 prompt/参考视频未要求某维度（如无对白则不评 Lip-Sync）时，rubric 必须显式写"若上下文不涉及，则跳过此项且不计入加权"
- 严格执行 Self-Reflection：rubric 中要求评委先列**可观察证据**（帧序号、运动方向、光照变化等具体感知线索）再给分
- 只输出 rubric 文本，不要给当前视频打分

## 输出格式
{{
    "rubrics": [
        "<具体 rubric 文本，含 Self-Reflection 提示与 Gating 条件> (weight=0.NN)",
        "<另一条 rubric> (weight=0.NN)",
        "..."
    ],
    "reason": "为什么这些 rubric 能复现该标注分数",
    "metadata": {{
        "failure_modes": [
            "失败模式 1：……",
            "失败模式 2：……"
        ],
        "weight_sum_check": "0.NN+0.NN+...=1.00"
    }}
}}
"""

T2V_POINTWISE_GENERATION_PROMPT_EN = """
You are a cinematic video evaluation expert. Based on the text prompt, the generated video, and the labeled expert score, generate **at least {generate_number}** rubrics for pointwise scoring.
{task_description_section}
{evalverse_dim_block}
{sample_content}

## Step 0: Root Failure Mode Summary (mandatory pre-step)
Before writing rubrics, abstract **3-5 root failure modes** that explain what would drop a video of this kind from 5 to 1:
- Failure modes must be **qualitative / structural**, not nitpicky (e.g. "subject identity drifts across shots" rather than "frame 2 has slight noise").
- These failure modes drive the rubric design so rubrics generalize from data without overfitting a single sample.

## Requirements
- Evaluation mode: Pointwise. Score one generated video on a continuous 1.0-5.0 scale; the final score is a *weighted average* of per-rubric integer verdicts (one decimal place is allowed).
- Number of rubrics: **at least {generate_number}, no upper bound**. Decide the count from context complexity; prefer coverage over brevity.
- Each rubric must be observable and actionable; include a dimension path such as `[Aesthetics > Visual Quality > Temporal Consistency]` (if the task_description requests "no benchmark tagging", the path may be omitted).
- **Each rubric must carry a fixed weight**: append `(weight=0.NN)` at the end of every rubric, and **all weights MUST sum to 1.0** (two-decimal precision; minor rounding tolerated).
- Enforce Context-Aware Gating: if the prompt/reference do not warrant a sub-dimension (e.g. silent clip -> Lip-Sync), the rubric must say "skip this item and exclude it from the weighted average".
- Enforce Self-Reflection: each rubric must require the judge to enumerate observable evidence (frame indices, motion direction, lighting cues, audio cues) before scoring.
- Output rubric text only; do not score the current video.

## Output Format
{{
    "rubrics": [
        "<concrete rubric text with Self-Reflection prompt and Gating clause> (weight=0.NN)",
        "<another rubric> (weight=0.NN)",
        "..."
    ],
    "reason": "Why these rubrics recover the labeled expert score",
    "metadata": {{
        "failure_modes": [
            "Failure mode 1: ...",
            "Failure mode 2: ..."
        ],
        "weight_sum_check": "0.NN+0.NN+...=1.00"
    }}
}}
"""

T2V_LISTWISE_GENERATION_PROMPT_ZH = """
请你扮演资深电影级视频评估专家，基于文本提示、{num_outputs} 段生成视频以及专家标注 Win/Lose 偏好，生成至少 {generate_number} 条用于 {ranking_mode} 排序的 rubric。
{task_description_section}
{evalverse_dim_block}
{sample_content}

## 第 0 步：根本失败模式总结（必须先做）
抽象出 3-5 条**根本性的胜负差异模式**：什么样的根本性差异会让一段视频明显胜出或落败？这些模式驱动 rubric 设计，避免 rubric 过拟合单个样本。

## 任务要求
- 评估模式: {ranking_mode}，rank=1 为最优（对应 Bradley-Terry 偏好中的 Vw）
- rubric 数量: **至少 {generate_number} 条，无上限**；按上下文复杂度自决数量
- 每条 rubric 必须可观察、可在两段视频之间形成区分；可选标明维度路径（如 `[Acting > Action > Action Tension]`，若 task_description 要求不带 benchmark 标签则可省）
- **每条 rubric 必须自带固定权重**：末尾以 `(weight=0.NN)` 标出；所有权重之和=1.0
- 显式包含 Self-Reflection（先列**可观察证据**再判优劣）与 Context-Aware Gating（无关维度跳过且不计入权重）
- 只输出排序标准，不要直接对当前视频排序

## 输出格式
{{
    "rubrics": [
        "<排序标准文本，含 Self-Reflection 步骤与 Gating 条件> (weight=0.NN)",
        "..."
    ],
    "reason": "为什么这些 rubric 能复现 Win/Lose 偏好",
    "metadata": {{
        "failure_modes": ["差异模式 1：……", "差异模式 2：……"],
        "weight_sum_check": "0.NN+0.NN+...=1.00"
    }}
}}
"""

T2V_LISTWISE_GENERATION_PROMPT_EN = """
You are a cinematic video evaluation expert. Based on the text prompt, {num_outputs} candidate videos, and the labeled expert Win/Lose preference, generate **at least {generate_number}** rubrics for {ranking_mode} ranking.
{task_description_section}
{evalverse_dim_block}
{sample_content}

## Step 0: Root Distinction Mode Summary (mandatory pre-step)
Abstract **3-5 root distinction modes**: what root differences make one video clearly win over another? These modes drive rubric design and prevent overfitting to a single sample.

## Requirements
- Evaluation mode: {ranking_mode}. rank=1 is best (corresponds to Vw in Bradley-Terry preference).
- Number of rubrics: **at least {generate_number}, no upper bound**. Decide the count from context complexity.
- Each rubric must be observable and discriminative between candidate videos; optionally tag a dimension path (e.g. `[Acting > Action > Action Tension]`; the path may be omitted if the task_description forbids benchmark tagging).
- **Each rubric MUST carry a fixed weight**: append `(weight=0.NN)`; all weights MUST sum to 1.0.
- Include Self-Reflection (list **observable evidence** before judging) and Context-Aware Gating (skip irrelevant sub-dims; gated rubrics do not count toward the weighted comparison).
- Output rubric text only; do not rank the current videos.

## Output Format
{{
    "rubrics": [
        "<ranking criterion with Self-Reflection and Gating> (weight=0.NN)",
        "..."
    ],
    "reason": "Why these rubrics recover the expert Win/Lose preference",
    "metadata": {{
        "failure_modes": ["Distinction mode 1: ...", "Distinction mode 2: ..."],
        "weight_sum_check": "0.NN+0.NN+...=1.00"
    }}
}}
"""

T2V_POINTWISE_EVALUATION_PROMPT_ZH = """
请基于评估标准对输入视频进行 pointwise 评分，采用专家校准的 Chain-of-Thought（CoT）流程。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

## 评分维度（每条 rubric 独立给一个 1-5 整数 verdict）
- 5 = 优秀（excellent）：本条 rubric 描述的能力被充分满足，几乎无可挑剔
- 4 = 良好（good）：本条 rubric 整体被满足，仅有一些次要瑕疵
- 3 = 一般（acceptable）：本条 rubric 部分满足，有明显但非致命的不足
- 2 = 较差（poor）：本条 rubric 大部分未满足，存在显著缺陷
- 1 = 不可接受（unacceptable）：本条 rubric 完全未满足或严重违反

## 关键约束（**严禁违反**）
- **每条 rubric 独立判定**：rubric A 的失败**不应**导致 rubric B、C、D 都被压低。每条 rubric 只评估它自己描述的能力。
- **不要因为一个细节缺陷就给 1 或 2**：先看这条 rubric 整体表现，再看细节程度。一处小瑕疵 ≈ 4，多处明显瑕疵 ≈ 3。
- **不要默认给 2-3**：保留 5 给真正优秀的实现，保留 1 给真正失败的实现，让分数充分使用 1-5 全区间。
- **不要计算最终加权分**：你只需要输出每条 rubric 的整数 verdict。最终 score 由系统在你给出 per_rubric 后做等权平均。

## CoT 步骤（必须严格按顺序执行）
1. **感知证据收集**：对每条 rubric 列出 2-3 条**可观察证据**（帧索引、运动方向、光影位置、镜头语言等）。证据**必须中性**：禁出现 "1/2/3/4/5 / 满足 / 不满足 / 通过 / 未通过" 等结论词，只描述看到了什么。
2. **Self-Reflection**：检查上述证据是否存在幻觉，删除任何无法证实的证据。
3. **逐条整数判定**：对每条 rubric，依据感知证据给出 1-5 的整数 verdict，遵循上面的评分维度定义和关键约束。

## 输出格式（系统将自动等权聚合 verdict 为最终 score）
{{
    "score": <占位，可填 0；最终 score 由系统计算，**你不需要算**>,
    "reason": "对整体表现的简短说明（≤80 字）",
    "metadata": {{
        "per_rubric": [
            {{
                "rubric_index": 1,
                "evidence": "中性描述的可观察证据（≤40 字，禁出现数字判定与结论词）",
                "verdict": 4
            }}
        ]
    }}
}}
"""

T2V_POINTWISE_EVALUATION_PROMPT_EN = """
Score the generated video using the rubrics below with an expert-calibrated Chain-of-Thought (CoT) process.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

## Scoring Scale (assign each rubric an INDEPENDENT integer 1-5 verdict)
- 5 = Excellent: this rubric's quality is fully met; nearly flawless
- 4 = Good: this rubric is overall met; only minor blemishes
- 3 = Acceptable: partially met; notable but non-fatal shortcomings
- 2 = Poor: largely unmet; significant defects on this rubric
- 1 = Unacceptable: completely unmet or severely violated

## Hard Constraints (DO NOT violate)
- **Independent per-rubric scoring**: a failure on rubric A must NOT drag down rubrics B/C/D. Each rubric is judged ONLY on the capability it describes.
- **Do NOT default to 1 or 2 because of one localized flaw**: judge the rubric's overall behavior first, then the severity. One minor blemish ~ 4; several notable blemishes ~ 3.
- **Do NOT collapse to 2-3**: reserve 5 for genuinely excellent execution and 1 for genuine failures. Use the full 1-5 range.
- **Do NOT compute the final weighted score**: just emit an integer verdict per rubric. The system will average them with equal weights afterwards.

## CoT Procedure (mandatory, in order)
1. **Perceptual Evidence Collection**: for each rubric, list 2-3 **observable cues** (frame indices, motion direction, lighting position, camera language, etc.). Evidence MUST be in a **neutral descriptive register**: forbid "1/2/3/4/5 / satisfies / fails / passes / violates". Describe what is seen, NOT the verdict.
2. **Self-Reflection**: verify each cue is grounded; remove any unverifiable cue.
3. **Per-rubric Verdict**: for every rubric, emit an integer 1-5 based on the evidence, following the scale and constraints above.

## Output Format (system will average per_rubric verdicts equally)
{{
    "score": <placeholder, set to 0; the final score is computed by the system, you do NOT need to compute it>,
    "reason": "Short overall justification (<=80 chars)",
    "metadata": {{
        "per_rubric": [
            {{
                "rubric_index": 1,
                "evidence": "neutral observable cues (<=40 chars; no digit verdicts, no conclusion words)",
                "verdict": 4
            }}
        ]
    }}
}}
"""

T2V_LISTWISE_EVALUATION_PROMPT_ZH = """
请根据下列评估标准，对 {num_outputs} 段生成视频进行 {ranking_mode} 排序，采用专家校准的 CoT 流程。
{task_description_section}
评估标准（每条末尾的 `(weight=0.NN)` 是固定权重，**严禁修改**）:
{rubrics}

{sample_content}

## CoT 步骤（必须严格按顺序执行）
1. **感知证据收集**：对每段视频按 rubric 列出 2-3 条**可观察证据**（帧索引、运动方向、光影位置、镜头语言等）。证据**必须中性**：禁出现 "Win/Lose/胜/负/优于/差于" 等结论词，只描述看到了什么。
2. **Self-Reflection**：剔除幻觉性描述，确保证据来自视频本身或参考材料。
3. **Context-Aware Gating**：跳过 prompt/参考不涉及的 rubric，跳过的 rubric 权重均匀分摊到存活 rubric。
4. **逐 rubric 偏好判定**：对每条存活 rubric，给出该 rubric 下的 video preference vector（list of int，长度 = {num_outputs}，rank=1 为该 rubric 下最佳）。
5. **偏序合成**：按权重加权合成最终 rank：每段视频的 weighted rank score = Σ(weight_i × rubric_rank_i)，再按 score 升序得到最终 rank。
   - 与 Bradley-Terry 偏好一致：Win=rank 1，Lose=rank {num_outputs}
   - rank 数组按视频原始顺序对应 Video 1、Video 2、...
   - rank 值必须是 1 到 {num_outputs} 的连续整数，不能并列

## 输出格式
{{
    "rank": [每段候选视频的最终 rank 值],
    "reason": "最终排序的简短说明（≤80 字）",
    "metadata": {{
        "per_rubric": [
            {{"rubric_index": 1, "rubric_path": "...", "evidence_per_video": ["...", "..."], "gated": false, "rubric_rank": [1, 2], "weight": 0.30}}
        ],
        "gating_decisions": "被跳过的 rubric 列表与原因（若无则填 'none'）",
        "aggregation": "加权合成过程，例如 video1 score = 0.30*1+0.20*2+..."
    }}
}}
"""

T2V_LISTWISE_EVALUATION_PROMPT_EN = """
Rank the {num_outputs} generated videos using the rubrics below in {ranking_mode} mode, with an expert-calibrated CoT process.
{task_description_section}
Evaluation Rubrics (the trailing `(weight=0.NN)` on each rubric is its FIXED weight; do NOT modify):
{rubrics}

{sample_content}

## CoT Procedure (mandatory, in order)
1. **Perceptual Evidence Collection**: per video and per rubric, list 2-3 **observable cues** (frame indices, motion direction, lighting position, camera language). Evidence MUST be **neutral**: forbidden words include "Win/Lose/wins/loses/better/worse"; describe only what is seen.
2. **Self-Reflection**: discard hallucinated descriptions; cite the actual video or reference material.
3. **Context-Aware Gating**: bypass rubrics whose context is not warranted; gated rubrics have their weight redistributed uniformly to the survivors.
4. **Per-rubric Preference**: for every surviving rubric, emit a per-rubric video preference vector (list of int, length = {num_outputs}, rank=1 = best under that rubric).
5. **Weighted Aggregation**: each video's weighted rank score = Σ(weight_i × rubric_rank_i); sort ascending to get the final rank.
   - Consistent with Bradley-Terry preference: Win = rank 1, Lose = rank {num_outputs}
   - rank array follows the original video order (Video 1, Video 2, ...)
   - rank values must be consecutive integers 1..{num_outputs}; no ties

## Output Format
{{
    "rank": [final_rank_value_per_candidate_video],
    "reason": "Short justification of the final ranking (<=80 chars)",
    "metadata": {{
        "per_rubric": [
            {{"rubric_index": 1, "rubric_path": "...", "evidence_per_video": ["...", "..."], "gated": false, "rubric_rank": [1, 2], "weight": 0.30}}
        ],
        "gating_decisions": "list of gated rubrics with reasons, or 'none'",
        "aggregation": "weighted aggregation trace, e.g. video1 score = 0.30*1+0.20*2+..."
    }}
}}
"""

T2V_POINTWISE_REVISION_PROMPT_ZH = """
之前生成的 pointwise 评分标准未能复现专家分数。请基于视频、参考材料、专家分数与失败反馈，生成至少 {generate_number} 条改进后的 rubric。
{task_description_section}
{evalverse_dim_block}
{sample_content}

## 之前的评分标准
{rubrics}

## 失败反馈（包含上一轮逐条 verdict / weight / 加权和与 ground-truth 对比）
{feedback}

## 改进要求
- 逐条针对失败反馈：指明哪些 rubric 在此样本上判定偏高、偏低或被错误 Gate
- 加入更具体的 Self-Reflection 证据要求（例如显式列出帧索引或运动方向）与 Context-Aware Gating 触发条件
- rubric 仍需自带 `(weight=0.NN)`，权重和=1.0；可以重新分配权重，让此前误判更严重的 rubric 权重相应下降
- rubric 数量不少于 {generate_number} 条，无上限；按需新增
- 标准应稳定区分 {min_score}-{max_score} 范围内的连续分数档位（必要时给出含一位小数的锚点，如 4.3 / 2.7）

## 输出格式
{{
    "rubrics": [
        "<改进后的 rubric 文本> (weight=0.NN)",
        "..."
    ],
    "reason": "改进点说明，逐条对应失败反馈",
    "metadata": {{
        "failure_modes_revised": ["..."],
        "weight_sum_check": "0.NN+0.NN+...=1.00"
    }}
}}
"""

T2V_POINTWISE_REVISION_PROMPT_EN = """
The previous pointwise rubrics failed to recover the expert score. Using the video, references, expert score, and failure feedback, generate **at least {generate_number}** improved rubrics.
{task_description_section}
{evalverse_dim_block}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback (includes the previous round's per-rubric verdicts / weights / weighted sum vs. ground truth)
{feedback}

## Improvement Requirements
- Address the failure feedback rubric-by-rubric: identify which rubrics over-rated, under-rated, or were wrongly gated on this sample.
- Strengthen Self-Reflection evidence requirements (e.g. require explicit frame indices or motion direction) and Context-Aware Gating triggers.
- Each rubric MUST still carry `(weight=0.NN)` with weights summing to 1.0; you may redistribute weights to down-weight rubrics that misled the aggregate.
- Number of rubrics: at least {generate_number}, no upper bound; add new rubrics as needed.
- Each rubric must reliably separate continuous score levels in [{min_score}, {max_score}] (use one-decimal anchors when needed).

## Output Format
{{
    "rubrics": [
        "<improved rubric text> (weight=0.NN)",
        "..."
    ],
    "reason": "Per-rubric justification mapped to the failure feedback",
    "metadata": {{
        "failure_modes_revised": ["..."],
        "weight_sum_check": "0.NN+0.NN+...=1.00"
    }}
}}
"""

T2V_LISTWISE_REVISION_PROMPT_ZH = """
之前生成的排序标准未能复现 Win/Lose 偏好。请基于 {num_outputs} 段视频、参考材料、专家偏好与失败反馈，生成至少 {generate_number} 条改进后的 rubric。
{task_description_section}
{evalverse_dim_block}
{sample_content}

## 之前的排序标准
{rubrics}

## 失败反馈（包含上一轮逐条 rubric 在两段视频上的偏好向量与权重）
{feedback}

## 改进要求
- 逐条针对失败反馈：指明哪些 rubric 在 Win/Lose 区分上失效
- 增强 Self-Reflection 证据要求（强迫罗列帧级 / 运动 / 音频证据），并明确 Context-Aware Gating 触发条件
- rubric 仍需自带 `(weight=0.NN)`，权重和=1.0；可重新分配权重，让此前误判的 rubric 权重相应下降
- rubric 数量不少于 {generate_number} 条，无上限
- rubric 应减少位置偏见与镜头顺序敏感性

## 输出格式
{{
    "rubrics": [
        "<改进后的排序标准> (weight=0.NN)",
        "..."
    ],
    "reason": "改进点说明，逐条对应失败反馈",
    "metadata": {{
        "failure_modes_revised": ["..."],
        "weight_sum_check": "0.NN+0.NN+...=1.00"
    }}
}}
"""

T2V_LISTWISE_REVISION_PROMPT_EN = """
The previous ranking rubrics failed to recover the Win/Lose preference. Using the {num_outputs} videos, references, expert preference, and failure feedback, generate **at least {generate_number}** improved rubrics.
{task_description_section}
{evalverse_dim_block}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback (per-rubric video preference vectors and weights from the previous round)
{feedback}

## Improvement Requirements
- Address the failure feedback rubric-by-rubric: identify which rubrics failed to discriminate Win from Lose.
- Strengthen Self-Reflection (force frame-level / motion / audio evidence) and Context-Aware Gating triggers.
- Each rubric MUST still carry `(weight=0.NN)` with weights summing to 1.0; redistribute weights to down-weight rubrics that misled the aggregation.
- Number of rubrics: at least {generate_number}, no upper bound.
- Rubrics should reduce position bias and shot-order sensitivity.

## Output Format
{{
    "rubrics": [
        "<improved ranking rubric> (weight=0.NN)",
        "..."
    ],
    "reason": "Per-rubric justification mapped to the failure feedback",
    "metadata": {{
        "failure_modes_revised": ["..."],
        "weight_sum_check": "0.NN+0.NN+...=1.00"
    }}
}}
"""


# ========== Placeholder consistency self-check ==========
# Lightweight import-time sanity check: every ZH/EN T2V prompt pair must
# expose the same set of `{placeholder}` tokens. Mismatches are likely typos
# introduced when one variant is updated without the other (a common mistake
# during the EvalVerse / v_arr_mvp prompt revisions). Failures are logged as
# warnings so that downstream training jobs are not aborted, but the warning
# is loud enough to catch in CI / dev runs.
def _extract_real_placeholders(template: str) -> set[str]:
    """Extract single-brace `{name}` placeholders, ignoring JSON `{{ }}` escapes."""
    import re

    # Strip escaped double-braces first so they do not contaminate matches.
    stripped = template.replace("{{", "").replace("}}", "")
    return set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", stripped))


def _assert_zh_en_placeholder_parity() -> None:
    pairs = (
        ("T2V_POINTWISE_GENERATION_PROMPT", T2V_POINTWISE_GENERATION_PROMPT_ZH, T2V_POINTWISE_GENERATION_PROMPT_EN),
        ("T2V_POINTWISE_EVALUATION_PROMPT", T2V_POINTWISE_EVALUATION_PROMPT_ZH, T2V_POINTWISE_EVALUATION_PROMPT_EN),
        ("T2V_POINTWISE_REVISION_PROMPT", T2V_POINTWISE_REVISION_PROMPT_ZH, T2V_POINTWISE_REVISION_PROMPT_EN),
        ("T2V_LISTWISE_GENERATION_PROMPT", T2V_LISTWISE_GENERATION_PROMPT_ZH, T2V_LISTWISE_GENERATION_PROMPT_EN),
        ("T2V_LISTWISE_EVALUATION_PROMPT", T2V_LISTWISE_EVALUATION_PROMPT_ZH, T2V_LISTWISE_EVALUATION_PROMPT_EN),
        ("T2V_LISTWISE_REVISION_PROMPT", T2V_LISTWISE_REVISION_PROMPT_ZH, T2V_LISTWISE_REVISION_PROMPT_EN),
    )
    for name, zh, en in pairs:
        zh_keys = _extract_real_placeholders(zh)
        en_keys = _extract_real_placeholders(en)
        if zh_keys != en_keys:
            logger.warning(
                "Placeholder set mismatch in {}: zh-only={}, en-only={}",
                name,
                sorted(zh_keys - en_keys),
                sorted(en_keys - zh_keys),
            )


_assert_zh_en_placeholder_parity()


# ========== Image Editing Prompts ==========

EDIT_POINTWISE_GENERATION_PROMPT_ZH = """
请基于编辑指令、原图（如已提供）、编辑结果和标注分数，生成**恰好 {generate_number} 个**用于图像编辑结果 pointwise 评分的 rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: Pointwise，对一个编辑结果独立评分，分数范围为 {min_score}-{max_score}，输出必须是整数
- 若提供 Image BASE，必须以原图为参照；若未提供，则根据编辑指令和编辑结果评估
- 标准应覆盖编辑指令执行、原图内容保真、局部/全局一致性、自然融合、伪影控制和美学质量
- 只输出评分标准，不要直接给当前编辑结果打分
- 必须严格生成 {generate_number} 个标准

## 输出格式
{{
    "rubrics": [
        "第一个图像编辑评分标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "生成这些标准的原因"
}}
"""

EDIT_POINTWISE_GENERATION_PROMPT_EN = """
Based on the edit instruction, the original image if provided, the edited result, and the labeled score, generate **exactly {generate_number}** rubrics for pointwise image-edit scoring.
{task_description_section}
{sample_content}

## Requirements
- Evaluation mode: Pointwise. Score one edited image independently on an integer scale from {min_score} to {max_score}
- If Image BASE is provided, use it as the reference; otherwise judge from the instruction and edited image
- Cover instruction execution, fidelity to the source image, local/global consistency, natural blending, artifact control, and aesthetic quality
- Do not score the current edit; only output reusable scoring criteria
- Generate exactly {generate_number} rubrics

## Output Format
{{
    "rubrics": [
        "First image-edit scoring rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were generated"
}}
"""

EDIT_LISTWISE_GENERATION_PROMPT_ZH = """
请基于编辑指令、原图（如已提供）、{num_outputs} 个编辑结果及标注排序，生成**恰好 {generate_number} 个**用于图像编辑结果排序的 rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: {ranking_mode}，只对编辑结果排序；rank=1 表示最佳编辑结果
- 若提供 Image BASE，必须以原图为基准比较每个编辑结果；若未提供，则根据编辑指令和结果本身比较
- 标准应区分编辑指令执行、原图保真、未编辑区域保持、边界融合、光照/透视一致、伪影控制和美学质量
- 只输出排序标准，不要直接给当前编辑结果排序
- 必须严格生成 {generate_number} 个标准

## 输出格式
{{
    "rubrics": [
        "第一个图像编辑排序标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "生成这些排序标准的原因"
}}
"""

EDIT_LISTWISE_GENERATION_PROMPT_EN = """
Based on the edit instruction, the original image if provided, {num_outputs} edited results, and the labeled ranking, generate **exactly {generate_number}** rubrics for ranking image edits.
{task_description_section}
{sample_content}

## Requirements
- Evaluation mode: {ranking_mode}. Rank only the edited results; rank=1 means the best edit
- If Image BASE is provided, use it as the reference for every edited result; otherwise compare using the edit instruction and outputs
- Distinguish instruction execution, source fidelity, preservation of untouched regions, boundary blending, lighting/perspective consistency, artifacts, and aesthetic quality
- Do not rank the current edits; only output reusable ranking criteria
- Generate exactly {generate_number} rubrics

## Output Format
{{
    "rubrics": [
        "First image-edit ranking rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were generated"
}}
"""

EDIT_POINTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准，对输入的图像编辑结果进行 pointwise 评分。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

评分范围: {min_score} 到 {max_score}，必须输出整数。

## 输出格式
{{
    "score": 分数值,
    "reason": "评分依据，说明编辑结果如何满足或违反各项标准"
}}
"""

EDIT_POINTWISE_EVALUATION_PROMPT_EN = """
Score the edited image using the evaluation rubrics.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

Score range: {min_score} to {max_score}. The score must be an integer.

## Output Format
{{
    "score": score_value,
    "reason": "Reasoning explaining how the edit satisfies or violates the rubrics"
}}
"""

EDIT_LISTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准，对输入的 {num_outputs} 个编辑结果进行 {ranking_mode} 排序。
{task_description_section}
评估标准:
{rubrics}

{sample_content}

## 任务要求
- 只对编辑结果排序，不要把 Image BASE 纳入 rank
- rank=1 表示最佳编辑结果，rank={num_outputs} 表示最差
- rank 数组必须恰好包含 {num_outputs} 个值，并按编辑结果原始顺序对应 Edited Image 1、Edited Image 2 ...
- rank 值必须是从 1 到 {num_outputs} 的连续整数，不能并列
- 若提供 Image BASE，所有判断都必须参考原图；同时关注指令执行、原图保真、自然融合和伪影控制
- 不要受图片排列顺序影响

## 输出格式
{{
    "rank": [每个编辑结果对应的rank值],
    "reason": "详细说明每个编辑结果与原图/指令的匹配程度和排序依据"
}}
"""

EDIT_LISTWISE_EVALUATION_PROMPT_EN = """
Rank the {num_outputs} edited results using the evaluation rubrics.
{task_description_section}
Evaluation Rubrics:
{rubrics}

{sample_content}

## Requirements
- Rank only the edited results. Do not include Image BASE in the rank array
- rank=1 is the best edit and rank={num_outputs} is the worst
- The rank array must contain exactly {num_outputs} values, corresponding to Edited Image 1, Edited Image 2, ... in the original order
- Rank values must be consecutive integers from 1 to {num_outputs}; ties are not allowed
- If Image BASE is provided, all judgments must reference it; also consider instruction execution, source fidelity, natural blending, and artifact control
- Do not let image order influence your judgment

## Output Format
{{
    "rank": [rank_value_for_each_edited_result],
    "reason": "Detailed explanation of each edit's match to the source/instruction and the final ranking"
}}
"""

EDIT_POINTWISE_REVISION_PROMPT_ZH = """
之前生成的图像编辑 pointwise 评分标准验证失败。请基于输入、标注分数和反馈，生成**恰好 {generate_number} 个**改进后的评分标准。
{task_description_section}
{sample_content}

## 之前的评分标准
{rubrics}

## 失败反馈
{feedback}

## 改进要求
- 分析旧标准为什么无法给出正确分数
- 强化指令执行、原图保真、未编辑区域保持、自然融合、伪影控制和美学质量的分数区分能力
- 标准要能稳定区分 {min_score}-{max_score} 的整数分数档位

## 输出格式
{{
    "rubrics": [
        "改进后的第一个图像编辑评分标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "改进原因"
}}
"""

EDIT_POINTWISE_REVISION_PROMPT_EN = """
The previous pointwise image-edit scoring rubrics failed validation. Generate **exactly {generate_number}** improved scoring rubrics based on the inputs, labeled score, and feedback.
{task_description_section}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback
{feedback}

## Improvement Requirements
- Analyze why the previous rubrics did not lead to the correct score
- Improve score-level discrimination for instruction execution, source fidelity, preservation of untouched regions, natural blending, artifact control, and aesthetics
- The rubrics must reliably distinguish integer score levels from {min_score} to {max_score}

## Output Format
{{
    "rubrics": [
        "First improved image-edit scoring rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were improved"
}}
"""

EDIT_LISTWISE_REVISION_PROMPT_ZH = """
之前生成的图像编辑排序标准验证失败。请基于输入的 {num_outputs} 个编辑结果、标注排序和反馈，生成**恰好 {generate_number} 个**改进后的排序标准。
{task_description_section}
{sample_content}

## 之前的排序标准
{rubrics}

## 失败反馈
{feedback}

## 改进要求
- 分析期望排序与实际排序的差异
- 强化能够区分编辑结果质量层次的标准，尤其是指令执行、原图保真、局部融合、伪影和美学
- 如果提供原图，标准必须明确要求对 Image BASE 进行比较

## 输出格式
{{
    "rubrics": [
        "改进后的第一个图像编辑排序标准",
        "... 共 {generate_number} 个"
    ],
    "reason": "改进原因"
}}
"""

EDIT_LISTWISE_REVISION_PROMPT_EN = """
The previous image-edit ranking rubrics failed validation. Generate **exactly {generate_number}** improved ranking rubrics based on the {num_outputs} edited results, labeled ranking, and feedback.
{task_description_section}
{sample_content}

## Previous Rubrics
{rubrics}

## Validation Feedback
{feedback}

## Improvement Requirements
- Analyze the gap between the expected and actual ranking
- Strengthen criteria that distinguish edit quality levels, especially instruction execution, source fidelity, local blending, artifacts, and aesthetics
- If the original image is provided, the criteria must explicitly require comparison with Image BASE

## Output Format
{{
    "rubrics": [
        "First improved image-edit ranking rubric",
        "... exactly {generate_number} items"
    ],
    "reason": "Why these rubrics were improved"
}}
"""


def _prompt_pair(
    task_type: str,
    mode: GraderMode,
    prompt_kind: str,
) -> tuple[str, str]:
    task = normalize_task_type(task_type)
    is_pointwise = mode == GraderMode.POINTWISE

    if task == TASK_T2I and prompt_kind == "generation" and is_pointwise:
        return T2I_POINTWISE_GENERATION_PROMPT_ZH, T2I_POINTWISE_GENERATION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "generation":
        return T2I_LISTWISE_GENERATION_PROMPT_ZH, T2I_LISTWISE_GENERATION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "evaluation" and is_pointwise:
        return T2I_POINTWISE_EVALUATION_PROMPT_ZH, T2I_POINTWISE_EVALUATION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "evaluation":
        return T2I_LISTWISE_EVALUATION_PROMPT_ZH, T2I_LISTWISE_EVALUATION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "revision" and is_pointwise:
        return T2I_POINTWISE_REVISION_PROMPT_ZH, T2I_POINTWISE_REVISION_PROMPT_EN
    if task == TASK_T2I and prompt_kind == "revision":
        return T2I_LISTWISE_REVISION_PROMPT_ZH, T2I_LISTWISE_REVISION_PROMPT_EN

    # T2V family (T2V, R2V, multi-shot, video-with-sound) shares the EvalVerse
    # rubric prompts and CoT scaffolding. Differences live in task_description
    # and the inputs (reference images / audio cues) attached by media.py.
    t2v_family = {TASK_T2V, TASK_R2V, TASK_T2V_MULTISHOT, TASK_T2V_SOUND}
    if task in t2v_family and prompt_kind == "generation" and is_pointwise:
        return T2V_POINTWISE_GENERATION_PROMPT_ZH, T2V_POINTWISE_GENERATION_PROMPT_EN
    if task in t2v_family and prompt_kind == "generation":
        return T2V_LISTWISE_GENERATION_PROMPT_ZH, T2V_LISTWISE_GENERATION_PROMPT_EN
    if task in t2v_family and prompt_kind == "evaluation" and is_pointwise:
        return T2V_POINTWISE_EVALUATION_PROMPT_ZH, T2V_POINTWISE_EVALUATION_PROMPT_EN
    if task in t2v_family and prompt_kind == "evaluation":
        return T2V_LISTWISE_EVALUATION_PROMPT_ZH, T2V_LISTWISE_EVALUATION_PROMPT_EN
    if task in t2v_family and prompt_kind == "revision" and is_pointwise:
        return T2V_POINTWISE_REVISION_PROMPT_ZH, T2V_POINTWISE_REVISION_PROMPT_EN
    if task in t2v_family and prompt_kind == "revision":
        return T2V_LISTWISE_REVISION_PROMPT_ZH, T2V_LISTWISE_REVISION_PROMPT_EN

    if task == TASK_IMAGE_EDIT and prompt_kind == "generation" and is_pointwise:
        return EDIT_POINTWISE_GENERATION_PROMPT_ZH, EDIT_POINTWISE_GENERATION_PROMPT_EN
    if task == TASK_IMAGE_EDIT and prompt_kind == "generation":
        return EDIT_LISTWISE_GENERATION_PROMPT_ZH, EDIT_LISTWISE_GENERATION_PROMPT_EN
    if task == TASK_IMAGE_EDIT and prompt_kind == "evaluation" and is_pointwise:
        return EDIT_POINTWISE_EVALUATION_PROMPT_ZH, EDIT_POINTWISE_EVALUATION_PROMPT_EN
    if task == TASK_IMAGE_EDIT and prompt_kind == "evaluation":
        return EDIT_LISTWISE_EVALUATION_PROMPT_ZH, EDIT_LISTWISE_EVALUATION_PROMPT_EN
    if task == TASK_IMAGE_EDIT and prompt_kind == "revision" and is_pointwise:
        return EDIT_POINTWISE_REVISION_PROMPT_ZH, EDIT_POINTWISE_REVISION_PROMPT_EN
    return EDIT_LISTWISE_REVISION_PROMPT_ZH, EDIT_LISTWISE_REVISION_PROMPT_EN


def _build_template(task_type: str, mode: GraderMode, prompt_kind: str) -> PromptTemplate:
    zh_prompt, en_prompt = _prompt_pair(task_type, mode, prompt_kind)
    include_system = prompt_kind in {"generation", "revision"}
    zh_messages = []
    en_messages = []
    if include_system:
        if is_video_task(task_type):
            zh_messages.append(ChatMessage(role="system", content="你是一个专业的视频生成评估标准制定专家。"))
            en_messages.append(
                ChatMessage(role="system", content="You are a professional video generation evaluation criteria expert."),
            )
        else:
            zh_messages.append(ChatMessage(role="system", content="你是一个专业的视觉评估标准制定专家。"))
            en_messages.append(ChatMessage(role="system", content="You are a professional visual evaluation criteria expert."))
    zh_messages.append(ChatMessage(role="user", content=textwrap.dedent(zh_prompt).strip()))
    en_messages.append(ChatMessage(role="user", content=textwrap.dedent(en_prompt).strip()))
    return PromptTemplate(messages={LanguageEnum.ZH: zh_messages, LanguageEnum.EN: en_messages})


def get_generation_template(
    grader_mode: GraderMode | str,
    task_type: str | None = TASK_T2I,
) -> PromptTemplate:
    mode = GraderMode(grader_mode)
    return _build_template(normalize_task_type(task_type), mode, "generation")


def get_evaluation_template(
    grader_mode: GraderMode | str,
    task_type: str | None = TASK_T2I,
) -> PromptTemplate:
    mode = GraderMode(grader_mode)
    return _build_template(normalize_task_type(task_type), mode, "evaluation")


def get_revision_template(
    grader_mode: GraderMode | str,
    task_type: str | None = TASK_T2I,
) -> PromptTemplate:
    mode = GraderMode(grader_mode)
    return _build_template(normalize_task_type(task_type), mode, "revision")


# Backward-compatible defaults.  New code should call get_*_template with an
# explicit task_type.
POINTWISE_GENERATION_TEMPLATE = get_generation_template(GraderMode.POINTWISE, TASK_T2I)
LISTWISE_GENERATION_TEMPLATE = get_generation_template(GraderMode.LISTWISE, TASK_T2I)
POINTWISE_EVALUATION_TEMPLATE = get_evaluation_template(GraderMode.POINTWISE, TASK_T2I)
LISTWISE_EVALUATION_TEMPLATE = get_evaluation_template(GraderMode.LISTWISE, TASK_T2I)
POINTWISE_REVISION_TEMPLATE = get_revision_template(GraderMode.POINTWISE, TASK_T2I)
LISTWISE_REVISION_TEMPLATE = get_revision_template(GraderMode.LISTWISE, TASK_T2I)


class RubricGenerationOutput(BaseModel):
    """Structured output expected from rubric generation/revision."""

    rubrics: List[str] = Field(description="List of generated rubrics")
    reason: str = Field(description="Reasoning for the generated rubrics")


class QuerySpecificRubricGenerator:
    """Generate, validate, and revise query-specific vision rubrics."""

    def __init__(
        self,
        model: BaseChatModel,
        grader_mode: GraderMode | str = GraderMode.POINTWISE,
        generate_number: int = 3,
        max_retries: int = 5,
        max_epochs: int = 3,
        min_score: int = 0,
        max_score: int = 4,
        language: LanguageEnum | str = LanguageEnum.ZH,
        task_description: str | None = None,
        task_type: str | None = TASK_T2I,
        validation_tolerance: float = 0.5,
    ):
        self.model = model
        self.grader_mode = GraderMode(grader_mode)
        self.language = LanguageEnum(language) if isinstance(language, str) else language
        self.task_type = normalize_task_type(task_type)
        self.generate_number = generate_number
        self.max_retries = max_retries
        self.max_epochs = max_epochs
        self.min_score = min_score
        self.max_score = max_score
        self.task_description = task_description
        # Tolerance used by `_validate_pointwise` when comparing the predicted
        # continuous score against the labeled continuous score. Default 0.5
        # corresponds to "within half a label band" on a 1-5 scale.
        self.validation_tolerance = float(validation_tolerance)

        self.generation_template = get_generation_template(self.grader_mode, self.task_type)
        self.evaluation_template = get_evaluation_template(self.grader_mode, self.task_type)
        self.revision_template = get_revision_template(self.grader_mode, self.task_type)

        logger.info(
            "QuerySpecificRubricGenerator initialized: mode={}, task_type={}, language={}",
            self.grader_mode.value,
            self.task_type,
            self.language.value,
        )

    async def generate_iterative(self, data: dict) -> Dict[str, Any]:
        rubrics = await self.generate(data)
        if not rubrics:
            return {
                "rubrics": [],
                "rubric_valid": False,
                "rubric_epoch": "0",
                "evaluation_result": {},
            }

        evaluation_result = {}
        for epoch in range(self.max_epochs):
            evaluation_result = await self.aevaluate(data, rubrics)
            is_correct = self.validate(data, evaluation_result)
            logger.debug(f"Epoch {epoch}: correctness = {is_correct}")

            if is_correct:
                return {
                    "rubrics": rubrics,
                    "rubric_valid": True,
                    "rubric_epoch": str(epoch),
                    "evaluation_result": evaluation_result,
                }

            feedback = self.generate_feedback(data, evaluation_result)
            revised_rubrics = await self.revise(data, rubrics, feedback)
            if not revised_rubrics:
                break
            rubrics = revised_rubrics

        return {
            "rubrics": rubrics,
            "rubric_valid": False,
            "rubric_epoch": str(self.max_epochs),
            "evaluation_result": evaluation_result,
        }

    async def generate(self, data: dict) -> List[str]:
        sample_content = self._format_data_context(data)
        media_paths = extract_media_paths(data, self.task_type)
        reference_images = extract_reference_images(data) if is_video_task(self.task_type) else {}
        num_outputs = max(1, count_outputs(data, self.task_type))
        task_description_section = self._format_task_description_section()

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def generate_rubrics():
            params = self._base_template_params(
                sample_content=sample_content,
                num_outputs=num_outputs,
                task_description_section=task_description_section,
                target_main_dim=self._extract_target_main_dim(data),
            )
            params.pop("language", None)
            messages = self.generation_template.format(language=self.language, **params)
            messages = attach_media_to_messages(
                messages,
                media_paths,
                self.task_type,
                has_base=has_base_media(data, self.task_type),
                reference_images=reference_images or None,
            )
            chat_response = await self.model.achat(
                messages=messages,
                structured_model=RubricGenerationOutput,
            )

            if not chat_response.parsed or "rubrics" not in chat_response.parsed:
                raise ValueError(f"Rubric generation returned invalid JSON: {chat_response.parsed}")

            rubrics = chat_response.parsed["rubrics"]
            if not rubrics:
                raise ValueError("No rubrics generated")
            return rubrics

        try:
            rubrics = await generate_rubrics()
            logger.debug(f"Generated {len(rubrics)} rubrics")
            return rubrics
        except Exception as e:
            # Surface the original exception wrapped by tenacity's RetryError.
            inner = getattr(e, "last_attempt", None)
            if inner is not None and inner.exception() is not None:
                root = inner.exception()
                logger.opt(exception=root).error(
                    "Failed to generate rubrics after {} attempts: {}: {}",
                    self.max_retries, type(root).__name__, root,
                )
            else:
                logger.opt(exception=e).error(
                    "Failed to generate rubrics after {} attempts: {}", self.max_retries, e,
                )
            return []

    async def aevaluate(self, data: dict, rubrics: List[str]) -> Dict[str, Any]:
        if self.grader_mode == GraderMode.POINTWISE:
            return await self._evaluate_pointwise(data, rubrics)
        return await self._evaluate_listwise(data, rubrics)

    def validate(self, data: dict, evaluation_result: Dict[str, Any]) -> bool:
        if self.grader_mode == GraderMode.POINTWISE:
            return self._validate_pointwise(data, evaluation_result)
        return self._validate_listwise(data, evaluation_result)

    async def revise(self, data: dict, rubrics: List[str], feedback: str) -> List[str]:
        sample_content = self._format_data_context(data)
        rubrics_text = self._format_rubrics_text(rubrics)
        media_paths = extract_media_paths(data, self.task_type)
        reference_images = extract_reference_images(data) if is_video_task(self.task_type) else {}
        num_outputs = max(1, count_outputs(data, self.task_type))
        task_description_section = self._format_task_description_section()

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def revise_rubrics():
            params = self._base_template_params(
                sample_content=sample_content,
                num_outputs=num_outputs,
                task_description_section=task_description_section,
                target_main_dim=self._extract_target_main_dim(data),
            )
            params.update({"rubrics": rubrics_text, "feedback": feedback})
            params.pop("language", None)
            messages = self.revision_template.format(language=self.language, **params)
            messages = attach_media_to_messages(
                messages,
                media_paths,
                self.task_type,
                has_base=has_base_media(data, self.task_type),
                reference_images=reference_images or None,
            )
            chat_response = await self.model.achat(
                messages=messages,
                structured_model=RubricGenerationOutput,
            )

            if not chat_response.parsed or "rubrics" not in chat_response.parsed:
                raise ValueError(f"Rubric revision returned invalid JSON: {chat_response.parsed}")

            revised_rubrics = chat_response.parsed["rubrics"]
            if not revised_rubrics:
                raise ValueError("No revised rubrics generated")
            return revised_rubrics

        try:
            revised_rubrics = await revise_rubrics()
            logger.debug(f"Revised {len(revised_rubrics)} rubrics")
            return revised_rubrics
        except Exception as e:
            logger.error(f"Failed to revise rubrics after {self.max_retries} attempts: {e}")
            return []

    def generate_feedback(self, data: dict, evaluation_result: Dict[str, Any]) -> str:
        if self.grader_mode == GraderMode.POINTWISE:
            return self._generate_pointwise_feedback(data, evaluation_result)
        return self._generate_listwise_feedback(data, evaluation_result)

    async def _evaluate_pointwise(self, data: dict, rubrics: List[str]) -> Dict[str, Any]:
        rubrics_text = self._format_rubrics_text(rubrics)
        sample_content = self._format_data_context(data)
        media_paths = extract_media_paths(data, self.task_type)
        reference_images = extract_reference_images(data) if is_video_task(self.task_type) else {}
        num_outputs = max(1, count_outputs(data, self.task_type))
        task_description_section = self._format_task_description_section()

        try:
            params = self._base_template_params(
                sample_content=sample_content,
                num_outputs=num_outputs,
                task_description_section=task_description_section,
                target_main_dim=self._extract_target_main_dim(data),
            )
            params["rubrics"] = rubrics_text
            params.pop("language", None)
            messages = self.evaluation_template.format(language=self.language, **params)
            messages = attach_media_to_messages(
                messages,
                media_paths,
                self.task_type,
                has_base=has_base_media(data, self.task_type),
                reference_images=reference_images or None,
            )
            response_obj = await self.model.achat(
                messages=messages,
                structured_model=GraderScoreCallback,
            )

            score: float = float(self.min_score)
            score_reason: str = ""
            score_metadata: Dict[str, Any] = {}
            if response_obj.parsed and "score" in response_obj.parsed:
                try:
                    score = float(response_obj.parsed["score"])
                except (TypeError, ValueError):
                    score = float(self.min_score)
                score = max(float(self.min_score), min(float(self.max_score), score))
                score = round(score, 3)
                score_reason = str(response_obj.parsed.get("reason", "") or "")
                raw_metadata = response_obj.parsed.get("metadata")
                if isinstance(raw_metadata, Mapping):
                    score_metadata = dict(raw_metadata)

            return {"scores": [score], "reason": score_reason, "metadata": score_metadata}

        except Exception as e:
            logger.error(f"Pointwise evaluation failed: {e}")
            return {"scores": [float(self.min_score)], "reason": f"<eval-error: {e}>", "metadata": {}}

    async def _evaluate_listwise(self, data: dict, rubrics: List[str]) -> Dict[str, Any]:
        rubrics_text = self._format_rubrics_text(rubrics)
        sample_content = self._format_data_context(data)
        media_paths = extract_media_paths(data, self.task_type)
        reference_images = extract_reference_images(data) if is_video_task(self.task_type) else {}
        num_outputs = count_outputs(data, self.task_type)
        task_description_section = self._format_task_description_section()

        try:
            params = self._base_template_params(
                sample_content=sample_content,
                num_outputs=num_outputs,
                task_description_section=task_description_section,
                target_main_dim=self._extract_target_main_dim(data),
            )
            params["rubrics"] = rubrics_text
            params.pop("language", None)
            messages = self.evaluation_template.format(language=self.language, **params)
            messages = attach_media_to_messages(
                messages,
                media_paths,
                self.task_type,
                has_base=has_base_media(data, self.task_type),
                reference_images=reference_images or None,
            )
            response_obj = await self.model.achat(
                messages=messages,
                structured_model=GraderRankCallback,
            )

            if response_obj.parsed and "rank" in response_obj.parsed:
                rank_values = [int(v) for v in response_obj.parsed["rank"]]
                rank_reason = str(response_obj.parsed.get("reason", "") or "")
                raw_metadata = response_obj.parsed.get("metadata")
                rank_metadata: Dict[str, Any] = (
                    dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
                )
                if len(rank_values) == num_outputs:
                    if len(set(rank_values)) != len(rank_values):
                        logger.warning(f"Duplicate rank values detected: {rank_values}")
                    return {
                        "rank_values": rank_values,
                        "reason": rank_reason,
                        "metadata": rank_metadata,
                    }
                logger.warning(
                    "Invalid rank length: got {} ranks {} for {} outputs",
                    len(rank_values),
                    rank_values,
                    num_outputs,
                )

            return {"rank_values": [], "reason": "", "metadata": {}}

        except Exception as e:
            logger.error(f"Listwise evaluation failed: {e}")
            return {"rank_values": [], "reason": f"<eval-error: {e}>", "metadata": {}}

    def _validate_pointwise(self, data: dict, evaluation_result: Dict[str, Any]) -> bool:
        """Pass when |pred - label| <= validation_tolerance (default 0.5).

        We no longer require an exact integer match because labels are now
        continuous floats (e.g. winner=4.84, loser=2.80, mid=3.48). Exact
        equality is essentially unreachable; tolerance-based validation gives
        the iterative loop a meaningful pass criterion while still penalising
        large miscalibrations.
        """
        scores = evaluation_result.get("scores", [])
        if not scores or len(scores) != 1:
            return False

        expected_score = data.get("label_score")
        if expected_score is None:
            return False
        try:
            return abs(float(scores[0]) - float(expected_score)) <= float(self.validation_tolerance)
        except (TypeError, ValueError):
            return False

    def _validate_listwise(self, data: dict, evaluation_result: Dict[str, Any]) -> bool:
        """Strict equality on the rank vector (no relative-order leniency).

        Previously we compared `_get_relative_order` of predicted vs expected
        ranks, which collapsed cases like predicted=[2,1] vs expected=[1,2]
        into the same "order" because both yield [1,0]. For 2-way comparisons
        that effectively made every prediction look correct as long as the
        predicted ranks weren't identical, masking real failures during
        rubric revision.
        """
        rank_values = evaluation_result.get("rank_values", [])
        expected_ranks = data.get("label_rank", [])

        if not rank_values or not expected_ranks:
            return False
        if len(rank_values) != len(expected_ranks):
            return False

        try:
            pred = [int(v) for v in rank_values]
            gold = [int(v) for v in expected_ranks]
        except (TypeError, ValueError):
            return False

        logger.debug(f"Expected ranks: {gold}")
        logger.debug(f"Predicted ranks: {pred}")
        return pred == gold

    @staticmethod
    def _get_relative_order(values: List[float]) -> List[int]:
        indexed_values = list(enumerate(values))
        indexed_values.sort(key=lambda x: x[1])
        return [idx for idx, _ in indexed_values]

    def _generate_pointwise_feedback(self, data: dict, evaluation_result: Dict[str, Any]) -> str:
        expected_score = data.get("label_score")
        actual_scores = evaluation_result.get("scores", [])
        actual_score = actual_scores[0] if actual_scores else None
        prev_reason = evaluation_result.get("reason", "")
        prev_metadata = evaluation_result.get("metadata") or {}

        lines = [
            f"Expected score: {expected_score}",
            f"Actual score:   {actual_score}",
        ]
        # Direction + magnitude diagnostics make it easier for the revise step
        # to know whether the previous rubrics were systematically too lenient
        # or too harsh.
        try:
            diff = float(actual_score) - float(expected_score)
            direction = "OVERESTIMATED" if diff > 0 else ("UNDERESTIMATED" if diff < 0 else "matched")
            lines.append(f"Error magnitude: {abs(diff):.2f}  (the previous rubrics {direction} the sample)")
            lines.append(
                f"Validation tolerance: {self.validation_tolerance:.2f}; this gap is"
                f"{' WITHIN' if abs(diff) <= self.validation_tolerance else ' BEYOND'} tolerance."
            )
        except (TypeError, ValueError):
            pass

        # Render per-rubric breakdown emitted by the previous evaluation. This
        # is the key signal the revise step needs to retarget the failing
        # rubric instead of rewriting everything.
        per_rubric = prev_metadata.get("per_rubric") if isinstance(prev_metadata, Mapping) else None
        if isinstance(per_rubric, list) and per_rubric:
            lines.append("Previous round per-rubric breakdown:")
            for entry in per_rubric:
                if not isinstance(entry, Mapping):
                    continue
                idx = entry.get("rubric_index", "?")
                path = entry.get("rubric_path", "untagged")
                evidence = str(entry.get("evidence", "") or "")
                if len(evidence) > 80:
                    evidence = evidence[:80] + "..."
                gated = entry.get("gated", False)
                verdict = entry.get("verdict", "-")
                weight = entry.get("weight", "-")
                gate_tag = " [GATED]" if gated else ""
                lines.append(
                    f"  - r{idx}{gate_tag} path={path} weight={weight} verdict={verdict} evidence={evidence}"
                )
        weighted_sum = prev_metadata.get("weighted_sum_calc") if isinstance(prev_metadata, Mapping) else None
        if weighted_sum:
            lines.append(f"Previous weighted-sum calculation: {weighted_sum}")
        gating = prev_metadata.get("gating_decisions") if isinstance(prev_metadata, Mapping) else None
        if gating and gating != "none":
            lines.append(f"Previous gating decisions: {gating}")

        if prev_reason:
            # Cap to 800 chars so the feedback block doesn't explode.
            snippet = prev_reason if len(prev_reason) <= 800 else prev_reason[:800] + " ...[truncated]"
            lines.append("Previous judge reasoning (verbatim):")
            lines.append(snippet)

        lines.append(
            "Revision targets:\n"
            "  - Identify which rubric line(s) failed to discriminate this sample's quality level.\n"
            "  - Rebalance weights: down-weight rubrics whose verdicts misled the aggregate.\n"
            "  - Add discriminative anchors so the same evidence cannot be re-scored at the wrong level.\n"
            "  - Tighten Context-Aware Gating so irrelevant rubrics stop pulling the score toward the mean."
        )
        return "\n".join(lines)

    def _generate_listwise_feedback(self, data: dict, evaluation_result: Dict[str, Any]) -> str:
        expected_ranks = data.get("label_rank", [])
        actual_ranks = evaluation_result.get("rank_values", [])
        prev_reason = evaluation_result.get("reason", "")
        prev_metadata = evaluation_result.get("metadata") or {}

        lines = [
            f"Expected ranks: {expected_ranks}",
            f"Actual ranks:   {actual_ranks}",
        ]
        # Pinpoint the position of the rank inversion(s) so the revise step
        # can re-target the failing rubric instead of rewriting everything.
        try:
            mismatches = [
                i + 1 for i, (a, b) in enumerate(zip(actual_ranks, expected_ranks)) if int(a) != int(b)
            ]
            if mismatches:
                lines.append(f"Inverted positions (1-indexed): {mismatches}")
        except (TypeError, ValueError):
            pass

        # Render per-rubric breakdown emitted by the previous listwise eval.
        per_rubric = prev_metadata.get("per_rubric") if isinstance(prev_metadata, Mapping) else None
        if isinstance(per_rubric, list) and per_rubric:
            lines.append("Previous round per-rubric breakdown:")
            for entry in per_rubric:
                if not isinstance(entry, Mapping):
                    continue
                idx = entry.get("rubric_index", "?")
                path = entry.get("rubric_path", "untagged")
                gated = entry.get("gated", False)
                rubric_rank = entry.get("rubric_rank", "-")
                weight = entry.get("weight", "-")
                gate_tag = " [GATED]" if gated else ""
                lines.append(
                    f"  - r{idx}{gate_tag} path={path} weight={weight} rubric_rank={rubric_rank}"
                )
        aggregation = prev_metadata.get("aggregation") if isinstance(prev_metadata, Mapping) else None
        if aggregation:
            lines.append(f"Previous aggregation: {aggregation}")

        if prev_reason:
            snippet = prev_reason if len(prev_reason) <= 800 else prev_reason[:800] + " ...[truncated]"
            lines.append("Previous judge reasoning (verbatim):")
            lines.append(snippet)

        lines.append(
            "Revision targets:\n"
            "  - Identify which rubric line(s) failed to separate the candidate videos at the inverted positions.\n"
            "  - Add evidence anchors that force a discriminative judgement at those positions.\n"
            "  - Reduce position bias by phrasing rubrics symmetrically over the candidates."
        )
        return "\n".join(lines)

    @staticmethod
    def _extract_target_main_dim(data: dict) -> str | None:
        """Pull an explicit EvalVerse main dim hint from the sample.

        Only an explicit ``target_main_dim`` field (top-level or nested in
        ``metadata``) triggers the EvalVerse-aware prompt block. The legacy
        ``dimension`` field is intentionally NOT used here, so that experiments
        such as v_arr_mvp (which set ``dimension`` for bookkeeping but want the
        "let rubrics emerge from data" prompt path) are not affected.
        """
        if not isinstance(data, Mapping):
            return None
        explicit = data.get("target_main_dim")
        if explicit:
            return str(explicit)
        meta = data.get("metadata")
        if isinstance(meta, Mapping):
            nested = meta.get("target_main_dim")
            if nested:
                return str(nested)
        return None

    def _format_data_context(self, data: dict) -> str:
        query = data.get("query") or data.get("prompt") or data.get("instruction") or ""
        media_paths = extract_media_paths(data, self.task_type)
        num_outputs = count_outputs(data, self.task_type)
        ranking_mode = "Pairwise" if num_outputs == 2 else "Listwise"
        lines: list[str] = []

        evalverse_task = (
            (data.get("metadata") or {}).get("evalverse_task")
            if isinstance(data.get("metadata"), Mapping)
            else None
        )

        if is_video_edit_task(self.task_type):
            lines.append(f"Video editing instruction: {query}")
            if has_base_media(data, self.task_type):
                lines.append("Video BASE is provided as the original/reference clip.")
                lines.append(f"Number of edited candidate videos: {num_outputs}")
            elif media_paths:
                lines.append("Only edited video(s) are provided; no Video BASE is available.")
                lines.append(f"Number of edited candidate videos: {num_outputs}")
        elif is_video_task(self.task_type):
            task_label = "EvalVerse task"
            if normalize_task_type(self.task_type) == TASK_R2V:
                lines.append(f"{task_label}: Reference-to-Video (R2V)")
            elif normalize_task_type(self.task_type) == TASK_T2V_MULTISHOT:
                lines.append(f"{task_label}: Text-to-Multi-Shot-Video")
            elif normalize_task_type(self.task_type) == TASK_T2V_SOUND:
                lines.append(f"{task_label}: Text-to-Video-with-Sound")
            else:
                lines.append(f"{task_label}: Text-to-Video")
            if evalverse_task and evalverse_task != normalize_task_type(self.task_type):
                lines.append(f"EvalVerse sub-task: {evalverse_task}")
            lines.append(f"Prompt: {query}")
            if has_base_media(data, self.task_type):
                lines.append("Ground-Truth (reference) video is provided alongside the generated candidate(s).")
            references = extract_reference_images(data)
            if references:
                ref_labels = ", ".join(sorted(references.keys()))
                lines.append(f"Reference images provided: {ref_labels}")
            audio = data.get("audio") or (data.get("metadata") or {}).get("references", {}).get("audio") if isinstance(data.get("metadata"), Mapping) else data.get("audio")
            if audio:
                lines.append(
                    "Audio track present. The VLM cannot perceive audio directly; "
                    "use sync cues (lip motion, sound-source action, scene-music coupling) for Sound-Design rubrics.",
                )
            shot_sequence = (
                (data.get("metadata") or {}).get("references", {}).get("shot_sequence")
                if isinstance(data.get("metadata"), Mapping)
                else data.get("shot_sequence")
            )
            if shot_sequence:
                lines.append(f"Multi-shot sequence metadata: {shot_sequence}")
            if media_paths:
                visible_outputs = num_outputs
                lines.append(f"Number of generated candidate videos: {visible_outputs}")
        elif is_image_edit_task(self.task_type):
            lines.append(f"Editing instruction: {query}")
            if has_base_media(data, self.task_type):
                lines.append("Image BASE is provided as the original/reference image.")
                lines.append(f"Number of edited candidate images: {num_outputs}")
            elif media_paths:
                lines.append("Only edited image(s) are provided; no Image BASE is available.")
                lines.append(f"Number of edited candidate images: {num_outputs}")
        else:
            lines.append(f"Caption: {query}")
            if media_paths:
                lines.append(f"Number of generated candidate images: {num_outputs}")

        if self.grader_mode == GraderMode.LISTWISE:
            lines.append(f"Evaluation mode for this sample: {ranking_mode}")

        if "label_score" in data:
            lines.append(f"Expected score: {data.get('label_score')}")
        if "label_rank" in data:
            lines.append(f"Expected rank values: {data.get('label_rank')}")
        if data.get("dimension") and is_video_task(self.task_type):
            lines.append(f"EvalVerse target dimension: {data.get('dimension')}")
        cot = (
            data.get("cot_rationale")
            or data.get("rationale")
            or (data.get("metadata") or {}).get("cot_rationale")
            if isinstance(data.get("metadata"), Mapping)
            else data.get("cot_rationale")
        )
        if cot:
            lines.append(f"Expert CoT rationale (for rubric induction only): {cot}")
        if data.get("human_dims") or (
            isinstance(data.get("metadata"), Mapping) and data["metadata"].get("human_dims")
        ):
            dims = data.get("human_dims") or data["metadata"]["human_dims"]
            lines.append(f"Per-dimension expert scores: {dims}")
        if data.get("task_type") and is_image_edit_task(self.task_type):
            lines.append(f"Edit operation tags: {data.get('task_type')}")

        return "\n".join(lines)

    def _base_template_params(
        self,
        sample_content: str,
        num_outputs: int,
        task_description_section: str,
        target_main_dim: str | None = None,
    ) -> Dict[str, Any]:
        ranking_mode = "Pairwise" if num_outputs == 2 else "Listwise"
        aspects_line = (
            _EVALVERSE_ASPECTS_LINE_ZH
            if self.language == LanguageEnum.ZH
            else _EVALVERSE_ASPECTS_LINE_EN
        )
        # When target_main_dim is provided AND it is a canonical EvalVerse main
        # dim, inject a per-dim block; otherwise the block is empty and the
        # prompt falls back to the task_description-driven path (this is what
        # v_arr_mvp relies on to keep its "reject fixed taxonomy" setting).
        evalverse_dim_block = _build_evalverse_dim_block(self.language, target_main_dim)
        return {
            "language": self.language,
            "sample_content": sample_content,
            "generate_number": self.generate_number,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "num_outputs": num_outputs,
            "ranking_mode": ranking_mode,
            "task_description_section": task_description_section,
            "aspects": aspects_line,
            "evalverse_dim_block": evalverse_dim_block,
        }

    @staticmethod
    def _format_rubrics_text(rubrics: List[str]) -> str:
        return "\n".join([f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics)])

    def _format_task_description_section(self) -> str:
        if not self.task_description:
            return ""
        if self.language == LanguageEnum.ZH:
            return f"\n## 任务场景描述\n{self.task_description}\n"
        return f"\n## Task Description\n{self.task_description}\n"
