import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.asyncio import tqdm as async_tqdm

from rubric_pipeline.generator.iterative_rubric.generator import (
    IterativeListwiseRubricsGeneratorConfig,
    IterativePointwiseRubricsGeneratorConfig,
    IterativeRubricsGenerator,
)
from rubric_pipeline.generator.iterative_rubric.query_rubric_generator import (
    get_evaluation_template,
)
from rubric_pipeline.graders.llm_grader import LLMGrader
from rubric_pipeline.graders.schema import GraderMode, GraderRank, GraderScore
from rubric_pipeline.models import OpenAIChatModel
from rubric_pipeline.models.schema.prompt_template import LanguageEnum
from rubric_pipeline.utils.media import (
    TASK_T2I,
    attach_media_to_messages,
    count_outputs,
    extract_media_paths,
    has_base_media,
    is_image_edit_task,
    is_video_edit_task,
    is_video_task,
    normalize_task_type,
)


NONE_VALUES = {None, "", "none", "null", "None"}

def _compute_pointwise_metrics(
    preds: list[float],
    labels: list[float],
) -> tuple[float | None, float | None, float | None]:
    """Return (PLCC, SRCC, MAE). Any of them may be None if undefined.

    Pearson/Spearman are undefined when n < 2 or when one side has zero
    variance (constant predictions or labels). MAE is always defined for n>=1.
    Implemented without scipy to avoid an extra hard dependency.
    """
    if not preds or not labels or len(preds) != len(labels):
        return None, None, None
    n = len(preds)
    mae = sum(abs(p - l) for p, l in zip(preds, labels)) / n

    plcc: float | None = None
    if n >= 2:
        mean_p = sum(preds) / n
        mean_l = sum(labels) / n
        num = sum((p - mean_p) * (l - mean_l) for p, l in zip(preds, labels))
        den_p = sum((p - mean_p) ** 2 for p in preds) ** 0.5
        den_l = sum((l - mean_l) ** 2 for l in labels) ** 0.5
        if den_p > 1e-12 and den_l > 1e-12:
            plcc = num / (den_p * den_l)

    srcc: float | None = None
    if n >= 2:
        rp = _ranks_with_ties(preds)
        rl = _ranks_with_ties(labels)
        mean_rp = sum(rp) / n
        mean_rl = sum(rl) / n
        num = sum((a - mean_rp) * (b - mean_rl) for a, b in zip(rp, rl))
        den_rp = sum((a - mean_rp) ** 2 for a in rp) ** 0.5
        den_rl = sum((b - mean_rl) ** 2 for b in rl) ** 0.5
        if den_rp > 1e-12 and den_rl > 1e-12:
            srcc = num / (den_rp * den_rl)

    return plcc, srcc, mae

def _ranks_with_ties(xs: list[float]) -> list[float]:
    """Average-rank with ties (Spearman-friendly), 1-indexed."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _load_structured_file(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        return json.load(f)


class Judger:
    """Auto-Rubric judger for image and video generation tasks."""

    def __init__(
        self,
        config_path: str,
        seed_dataset: str | None = None,
        rubrics_file: str | None = None,
        base_url: str | None = None,
    ):
        self.config_path = Path(config_path)
        self.config = _load_structured_file(self.config_path)
        self.config_dir = self.config_path.parent

        self.task_type = self._resolve_task_type()
        self.eval_mode = str(
            self.config.get("eval_mode")
            or self.config.get("grader_mode")
            or self.config.get("mode")
            or "pairwise",
        ).lower()
        self.grader_mode = GraderMode.POINTWISE if self.eval_mode == "pointwise" else GraderMode.LISTWISE
        self.language = LanguageEnum(self.config.get("language", "en"))
        self.min_score = int(self.config.get("min_score", 1))
        self.max_score = int(self.config.get("max_score", 5))

        self.seed_dataset = self._first_non_empty(seed_dataset, self.config.get("seed_dataset"))
        self.rubrics_file = self._first_non_empty(rubrics_file, self.config.get("rubrics_file"))
        self.base_url = self._expand_env(self._first_non_empty(base_url, self.config.get("base_url")))
        self.judger: LLMGrader | None = None
        # Populated by `initialize()`; useful for external scripts that want to
        # persist the verified rubric text (e.g. per-dimension rubric generation).
        self.rubrics: str | None = None

    @staticmethod
    def _detect_eval_mode_from_dataset(path: str | Path) -> str | None:
        """Sniff a dataset file to decide whether it's pointwise or listwise.

        Returns ``"pointwise"`` if the first record has ``label_score`` (or a
        single-string ``response``); ``"listwise"`` if it has ``label_rank``
        (or a list ``response`` of length>=2). Returns ``None`` if the file
        cannot be loaded or is empty.
        """
        try:
            data = _load_structured_file(path)
        except Exception:
            return None
        if not isinstance(data, list) or not data:
            return None
        first = data[0]
        if not isinstance(first, dict):
            return None
        if "label_score" in first:
            return "pointwise"
        if "label_rank" in first:
            return "listwise"
        resp = first.get("response")
        if isinstance(resp, list) and len(resp) >= 2:
            return "listwise"
        if isinstance(resp, str):
            return "pointwise"
        return None

    def _apply_eval_mode(self, mode: str) -> None:
        """Update ``self.eval_mode`` and ``self.grader_mode`` in-place."""
        m = (mode or "").strip().lower()
        if m == "pairwise":
            m = "listwise"
        if m not in {"pointwise", "listwise"}:
            return
        self.eval_mode = m
        self.grader_mode = (
            GraderMode.POINTWISE if m == "pointwise" else GraderMode.LISTWISE
        )

    @staticmethod
    def _first_non_empty(*values: Any) -> Any:
        for value in values:
            if value not in NONE_VALUES:
                return value
        return None

    @staticmethod
    def _expand_env(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value

    def _resolve_path(self, value: str | Path | None) -> Path | None:
        if value in NONE_VALUES:
            return None
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        project_relative = Path.cwd() / path
        if project_relative.exists():
            return project_relative
        return self.config_dir / path

    def _resolve_task_type(self) -> str:
        configured = self.config.get("task_type") or self.config.get("task")
        if configured not in NONE_VALUES:
            return normalize_task_type(str(configured))
        stem = self.config_path.stem.lower()
        if "video_sound" in stem or "with_sound" in stem or "t2v_sound" in stem:
            return normalize_task_type("t2v_sound")
        if "multishot" in stem or "multi_shot" in stem or "t2v_multishot" in stem:
            return normalize_task_type("t2v_multishot")
        if "r2v" in stem or "reference_to_video" in stem:
            return normalize_task_type("r2v")
        if "video_edit" in stem or ("video" in stem and "edit" in stem):
            return normalize_task_type("video_edit")
        if "t2v" in stem or "video" in stem:
            return normalize_task_type("t2v")
        if "edit" in stem:
            return normalize_task_type("image_edit")
        return normalize_task_type(TASK_T2I)

    def _task_description_section(self) -> str:
        task_description = self.config.get("task_description", "")
        if not task_description:
            return ""
        if self.language == LanguageEnum.ZH:
            return f"\n## 任务场景描述\n{task_description}\n"
        return f"\n## Task Description\n{task_description}\n"

    def _build_model(self) -> OpenAIChatModel:
        api_key = self._expand_env(self.config.get("api_key")) or os.getenv("OPENAI_API_KEY")
        return OpenAIChatModel(
            model=self._expand_env(self.config["model_name"]),
            api_key=api_key,
            base_url=self.base_url,
            temperature=self.config.get("temperature", 0),
        )

    async def initialize(self):
        """Build the final grader from a rubrics file or generate rubrics from data."""
        model = self._build_model()
        rubrics = self.config.get("rubrics")

        rubrics_path = self._resolve_path(self.rubrics_file)
        if rubrics_path:
            rubrics = rubrics_path.read_text(encoding="utf-8")
        elif not rubrics:
            seed_path = self._resolve_path(self.seed_dataset)
            if not seed_path:
                raise ValueError("Config must provide `rubrics`, `rubrics_file`, or `seed_dataset`.")
            # Auto-detect rubric-generation mode from the seed dataset itself,
            # overriding any (possibly stale) ``eval_mode`` written in the yaml.
            # This prevents the failure where a pointwise seed (label_score)
            # is fed into a listwise validator (label_rank) and every rubric
            # is rejected with valid=False.
            seed_mode = self._detect_eval_mode_from_dataset(seed_path)
            if seed_mode and seed_mode != self.eval_mode:
                print(
                    f"[judger] auto-detected seed_dataset mode = {seed_mode}; "
                    f"overriding yaml eval_mode={self.eval_mode}"
                )
                self._apply_eval_mode(seed_mode)
            rubrics_dataset = _load_structured_file(seed_path)
            generator_config_kwargs = {
                "grader_name": self.config["grader_name"],
                "model": model,
                "language": self.language,
                "query_specific_generate_number": int(self.config.get("query_specific_generate_number", 1)),
                "enable_categorization": bool(self.config.get("enable_categorization", False)),
                "categories_number": int(self.config.get("categories_number", 5)),
                "task_description": self.config.get("task_description"),
                "task_type": self.task_type,
                "max_retries": int(self.config.get("max_retries", 5)),
                "max_epochs": int(self.config.get("max_epochs", 5)),
            }
            if self.grader_mode == GraderMode.POINTWISE:
                generator_config = IterativePointwiseRubricsGeneratorConfig(
                    min_score=self.min_score,
                    max_score=self.max_score,
                    validation_tolerance=float(self.config.get("validation_tolerance", 0.5)),
                    **generator_config_kwargs,
                )
            else:
                generator_config = IterativeListwiseRubricsGeneratorConfig(**generator_config_kwargs)
            generator = IterativeRubricsGenerator(generator_config)
            generated_grader = await generator.generate(rubrics_dataset)
            rubrics = generated_grader.kwargs.get("rubrics", "")
            print("Rubrics generated:")
            print(rubrics)

        self.rubrics = rubrics if isinstance(rubrics, str) else None

        grader_kwargs: dict[str, Any] = {
            "task_type": self.task_type,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "task_description_section": self._task_description_section(),
        }
        if self.config.get("video_input_strategy"):
            grader_kwargs["video_input_strategy"] = self.config["video_input_strategy"]

        self.judger = LLMGrader(
            name=self.config["grader_name"],
            model=model,
            mode=self.grader_mode,
            rubrics=rubrics,
            language=self.language,
            template=get_evaluation_template(self.grader_mode, self.task_type),
            **grader_kwargs,
        )
        return self

    def _format_sample_content(self, item: dict) -> str:
        query = item.get("query") or item.get("prompt") or item.get("instruction") or ""
        num_outputs = count_outputs(item, self.task_type)
        lines: list[str] = []

        if is_video_edit_task(self.task_type):
            lines.append(f"Video editing instruction: {query}")
            lines.append(f"Number of edited candidate videos: {num_outputs}")
            if has_base_media(item, self.task_type):
                lines.append("Video BASE is provided as the original/reference clip.")
        elif is_video_task(self.task_type):
            lines.append(f"Caption: {query}")
            lines.append(f"Number of generated candidate videos: {num_outputs}")
        elif is_image_edit_task(self.task_type):
            lines.append(f"Editing instruction: {query}")
            lines.append(f"Number of edited candidate images: {num_outputs}")
            if has_base_media(item, self.task_type):
                lines.append("Image BASE is provided as the original/reference image.")
        else:
            lines.append(f"Caption: {query}")
            lines.append(f"Number of generated candidate images: {num_outputs}")

        return "\n".join(lines)

    async def _evaluate_item(self, item: dict) -> GraderScore | GraderRank:
        if self.judger is None:
            raise RuntimeError("Judger not initialized. Call `await judger.initialize()` first.")

        media_paths = extract_media_paths(item, self.task_type)
        return await self.judger.aevaluate(
            media_paths=media_paths,
            has_base_media=has_base_media(item, self.task_type),
            sample_content=self._format_sample_content(item),
        )

    async def _evaluate_single(self, item: dict, semaphore: asyncio.Semaphore):
        async with semaphore:
            try:
                return await self._evaluate_item(item)
            except Exception as e:
                print(f"Error evaluating item: {e}")
                return None

    @staticmethod
    def _rank_to_rewards(rank: list[int]) -> list[float]:
        if not rank:
            return []
        if len(rank) == 1:
            return [1.0]
        if len(rank) == 2:
            return [1.0 if value == 1 else -0.1 for value in rank]

        worst = max(rank)
        if worst <= 1:
            return [1.0 for _ in rank]
        return [1.0 - ((value - 1) / (worst - 1)) * 1.1 for value in rank]

    async def get_reward_single(self, item: dict, device) -> torch.Tensor:
        result = await self._evaluate_item(item)
        if isinstance(result, GraderScore):
            return torch.tensor([float(result.score)], device=device)
        return torch.tensor(self._rank_to_rewards(result.rank), device=device)

    async def get_reward(self, data: list[dict], concurrency_limit: int = 50) -> list[Any]:
        if self.judger is None:
            raise RuntimeError("Judger not initialized. Call `await judger.initialize()` first.")

        semaphore = asyncio.Semaphore(concurrency_limit)
        tasks = [self._evaluate_single(item, semaphore) for item in data]
        results = await async_tqdm.gather(*tasks, desc="Getting rewards")
        outputs = []
        for result in results:
            if isinstance(result, GraderScore):
                outputs.append(result.score)
            elif isinstance(result, GraderRank):
                outputs.append(result.rank)
            else:
                outputs.append(None)
        return outputs

    async def evaluate(self, dataset_path: str, concurrency_limit: int = 50) -> dict:
        if self.judger is None:
            raise RuntimeError("Judger not initialized. Call `await judger.initialize()` first.")

        dataset = _load_structured_file(dataset_path)

        # If the test dataset has a different shape than the rubric-generation
        # phase (e.g. rubric was generated from a pairwise seed but applied to
        # a pointwise shared test set in RewardVerse MVP head-to-head), rebuild the
        # grader on-the-fly with the test-set mode. The rubric text itself is
        # mode-agnostic and is reused verbatim.
        test_mode = self._detect_eval_mode_from_dataset(dataset_path)
        if test_mode and test_mode != self.eval_mode:
            print(
                f"[judger] auto-detected test_dataset mode = {test_mode}; "
                f"rebuilding grader from {self.eval_mode} -> {test_mode}"
            )
            self._apply_eval_mode(test_mode)
            rubrics_text = self.rubrics or (
                self.judger.rubrics if self.judger is not None else None
            )
            grader_kwargs: dict[str, Any] = {
                "task_type": self.task_type,
                "min_score": self.min_score,
                "max_score": self.max_score,
                "task_description_section": self._task_description_section(),
            }
            if self.config.get("video_input_strategy"):
                grader_kwargs["video_input_strategy"] = self.config["video_input_strategy"]
            self.judger = LLMGrader(
                name=self.config["grader_name"],
                model=self._build_model(),
                mode=self.grader_mode,
                rubrics=rubrics_text,
                language=self.language,
                template=get_evaluation_template(self.grader_mode, self.task_type),
                **grader_kwargs,
            )

        predictions = await self.get_reward(dataset, concurrency_limit=concurrency_limit)

        total = len(dataset)
        if self.grader_mode == GraderMode.POINTWISE:
            preds: list[float] = []
            labels: list[float] = []
            for item, prediction in zip(dataset, predictions):
                if prediction is None:
                    continue
                expected = item.get("label_score")
                if expected is None:
                    continue
                try:
                    preds.append(float(prediction))
                    labels.append(float(expected))
                except (TypeError, ValueError):
                    continue

            plcc, srcc, mae = _compute_pointwise_metrics(preds, labels)
            print("-" * 30)
            print(f"Task:        {self.task_type}")
            print(f"Mode:        {self.eval_mode}")
            print(f"Total Tests: {total}")
            print(f"Parseable:   {len(preds)}")
            plcc_str = f"{plcc:.4f}" if plcc is not None else "n/a"
            srcc_str = f"{srcc:.4f}" if srcc is not None else "n/a"
            mae_str  = f"{mae:.4f}"  if mae  is not None else "n/a"
            print(f"PLCC:        {plcc_str}")
            print(f"SRCC:        {srcc_str}")
            print(f"MAE:         {mae_str}")
            print("-" * 30)
            return {
                "mode": "pointwise",
                "total": total,
                "parseable": len(preds),
                "plcc": plcc,
                "srcc": srcc,
                "mae": mae,
            }

        # Listwise / pairwise: use STRICT rank equality (no relative-order leniency).
        correct = 0
        parseable = 0
        for item, prediction in zip(dataset, predictions):
            if prediction is None:
                continue
            parseable += 1
            if list(prediction) == list(item.get("label_rank") or []):
                correct += 1
        accuracy = correct / parseable if parseable > 0 else 0.0
        print("-" * 30)
        print(f"Task:        {self.task_type}")
        print(f"Mode:        {self.eval_mode}")
        print(f"Total Tests: {total}")
        print(f"Parseable:   {parseable}")
        print(f"Correct:     {correct}")
        print(f"Accuracy:    {accuracy:.2%}")
        print("-" * 30)
        return {
            "mode": "listwise",
            "total": total,
            "parseable": parseable,
            "correct": correct,
            "accuracy": accuracy,
        }


async def main(args):
    judger = await Judger(
        config_path=args.config_path,
        seed_dataset=args.seed_dataset,
        rubrics_file=args.rubrics_file,
        base_url=args.base_url,
    ).initialize()

    # Persist the (possibly auto-generated) rubric so downstream RQ2/RQ3 can run
    # without the user having to copy-paste from stdout.
    if args.rubrics_out and judger.rubrics:
        out_path = Path(args.rubrics_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(judger.rubrics, encoding="utf-8")
        print(f"[judger] rubric saved to {out_path}")

    metrics = None
    if args.test_dataset:
        metrics = await judger.evaluate(
            dataset_path=args.test_dataset,
            concurrency_limit=args.concurrency_limit,
        )

    if args.metrics_out and metrics is not None:
        out = Path(args.metrics_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[judger] metrics saved to {out}")
    elif args.metrics_out and metrics is None:
        print("[judger] --metrics_out specified but --test_dataset was omitted; "
              "skipping metrics dump.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config yaml/json")
    parser.add_argument("--rubrics_file", type=str, default=None, help="Optional rubrics text file")
    parser.add_argument("--seed_dataset", type=str, default=None, help="Optional seed dataset for Auto-Rubric generation")
    parser.add_argument("--test_dataset", type=str, default=None,
                        help="Optional dataset path for evaluation. When omitted, the judger "
                             "only generates the rubric (Auto-Rubric RQ1) and exits — useful "
                             "for the per-dimension v_arr_mvp pipelines that produce one "
                             "rubric file per EvalVerse second-level dimension.")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--concurrency_limit", type=int, default=50, help="Concurrent evaluation calls")
    parser.add_argument("--rubrics_out", type=str, default=None, help="Optional path to write the (auto-generated) rubric text")
    parser.add_argument("--metrics_out", type=str, default=None, help="Optional path to write evaluation metrics as JSON (PLCC/SRCC/MAE for pointwise, accuracy for listwise)")
    asyncio.run(main(parser.parse_args()))
