"""
Prime-RL environment that rewards only the absence of repetition loops.

Reward is ``1 - block_coverage``: the share of the rollout taken up by its single
most-repeated 30-word block, subtracted from one. Nothing about answer quality is
scored, so this is a surgical fix for the SFT init's looping, not a general
training env — run it with a reference-KL anchor so the policy cannot pay for low
repetition with everything else. open-instruct's binary detector is still computed,
as a zero-weight metric and as an alternative ``reward_mode``.

Prompts default to Dolci's **math** split. Job 53239 generated k=8 on 100 dolci
math prompts at the training sampling config and measured, per passrate band:

    band   strict fire%        trainable groups% (k=8)   our acc   groups w/ varying
                               binary detector                     block_coverage
    p0      6.00 [3.0,  9.5]    28.0 [12, 44]             0.150      80.0
    hard    9.50 [6.0, 13.5]    52.0 [36, 68]             0.200     100.0
    mid     8.50 [6.0, 11.5]    56.0 [40, 72]             0.280      96.0
    easy    7.00 [4.0, 10.0]    44.0 [28, 60]             0.380      96.0

Three things follow. Dolci's recorded ``passrate`` does **not** predict where we loop
— every CI overlaps — so there is no band to select and the whole split is used.
But ``passrate == 0`` fires least and is 3k tokens shorter than the rest (the model
gives up on unsolvable rows rather than grinding), so the default ``min_passrate``
drops it. And the continuous reward nearly doubles the usable fraction of the batch:
the binary detector leaves 28-56% of groups with a constant reward and therefore no
gradient, where ``block_coverage`` varies inside 80-100% of the same groups.

Rates of 6-9.5% bracket AIME25 at the same sampling (7.08%) and are ~15x the live
training mix (0.5%), which is what makes this dataset usable at all.
"""

import logging

import verifiers as vf
from datasets import Dataset
from rubric_judge_env.rubric_judge_env import _build_example, _load_raw

from .openinstruct_repetition import detect_exact_block_repetition, find_ngram_repetitions

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_PATH = "/shared_silo/scratch/datasets/RL/Dolci-Think-RL-7B"

# Constraints that *order* the model to repeat itself. On dolci's ifeval split
# 4334 of the 25387 rows surviving the passrate filter (17.1%) carry one, and on
# those rows a compliant answer has a high block_coverage by construction -- this
# reward would pay the model to disobey. Dropped by default; inert on math, where
# no row matches.
_REPEAT_INSTRUCTION_IDS = (
    "copy:repeat_phrase",
    "copy:copying_multiple",
    "copy:copying_simple",
    "copy:copy",
    "combination:repeat_prompt",
)

# Neutral on purpose: the prompts are now math, and the env scores only repetition,
# so nothing here should push the policy toward a particular answer format.
_SYSTEM_PROMPT = "You are a helpful assistant."

_SYSTEM_PROMPT_FI = "Olet avulias assistentti."


_THINK_END = "</think>"

# Long enough that no healthy derivation repeats one verbatim. This is what keeps
# block coverage from being the length proxy the n-gram fraction is: over 240 AIME
# rollouts the n-gram fraction had corr(length) = +0.68 and a median of 0.46 -- every
# healthy output "repeats" trigrams, so minimising it just means writing less -- while
# block coverage is exactly 0 on half of job 53239's 800 dolci rollouts (median 0.0,
# p90 0.084 against an n-gram median of 0.445) and only lights up on real loops.
_BLOCK_WORDS = 30


def _field(message, name: str) -> str:
    value = message.get(name) if isinstance(message, dict) else getattr(message, name, None)
    return value if isinstance(value, str) else ""


def _completion_text(completion) -> str:
    """Full assistant text, reasoning block included.

    The loops we are targeting happen inside <think>, so unlike the judge env this
    must not strip reasoning before scoring.  With a vLLM reasoning parser active
    -- prime-rl defaults to reasoning_parser="auto" -- the thinking never reaches
    `content` at all, it arrives as `reasoning_content`.  Both fields are stitched
    back together with the closing tag reinstated, so _split_reasoning sees the
    same shape either way.
    """
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return str(completion)

    parts = []
    for message in completion:
        if _field(message, "role") != "assistant":
            continue
        reasoning, content = _field(message, "reasoning_content"), _field(message, "content")
        parts.append(reasoning + _THINK_END + content if reasoning else content)
    return "\n".join(parts)


def _split_reasoning(text: str) -> tuple[str, str]:
    """Split into (reasoning, answer) at the first closing think tag.

    Qwen-style completions open *inside* the reasoning block, so the leading
    <think> is optional.  No closing tag anywhere means the model never opened a
    thinking block -- that is untagged answer text, not an unterminated thought.
    A rollout that really did run out of budget mid-thought reaches us with a
    non-empty reasoning_content and empty content, which _completion_text turns
    into a closing tag with nothing after it.
    """
    before, sep, after = text.partition(_THINK_END)
    if not sep:
        return "", text
    return before.lstrip().removeprefix("<think>"), after


def _output_tokens(state: dict) -> float:
    return (state.get("usage") or {}).get("output_tokens", 0.0)


def block_coverage(text: str) -> tuple[float, int]:
    """Share of `text` covered by its most-repeated block, and that block's count.

    Coverage is the size of the *union* of the windows that block occupies, over the
    length of the text. Counting `occurrences * _BLOCK_WORDS` instead is wrong twice
    over: a loop whose period is shorter than the window recurs at nearly every offset
    inside it, so the product saturates at 1.0 no matter how little of the rollout the
    loop occupies, while two copies of a boilerplate block at opposite ends of a long
    healthy answer would, by their span, look like total repetition.

    A block seen once is not repetition, so coverage is 0 until some window recurs —
    without that guard a 40-word answer would score 0.75 for saying 30 distinct words.
    Coverage has no floor of its own: an empty rollout repeats nothing and would score
    perfectly, which is what `min_completion_tokens` and `require_reasoning_and_answer`
    in `RepetitionRubric` are there to prevent.
    """
    words = text.split()
    if len(words) < _BLOCK_WORDS:
        return 0.0, int(bool(words))

    positions: dict[str, list[int]] = {}
    for i in range(len(words) - _BLOCK_WORDS + 1):
        positions.setdefault(" ".join(words[i : i + _BLOCK_WORDS]), []).append(i)

    hits = max(positions.values(), key=len)
    if len(hits) < 2:
        return 0.0, len(hits)

    # hits is ascending by construction, so one sweep merges the overlaps.
    covered, previous_end = 0, 0
    for start in hits:
        end = start + _BLOCK_WORDS
        covered += end - max(start, previous_end)
        previous_end = end
    return covered / len(words), len(hits)


REWARD_MODES = ("blockcov", "detector")


class RepetitionRubric(vf.Rubric):
    """Low-repetition reward plus zero-weight diagnostics."""

    def __init__(
        self,
        min_repetitions: int,
        min_sentence_repetitions: int,
        min_consecutive_sentence_repetitions: int,
        min_completion_tokens: int,
        require_reasoning_and_answer: bool,
        ngram_size: int,
        reward_mode: str,
    ):
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        self._thresholds = dict(
            min_repetitions=min_repetitions,
            min_sentence_repetitions=min_sentence_repetitions,
            min_consecutive_sentence_repetitions=min_consecutive_sentence_repetitions,
        )
        self._min_completion_tokens = min_completion_tokens
        self._require_reasoning_and_answer = require_reasoning_and_answer
        self._ngram_size = ngram_size
        self._reward_mode = reward_mode

        self.add_reward_func(self.no_repetition)
        self.add_metric(self.block_coverage)
        self.add_metric(self.max_block_repeats)
        self.add_metric(self.repetition_detected)
        self.add_metric(self.repeated_ngram_fraction)
        self.add_metric(self.has_reasoning)
        self.add_metric(self.has_answer)
        self.add_metric(self.truncated)
        self.add_metric(self.output_tokens)

    def _analyze(self, completion, state: dict) -> dict:
        """Run both detectors once per rollout and cache on the state.

        The reward and three of the metrics all need the same scan of a text that
        can be 32k tokens long; the rubric calls them separately.
        """
        cached = state.get("_repetition_analysis")
        if cached is not None:
            return cached

        text = _completion_text(completion)
        has_repetition, reason, _ = detect_exact_block_repetition(text, **self._thresholds)

        # Share of n-gram slots belonging to an n-gram that occurs more than once.
        # Continuous, unlike the reward, so it keeps moving after the block
        # detector has stopped firing on everything.
        words = text.lower().split()
        slots = len(words) - self._ngram_size + 1
        if slots > 0:
            repeated = find_ngram_repetitions(text, n=self._ngram_size)
            ngram_fraction = sum(len(positions) for positions in repeated.values()) / slots
        else:
            ngram_fraction = 0.0

        coverage, max_repeats = block_coverage(text)
        reasoning, answer = _split_reasoning(text)

        analysis = {
            "has_repetition": has_repetition,
            "reason": reason,
            "ngram_fraction": ngram_fraction,
            "block_coverage": coverage,
            "max_block_repeats": max_repeats,
            "has_reasoning": bool(reasoning.strip()),
            "has_answer": bool(answer.strip()),
        }
        state["_repetition_analysis"] = analysis
        return analysis

    def no_repetition(self, completion, state: dict, **kwargs) -> float:
        # Without these floors, saying nothing is a perfect answer: it repeats nothing.
        if _output_tokens(state) < self._min_completion_tokens:
            return 0.0
        analysis = self._analyze(completion, state)
        if self._require_reasoning_and_answer and not (analysis["has_reasoning"] and analysis["has_answer"]):
            return 0.0
        if self._reward_mode == "detector":
            return 0.0 if analysis["has_repetition"] else 1.0
        return 1.0 - analysis["block_coverage"]

    def block_coverage(self, completion, state: dict, **kwargs) -> float:
        return self._analyze(completion, state)["block_coverage"]

    def max_block_repeats(self, completion, state: dict, **kwargs) -> float:
        return float(self._analyze(completion, state)["max_block_repeats"])

    def repetition_detected(self, completion, state: dict, **kwargs) -> float:
        return float(self._analyze(completion, state)["has_repetition"])

    def repeated_ngram_fraction(self, completion, state: dict, **kwargs) -> float:
        return self._analyze(completion, state)["ngram_fraction"]

    def has_reasoning(self, completion, state: dict, **kwargs) -> float:
        return float(self._analyze(completion, state)["has_reasoning"])

    def has_answer(self, completion, state: dict, **kwargs) -> float:
        return float(self._analyze(completion, state)["has_answer"])

    def truncated(self, state: dict, **kwargs) -> float:
        return float(state.get("is_truncated", False))

    def output_tokens(self, state: dict, **kwargs) -> float:
        return float(_output_tokens(state))


def load_environment(
    dataset_path: str = _DEFAULT_DATASET_PATH,
    categories: list[str] = ("math",),
    use_finnish: bool = False,
    min_passrate: float = 0.05,
    max_passrate: float = 0.95,
    exclude_instruction_ids: list[str] = _REPEAT_INSTRUCTION_IDS,
    dataset_shuffle: bool = True,
    dataset_seed: int = 42,
    system_prompt: str | None = _SYSTEM_PROMPT,
    reward_mode: str = "blockcov",
    min_repetitions: int = 10,
    min_sentence_repetitions: int = 40,
    min_consecutive_sentence_repetitions: int = 4,
    min_completion_tokens: int = 64,
    require_reasoning_and_answer: bool = True,
    ngram_size: int = 3,
    map_kwargs: dict = {},
    **kwargs,
) -> vf.Environment:
    """
    Load the repetition-only environment.

    Args:
        dataset_path:   Dolci parquet directory, or the Finnish translated jsonl.
        categories:     Dolci categories to keep. Math is the default because it is the
                        only split measured to loop often enough to train on (job 53239).
                        Instantiate the env once per category rather than passing several
                        at once -- separate envs give separate reward traces, which is the
                        only way to see which prompt distribution is carrying the signal.
        use_finnish:    Read translated_prompt instead of prompt.
        min_passrate:   Drop examples below this passrate. The default drops dolci's
                        `passrate == 0` rows and nothing else: the next value up is
                        1/8, and that floor is the one selection the probe justified.
        max_passrate:   Drop examples above this passrate. Inert on the math split,
                        which the dataset authors already cut at 0.75.
        exclude_instruction_ids:
                        Drop examples whose ground_truth names one of these IFEval
                        instruction ids. Needed for the ifeval split, where a sixth
                        of the prompts demand literal repetition and complying with
                        them is exactly what this reward punishes. Pass [] to keep them.
        system_prompt:  Prepended to every conversation. Pass None or "" to disable.
        reward_mode:    "blockcov" rewards `1 - block_coverage`, continuous, so every
                        group has gradient (100% of k=8 groups vary, against 28-56%
                        for the detector). "detector" is the binary open-instruct
                        reward. GRPO here centers by the group mean without dividing
                        by its std (`algo/grpo.py`), so a group that barely varies
                        yields proportionally small advantages rather than amplified
                        noise -- which is what makes the continuous mode safe.
        min_repetitions, min_sentence_repetitions, min_consecutive_sentence_repetitions:
                        Thresholds handed to open-instruct's detector. The defaults are
                        upstream's own (10 / 40 / 4), tuned for filtering SFT corpora;
                        lower them to make the reward fire on milder looping. In
                        "blockcov" mode they only affect the repetition_detected metric.
        min_completion_tokens:
                        Rollouts shorter than this score 0 regardless of repetition,
                        which is what stops "answer with nothing" from being optimal.
        require_reasoning_and_answer:
                        Also require at least one non-whitespace character on *each*
                        side of </think>. A rollout that never closes its thinking
                        block scores 0, as does an empty <think></think> -- so if the
                        chat template silently disables thinking, has_reasoning goes
                        to 0 and the whole batch zeroes out instead of training on it.
        ngram_size:     n for the repeated_ngram_fraction diagnostic only.
    """
    if use_finnish and system_prompt == _SYSTEM_PROMPT:
        system_prompt = _SYSTEM_PROMPT_FI
    system_prompt = system_prompt or None

    categories = list(categories)

    def build_dataset():
        raw = _load_raw(dataset_path, categories, use_finnish=use_finnish)

        if "passrate" in raw.column_names:
            import math

            raw = raw.filter(
                lambda x: (
                    x["passrate"] is None
                    or (isinstance(x["passrate"], float) and math.isnan(x["passrate"]))
                    or (x["passrate"] >= min_passrate and x["passrate"] <= max_passrate)
                ),
                **map_kwargs,
            )

        if exclude_instruction_ids and "ground_truth" in raw.column_names:
            # ground_truth serialises the instruction list as a Python literal; the ids
            # are namespaced tokens, so a substring test is exact here and cannot be
            # tripped by the prose around them. Verified against an ast.literal_eval
            # pass: identical 4334 rows on the ifeval split.
            excluded = tuple(exclude_instruction_ids)
            raw = raw.filter(lambda x: not any(e in str(x["ground_truth"]) for e in excluded), **map_kwargs)

        examples = [ex for ex in (_build_example(row) for row in raw) if ex is not None]
        if not examples:
            raise RuntimeError(
                f"No usable examples found in {dataset_path} for categories={categories}. "
                "Check dataset path and passrate filters."
            )

        ds = Dataset.from_list(examples)
        if dataset_shuffle:
            ds = ds.shuffle(seed=dataset_seed)
        return ds

    rubric = RepetitionRubric(
        min_repetitions=min_repetitions,
        min_sentence_repetitions=min_sentence_repetitions,
        min_consecutive_sentence_repetitions=min_consecutive_sentence_repetitions,
        min_completion_tokens=min_completion_tokens,
        require_reasoning_and_answer=require_reasoning_and_answer,
        ngram_size=ngram_size,
        reward_mode=reward_mode,
    )

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
        **kwargs,
    )
