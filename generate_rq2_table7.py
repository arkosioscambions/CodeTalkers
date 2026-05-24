#!/usr/bin/env python3
"""Generate the RQ2 Table 7 behavioral metrics as CSV and JSON.

This script reproduces the weighted behavioral metrics reported in Table 7 of
the TOSEM manuscript, but writes machine-readable outputs instead of LaTeX.

It aggregates the same benchmark outputs used by the earlier fidelity scripts:
- Infilling: HEFIM, SAFIM (block/control/api weighted), ClassEval-LineInfilling
- Completion: ClassEval-Completion, BigCodeBench-Complete (raw_solution)

Each Table 7 cell is represented as an overall weighted metric together with
its failure-conditioned counterpart computed on failed samples only.
"""

from __future__ import annotations

import argparse
import csv
import json
import keyword
import re
import token
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import mean


CODETALKERS_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_ROOT = CODETALKERS_ROOT.parent

HEFIM_WEIGHT = 1033
CLASS_EVAL_INFILLING_WEIGHT = 2557
CLASS_EVAL_COMPLETION_WEIGHT = 396
BIGCODEBENCH_COMPLETE_WEIGHT = 1140
SAFIM_WEIGHTS = {"block": 8780, "control": 8630, "api": 310}
SAFIM_WEIGHT_SUM = sum(SAFIM_WEIGHTS.values())
CTR_THRESHOLD = 0.40

TEXT_KEYS = [
    "completion",
    "solution",
    "code",
    "prediction",
    "generated_text",
    "output",
    "text",
    "full_code",
]

FAILURE_CASES = [
    "empty_completion",
    "comment_only_output",
    "low_code_token_ratio",
    "markdown_fence_or_heading_or_bullet",
    "natural_language_markers",
    "extra_def_or_class",
    "special_tokens",
]

NL_MARKER_RE = re.compile(
    r"\b("
    r"here|this|explanation|because|therefore|note|please|you|we|"
    r"the following|in summary|overall|i think|let me"
    r")\b",
    re.IGNORECASE,
)
DEF_CLASS_RE = re.compile(r"^\s*(def|class)\s+\w+", re.MULTILINE)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*([-*+]\s+|\d+\.\s+)", re.MULTILINE)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")


@dataclass(frozen=True)
class ModelSpec:
    family_key: str
    family_display: str
    display_label: str
    size_key: str
    variant: str
    hefim_rel: str
    safim_block_rel: str
    safim_control_rel: str
    safim_api_rel: str
    classeval_infilling_rel: str
    classeval_completion_rel: str
    bigcodebench_raw_rel: str

    def resolved_paths(self, workspace_root: Path) -> dict[str, Path]:
        return {
            "hefim": workspace_root / self.hefim_rel,
            "safim_block": workspace_root / self.safim_block_rel,
            "safim_control": workspace_root / self.safim_control_rel,
            "safim_api": workspace_root / self.safim_api_rel,
            "classeval_infilling": workspace_root / self.classeval_infilling_rel,
            "classeval_completion": workspace_root / self.classeval_completion_rel,
            "bigcodebench_raw": workspace_root / self.bigcodebench_raw_rel,
        }


# Edit these relative paths if your benchmark artifacts are stored in a
# different layout on your machine.
MODEL_SPECS = [
    ModelSpec(
        family_key="qwen",
        family_display="Qwen2.5-Coder",
        display_label="1.5B",
        size_key="1.5b",
        variant="base",
        hefim_rel="fidelity/HEFIM/qwen/re_samples_base-1.5b.jsonl",
        safim_block_rel="fidelity/safim-block/qwen/Qwen2.5-Coder-1.5B-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/qwen/Qwen2.5-Coder-1.5B-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/qwen/api_generate_samples_base1.5b_all.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/qwen/qwen1.5base_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/qwen-0.25/Qwen2.5-Coder-1.5B_ClassEval-Completion_3rd.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/qwen/Qwen--Qwen2.5-Coder-1.5B--main--bigcodebench-complete--hf-0-1-sanitized_calibrated (2).jsonl",
    ),
    ModelSpec(
        family_key="qwen",
        family_display="Qwen2.5-Coder",
        display_label="7B",
        size_key="7b",
        variant="base",
        hefim_rel="fidelity/HEFIM/qwen/re_samples_base-7B.jsonl",
        safim_block_rel="fidelity/safim-block/qwen/Qwen2.5-Coder-7B-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/qwen/Qwen2.5-Coder-7B-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/qwen/api_generate_samples_base7b_all.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/qwen/qwen7bbase_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/qwen-0.25/Qwen2.5-Coder-7B_ClassEval-Completion-2nd.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/qwen/Qwen--Qwen2.5-Coder-7B--main--bigcodebench-complete--hf-0-1-sanitized_calibrated.jsonl",
    ),
    ModelSpec(
        family_key="qwen",
        family_display="Qwen2.5-Coder",
        display_label="14B",
        size_key="14b",
        variant="base",
        hefim_rel="fidelity/HEFIM/qwen/he-qwen-base-14B.jsonl",
        safim_block_rel="fidelity/safim-block/qwen/Qwen2.5-Coder-14B-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/qwen/Qwen2.5-Coder-14B-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/qwen/api_generate_samples_base14b_all.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/qwen/Qwen2.5-Coder-14B_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/qwen-0.25/Qwen2.5-Coder-14B-ClassEval-Completion-0.25-2nd.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/qwen/Qwen--Qwen2.5-Coder-14B--main--bigcodebench-complete--hf-0-1-sanitized_calibrated (1).jsonl",
    ),
    ModelSpec(
        family_key="qwen",
        family_display="Qwen2.5-Coder",
        display_label="32B",
        size_key="32b",
        variant="base",
        hefim_rel="fidelity/HEFIM/qwen/he-qwen-base-32B.jsonl",
        safim_block_rel="fidelity/safim-block/qwen/Qwen2.5-Coder-32B-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/qwen/Qwen2.5-Coder-32B-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/qwen/api_generate_samples_qwenbase32b.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/qwen/Qwen2.5-Coder-32B_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/qwen-0.25/Qwen2.5-Coder-32B-ClassEval-Completion-0.25-pass1.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/qwen/Qwen--Qwen2.5-Coder-32B--main--bigcodebench-complete--hf-0-1-sanitized_calibrated (1).jsonl",
    ),
    ModelSpec(
        family_key="qwen",
        family_display="Qwen2.5-Coder",
        display_label="1.5B-Instruct",
        size_key="1.5b",
        variant="instruct",
        hefim_rel="fidelity/HEFIM/qwen/re_samples_instruct-1.5b.jsonl",
        safim_block_rel="fidelity/safim-block/qwen/Qwen2.5-Coder-1.5B-Instruct-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/qwen/Qwen2.5-Coder-1.5B-Instruct-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/qwen/api_generate_samples_instruct1.5b_all.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/qwen/qwen1.5binstruct_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/qwen-0.25/Qwen2.5-Coder-1.5B-Instruct_ClassEval-Completion_1st.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/qwen/Qwen--Qwen2.5-Coder-1.5B-Instruct--main--bigcodebench-complete--hf-0-1-sanitized_calibrated.jsonl",
    ),
    ModelSpec(
        family_key="qwen",
        family_display="Qwen2.5-Coder",
        display_label="7B-Instruct",
        size_key="7b",
        variant="instruct",
        hefim_rel="fidelity/HEFIM/qwen/re_samples_instruct-7B.jsonl",
        safim_block_rel="fidelity/safim-block/qwen/Qwen2.5-Coder-7B-Instruct-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/qwen/Qwen2.5-Coder-7B-Instruct-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/qwen/api_generate_samples_instruct7b_all.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/qwen/qwen7binstruct_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/qwen-0.25/Qwen2.5-Coder-7B-Instruct_ClassEval-Completion-1st.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/qwen/Qwen--Qwen2.5-Coder-7B-Instruct--main--bigcodebench-complete--hf-0-1-sanitized_calibrated.jsonl",
    ),
    ModelSpec(
        family_key="qwen",
        family_display="Qwen2.5-Coder",
        display_label="14B-Instruct",
        size_key="14b",
        variant="instruct",
        hefim_rel="fidelity/HEFIM/qwen/he-qwen-instruct-14B.jsonl",
        safim_block_rel="fidelity/safim-block/qwen/Qwen2.5-Coder-14B-Instruct-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/qwen/Qwen2.5-Coder-14B-Instruct-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/qwen/api_generate_samples_instruct_14b_all.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/qwen/Qwen2.5-Coder-14B-Instruct_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/qwen-0.25/Qwen2.5-Coder-14B-Instruct-ClassEval-Completion-0.25-pass1.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/qwen/Qwen--Qwen2.5-Coder-14B-Instruct--main--bigcodebench-complete--hf-0-1-sanitized_calibrated (1).jsonl",
    ),
    ModelSpec(
        family_key="qwen",
        family_display="Qwen2.5-Coder",
        display_label="32B-Instruct",
        size_key="32b",
        variant="instruct",
        hefim_rel="fidelity/HEFIM/qwen/he-qwen-instruct-32B.jsonl",
        safim_block_rel="fidelity/safim-block/qwen/Qwen2.5-Coder-32B-Instruct-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/qwen/Qwen2.5-Coder-32B-Instruct-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/qwen/api_generate_samples_qweninstruct32b.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/qwen/Qwen2.5-Coder-32B-Instruct_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/qwen-0.25/Qwen2.5-Coder-32B-Instruct-ClassEval-Completion-0.25-2nd.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/qwen/Qwen--Qwen2.5-Coder-32B-Instruct--main--bigcodebench-complete--hf-0-1-sanitized_calibrated (1).jsonl",
    ),
    ModelSpec(
        family_key="dscoder",
        family_display="DeepSeek-Coder",
        display_label="1.3B",
        size_key="1.3b",
        variant="base",
        hefim_rel="fidelity/HEFIM/dscoder/HEFIM-dsc-1.3b-base-trimmed-before-hash-from.jsonl",
        safim_block_rel="fidelity/safim-block/dscoder/deepseek-coder-1.3b-base-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/dscoder/deepseek-coder-1.3b-base-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/dscoder/SAFIM-api-dsc-1.3b-base.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/dscoder/dscoder1.3bbase_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/dscoder-0.1/deepseek-coder-1.3b-base-ClassEval-Completion-changedpromptv4.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/dscoder/deepseek-ai--deepseek-coder-1.3b-base--main--bigcodebench-complete--hf-0-1-sanitized_calibrated.jsonl",
    ),
    ModelSpec(
        family_key="dscoder",
        family_display="DeepSeek-Coder",
        display_label="6.7B",
        size_key="6.7b",
        variant="base",
        hefim_rel="fidelity/HEFIM/dscoder/HEFIM-dsc-6.7b-base-trimmed-before-hash-from.jsonl",
        safim_block_rel="fidelity/safim-block/dscoder/deepseek-coder-6.7b-base-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/dscoder/deepseek-coder-6.7b-base-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/dscoder/SAFIM-api-dsc-6.7b-base.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/dscoder/deepseek-coder-6.7b-base_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/dscoder-0.1/deepseek-coder-6.7b-base-ClassEval-Completion-changedpromptv4.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/dscoder/deepseek-ai--deepseek-coder-6.7b-base--main--bigcodebench-complete--hf-0-1-sanitized_calibrated.jsonl",
    ),
    ModelSpec(
        family_key="dscoder",
        family_display="DeepSeek-Coder",
        display_label="33B",
        size_key="33b",
        variant="base",
        hefim_rel="fidelity/HEFIM/dscoder/HEFIM-dsc-33b-base-first-nonempty-line.jsonl",
        safim_block_rel="fidelity/safim-block/dscoder/deepseek-coder-33b-base-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/dscoder/deepseek-coder-33b-base-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/dscoder/SAFIM-api-dsc-33b-base.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/dscoder/deepseek-coder-33b-base_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/dscoder-0.1/deepseek-coder-33b-base-ClassEval-Completion-changedpromptv4-pass1.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/dscoder/deepseek-ai--deepseek-coder-33b-base--main--bigcodebench-complete--hf-0-1-sanitized_calibrated.jsonl",
    ),
    ModelSpec(
        family_key="dscoder",
        family_display="DeepSeek-Coder",
        display_label="1.3B-Instruct",
        size_key="1.3b",
        variant="instruct",
        hefim_rel="fidelity/HEFIM/dscoder/HEI-dsc-1.3b-instruct-trimmed-before-hash-from.jsonl",
        safim_block_rel="fidelity/safim-block/dscoder/deepseek-coder-1.3b-instruct-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/dscoder/deepseek-coder-1.3b-instruct-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/dscoder/SAFIM-api-1.3b-instruct.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/dscoder/dscoder1.3binstruct_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/dscoder-0.1/deepseek-coder-1.3b-instruct-ClassEval-Completion-changedpromptv4.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/dscoder/deepseek-ai--deepseek-coder-1.3b-instruct--main--bigcodebench-complete--hf-0-1-sanitized_calibrated.jsonl",
    ),
    ModelSpec(
        family_key="dscoder",
        family_display="DeepSeek-Coder",
        display_label="6.7B-Instruct",
        size_key="6.7b",
        variant="instruct",
        hefim_rel="fidelity/HEFIM/dscoder/HEI-dsc-6.7b-instruct-trimmed-before-hash-from.jsonl",
        safim_block_rel="fidelity/safim-block/dscoder/deepseek-coder-6.7b-instruct-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/dscoder/deepseek-coder-6.7b-instruct-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/dscoder/SAFIM-api-6.7b-instruct.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/dscoder/deepseek-coder-6.7b-instruct_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/dscoder-0.1/deepseek-coder-6.7b-instruct-ClassEval-Completion-changedpromptv4.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/dscoder/deepseek-ai--deepseek-coder-6.7b-instruct--main--bigcodebench-complete--hf-0-1-sanitized_calibrated.jsonl",
    ),
    ModelSpec(
        family_key="dscoder",
        family_display="DeepSeek-Coder",
        display_label="33B-Instruct",
        size_key="33b",
        variant="instruct",
        hefim_rel="fidelity/HEFIM/dscoder/HEI-dsc-33b-instruct-trimmed-before-hash-from.jsonl",
        safim_block_rel="fidelity/safim-block/dscoder/deepseek-coder-33b-instruct-fim-tb.jsonl",
        safim_control_rel="fidelity/safim-control/dscoder/deepseek-coder-33b-instruct-fim-tc.jsonl",
        safim_api_rel="fidelity/safim-api/dscoder/SAFIM-api-33b-instruct.jsonl",
        classeval_infilling_rel="fidelity/classevalinfilling/dscoder/deepseek-coder-33b-instruct_ClassEval-LineInfilling.jsonl",
        classeval_completion_rel="fidelity/classevalcompletion/dscoder-0.1/deepseek-coder-33b-instruct-ClassEval-Completion-changedpromptv4-maxcorrect.jsonl",
        bigcodebench_raw_rel="fidelity/bigcodebench/raw/dscoder/deepseek-ai--deepseek-coder-33b-instruct--main--bigcodebench-complete--hf-0-1-sanitized_calibrated (1).jsonl",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RQ2 Table 7 behavioral metrics as CSV and JSON."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
        help="Workspace root containing both CodeTalkers/ and fidelity/.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=CODETALKERS_ROOT / "results" / "rq2_table7.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=CODETALKERS_ROOT / "results" / "rq2_table7.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def relative_to_root(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.relative_to(workspace_root))
    except ValueError:
        return str(path)


def ensure_paths_exist(specs: list[ModelSpec], workspace_root: Path) -> None:
    missing: list[str] = []
    for spec in specs:
        for path in spec.resolved_paths(workspace_root).values():
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(sorted(missing)))


def extract_default_text(record: dict) -> str:
    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def extract_raw_solution(record: dict) -> str:
    value = record.get("raw_solution")
    return value if isinstance(value, str) else ""


def is_empty_output(text: str) -> bool:
    return text.strip() == ""


def is_comment_only_output(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return all(line.startswith("#") for line in lines)


def code_token_ratio(text: str) -> float:
    try:
        tokens = list(tokenize.generate_tokens(StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return 0.0

    meaningful = []
    for tok in tokens:
        if tok.type in (
            token.NEWLINE,
            tokenize.NL,
            token.INDENT,
            token.DEDENT,
            token.ENDMARKER,
            tokenize.ENCODING,
        ):
            continue
        if tok.type == token.COMMENT:
            continue
        meaningful.append(tok)

    if not meaningful:
        return 0.0

    code_like = 0
    for tok in meaningful:
        if tok.type in (token.OP, token.NUMBER, token.STRING):
            code_like += 1
        elif tok.type == token.NAME:
            if keyword.iskeyword(tok.string) or re.match(r"^[A-Za-z_]\w*$", tok.string):
                code_like += 1

    return code_like / len(meaningful)


def has_markdown(text: str) -> bool:
    return "```" in text or bool(HEADING_RE.search(text)) or bool(BULLET_RE.search(text))


def has_nl_markers(text: str) -> bool:
    if NL_MARKER_RE.search(text):
        return True
    alpha_words = re.findall(r"[A-Za-z]{3,}", text)
    punct = re.findall(r"[(){}\[\]:=,+\-*/%<>]", text)
    return len(alpha_words) >= 8 and len(punct) <= 2


def has_extra_def_or_class(text: str) -> bool:
    return bool(DEF_CLASS_RE.search(text))


def has_special_tokens(text: str) -> bool:
    return bool(SPECIAL_TOKEN_RE.search(text))


def classify_text(text: str) -> tuple[bool, float, dict[str, bool]]:
    ratio = code_token_ratio(text)
    flags = {key: False for key in FAILURE_CASES}

    if is_empty_output(text):
        flags["empty_completion"] = True
    elif is_comment_only_output(text):
        flags["comment_only_output"] = True
    elif ratio < CTR_THRESHOLD:
        flags["low_code_token_ratio"] = True

    if has_markdown(text):
        flags["markdown_fence_or_heading_or_bullet"] = True
    if has_nl_markers(text):
        flags["natural_language_markers"] = True
    if has_extra_def_or_class(text):
        flags["extra_def_or_class"] = True
    if has_special_tokens(text):
        flags["special_tokens"] = True

    return any(flags.values()), ratio, flags


def collect_metrics(path: Path, extractor) -> dict[str, float | int]:
    texts: list[str] = []
    ratios: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = extractor(record)
            texts.append(text)
            ratios.append(code_token_ratio(text))

    counts = {key: 0 for key in FAILURE_CASES}
    any_failure = 0
    failed_ratios: list[float] = []
    for text in texts:
        failed, ratio, flags = classify_text(text)
        for key, active in flags.items():
            if active:
                counts[key] += 1
        if failed:
            any_failure += 1
            failed_ratios.append(ratio)

    num_samples = len(texts)
    return {
        "num_samples": num_samples,
        "mean_ctr": mean(ratios) if ratios else 0.0,
        "failed_mean_ctr": mean(failed_ratios) if failed_ratios else 0.0,
        "natural_language_markers_rate": (
            counts["natural_language_markers"] / num_samples if num_samples else 0.0
        ),
        "natural_language_markers_given_failure_rate": (
            counts["natural_language_markers"] / any_failure if any_failure else 0.0
        ),
        "extra_def_or_class_rate": (
            counts["extra_def_or_class"] / num_samples if num_samples else 0.0
        ),
        "extra_def_or_class_given_failure_rate": (
            counts["extra_def_or_class"] / any_failure if any_failure else 0.0
        ),
    }


def weighted_metric(parts: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _, weight in parts)
    if total_weight == 0:
        return 0.0
    return sum(value * weight for value, weight in parts) / total_weight


def weighted_safim(metrics_by_path: dict[Path, dict[str, float | int]], paths: dict[str, Path], key: str) -> float:
    return weighted_metric(
        [
            (float(metrics_by_path[paths["safim_block"]][key]), SAFIM_WEIGHTS["block"]),
            (float(metrics_by_path[paths["safim_control"]][key]), SAFIM_WEIGHTS["control"]),
            (float(metrics_by_path[paths["safim_api"]][key]), SAFIM_WEIGHTS["api"]),
        ]
    )


def weighted_infilling_metric(
    metrics_by_path: dict[Path, dict[str, float | int]], paths: dict[str, Path], key: str
) -> float:
    return weighted_metric(
        [
            (float(metrics_by_path[paths["hefim"]][key]), HEFIM_WEIGHT),
            (weighted_safim(metrics_by_path, paths, key), SAFIM_WEIGHT_SUM),
            (float(metrics_by_path[paths["classeval_infilling"]][key]), CLASS_EVAL_INFILLING_WEIGHT),
        ]
    )


def weighted_completion_metric(
    metrics_by_path: dict[Path, dict[str, float | int]], paths: dict[str, Path], key: str
) -> float:
    return weighted_metric(
        [
            (float(metrics_by_path[paths["classeval_completion"]][key]), CLASS_EVAL_COMPLETION_WEIGHT),
            (float(metrics_by_path[paths["bigcodebench_raw"]][key]), BIGCODEBENCH_COMPLETE_WEIGHT),
        ]
    )


def metric_bundle(overall: float, failure_only: float) -> dict[str, float | str]:
    return {
        "overall": overall,
        "failure_only": failure_only,
        "overall_pct": round(100.0 * overall, 2),
        "failure_only_pct": round(100.0 * failure_only, 2),
        "table7": f"{100.0 * overall:.2f} ({100.0 * failure_only:.2f})",
    }


def build_result_rows(
    specs: list[ModelSpec],
    workspace_root: Path,
    metrics_by_path: dict[Path, dict[str, float | int]],
) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        paths = spec.resolved_paths(workspace_root)
        infilling_ctr = metric_bundle(
            weighted_infilling_metric(metrics_by_path, paths, "mean_ctr"),
            weighted_infilling_metric(metrics_by_path, paths, "failed_mean_ctr"),
        )
        infilling_nl = metric_bundle(
            weighted_infilling_metric(metrics_by_path, paths, "natural_language_markers_rate"),
            weighted_infilling_metric(
                metrics_by_path, paths, "natural_language_markers_given_failure_rate"
            ),
        )
        infilling_defcls = metric_bundle(
            weighted_infilling_metric(metrics_by_path, paths, "extra_def_or_class_rate"),
            weighted_infilling_metric(metrics_by_path, paths, "extra_def_or_class_given_failure_rate"),
        )
        completion_ctr = metric_bundle(
            weighted_completion_metric(metrics_by_path, paths, "mean_ctr"),
            weighted_completion_metric(metrics_by_path, paths, "failed_mean_ctr"),
        )
        completion_nl = metric_bundle(
            weighted_completion_metric(metrics_by_path, paths, "natural_language_markers_rate"),
            weighted_completion_metric(
                metrics_by_path, paths, "natural_language_markers_given_failure_rate"
            ),
        )
        completion_defcls = metric_bundle(
            weighted_completion_metric(metrics_by_path, paths, "extra_def_or_class_rate"),
            weighted_completion_metric(metrics_by_path, paths, "extra_def_or_class_given_failure_rate"),
        )

        rows.append(
            {
                "family_key": spec.family_key,
                "family": spec.family_display,
                "model": spec.display_label,
                "size_key": spec.size_key,
                "variant": spec.variant,
                "sources": {
                    key: relative_to_root(path, workspace_root) for key, path in paths.items()
                },
                "infilling": {
                    "ctr": infilling_ctr,
                    "nl": infilling_nl,
                    "defcls": infilling_defcls,
                },
                "completion": {
                    "ctr": completion_ctr,
                    "nl": completion_nl,
                    "defcls": completion_defcls,
                },
            }
        )
    return rows


def csv_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    flattened = []
    for row in rows:
        infilling = row["infilling"]
        completion = row["completion"]
        flattened.append(
            {
                "family": row["family"],
                "family_key": row["family_key"],
                "model": row["model"],
                "size_key": row["size_key"],
                "variant": row["variant"],
                "infilling_ctr_overall_pct": infilling["ctr"]["overall_pct"],
                "infilling_ctr_failure_only_pct": infilling["ctr"]["failure_only_pct"],
                "infilling_ctr_table7": infilling["ctr"]["table7"],
                "infilling_nl_overall_pct": infilling["nl"]["overall_pct"],
                "infilling_nl_failure_only_pct": infilling["nl"]["failure_only_pct"],
                "infilling_nl_table7": infilling["nl"]["table7"],
                "infilling_defcls_overall_pct": infilling["defcls"]["overall_pct"],
                "infilling_defcls_failure_only_pct": infilling["defcls"]["failure_only_pct"],
                "infilling_defcls_table7": infilling["defcls"]["table7"],
                "completion_ctr_overall_pct": completion["ctr"]["overall_pct"],
                "completion_ctr_failure_only_pct": completion["ctr"]["failure_only_pct"],
                "completion_ctr_table7": completion["ctr"]["table7"],
                "completion_nl_overall_pct": completion["nl"]["overall_pct"],
                "completion_nl_failure_only_pct": completion["nl"]["failure_only_pct"],
                "completion_nl_table7": completion["nl"]["table7"],
                "completion_defcls_overall_pct": completion["defcls"]["overall_pct"],
                "completion_defcls_failure_only_pct": completion["defcls"]["failure_only_pct"],
                "completion_defcls_table7": completion["defcls"]["table7"],
            }
        )
    return flattened


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_preview(rows: list[dict[str, object]]) -> None:
    print("family\tmodel\tinfilling_ctr\tinfilling_nl\tinfilling_defcls\tcompletion_ctr\tcompletion_nl\tcompletion_defcls")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["family"]),
                    str(row["model"]),
                    str(row["infilling"]["ctr"]["table7"]),
                    str(row["infilling"]["nl"]["table7"]),
                    str(row["infilling"]["defcls"]["table7"]),
                    str(row["completion"]["ctr"]["table7"]),
                    str(row["completion"]["nl"]["table7"]),
                    str(row["completion"]["defcls"]["table7"]),
                ]
            )
        )


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    ensure_paths_exist(MODEL_SPECS, workspace_root)

    metrics_by_path: dict[Path, dict[str, float | int]] = {}
    for spec in MODEL_SPECS:
        paths = spec.resolved_paths(workspace_root)
        for key, extractor in [
            ("hefim", extract_default_text),
            ("safim_block", extract_default_text),
            ("safim_control", extract_default_text),
            ("safim_api", extract_default_text),
            ("classeval_infilling", extract_default_text),
            ("classeval_completion", extract_default_text),
            ("bigcodebench_raw", extract_raw_solution),
        ]:
            path = paths[key]
            if path not in metrics_by_path:
                metrics_by_path[path] = collect_metrics(path, extractor)

    result_rows = build_result_rows(MODEL_SPECS, workspace_root, metrics_by_path)
    write_csv(args.output_csv, csv_rows(result_rows))
    write_json(
        args.output_json,
        {
            "metadata": {
                "generated_by": relative_to_root(Path(__file__).resolve(), workspace_root),
                "workspace_root": str(workspace_root),
                "description": "RQ2 Table 7 weighted behavioral metrics with failure-conditioned values.",
                "infilling_overall_source": "Computed from raw infilling benchmark outputs using the same weights as Table 7.",
                "weights": {
                    "hefim": HEFIM_WEIGHT,
                    "safim_block": SAFIM_WEIGHTS["block"],
                    "safim_control": SAFIM_WEIGHTS["control"],
                    "safim_api": SAFIM_WEIGHTS["api"],
                    "safim_total": SAFIM_WEIGHT_SUM,
                    "classeval_infilling": CLASS_EVAL_INFILLING_WEIGHT,
                    "classeval_completion": CLASS_EVAL_COMPLETION_WEIGHT,
                    "bigcodebench_complete": BIGCODEBENCH_COMPLETE_WEIGHT,
                },
                "failure_cases": FAILURE_CASES,
                "code_token_ratio_threshold": CTR_THRESHOLD,
                "bigcodebench_text_field": "raw_solution",
            },
            "rows": result_rows,
        },
    )

    print_preview(result_rows)
    print(f"\nWrote CSV : {args.output_csv}")
    print(f"Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
