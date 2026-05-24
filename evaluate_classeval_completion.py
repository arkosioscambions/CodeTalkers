#!/usr/bin/env python3
import argparse
import io
import json
import multiprocessing
import queue
import re
import textwrap
import traceback
import types
import unittest
from pathlib import Path

from datasets import load_dataset as hf_load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine ClassEval completion codes with model completions and run the provided tests."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="annachaaang/ClassEval-Completion",
        help="Hugging Face dataset name for ClassEval-Completion.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load from Hugging Face.",
    )
    parser.add_argument(
        "--completions",
        type=Path,
        default=Path("qwen1.5bbase_ClassEval-Completion.jsonl"),
        help="JSONL file containing model completions keyed by task_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ClassEval/output"),
        help="Directory where combined code and evaluation results will be written.",
    )
    parser.add_argument(
        "--per-task-timeout",
        type=float,
        default=15.0,
        help="Maximum number of seconds allowed for each test suite before timing out.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Optional start index (0-based, inclusive) for partial evaluation.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Optional end index (0-based, exclusive) for partial evaluation.",
    )
    return parser.parse_args()


def load_dataset(dataset_name, split):
    dataset = hf_load_dataset(dataset_name, split=split)
    columns = ["task_id", "new_task_id", "method_name", "completion_code", "test"]
    return {column: dataset[column] for column in columns}


def load_completions(jsonl_path):
    completions = {}
    with jsonl_path.open() as fh:
        for line in fh:
            record = json.loads(line)
            completions[record["task_id"]] = record["completion"]
    return completions


MAIN_GUARD_RE = re.compile(r'^\s*if __name__ == ["\']__main__["\']:\s*(?:#.*)?$', re.MULTILINE)
IGNORE_LINE_MARKERS = ("# END",)
TRIM_AFTER_MARKERS = ("# NOTE:",)
STOP_PREFIXES = ("def ", "class ", "@", "import ", "from ")


def remove_main_guard(block):
    match = MAIN_GUARD_RE.search(block)
    if match:
        return block[: match.start()]
    return block


def sanitize_completion(block):
    without_guard = remove_main_guard(block)
    sanitized_lines = []
    stop = False
    for raw_line in without_guard.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            sanitized_lines.append("")
            continue
        if any(marker in stripped_line for marker in IGNORE_LINE_MARKERS):
            continue
        line = raw_line
        for marker in TRIM_AFTER_MARKERS:
            idx = line.find(marker)
            if idx != -1:
                line = line[:idx]
        if STOP_PREFIXES and stripped_line and stripped_line.lstrip().startswith(STOP_PREFIXES):
            stop = True
            break
        sanitized_lines.append(line.rstrip())
    if stop:
        return "\n".join(sanitized_lines).rstrip()
    return "\n".join(sanitized_lines).rstrip()


def snippet_compiles(snippet):
    indented_lines = []
    for line in snippet.split("\n"):
        if line.strip() == "":
            indented_lines.append("")
        else:
            indented_lines.append("    " + line)
    wrapper = "def __temp_func():\n" + "\n".join(indented_lines) + "\n"
    try:
        compile(wrapper, "<string>", "exec")
        return True
    except IndentationError:
        return False
    except SyntaxError:
        return True


def normalize_snippet(snippet):
    snippet = snippet.strip("\n")
    if not snippet:
        return ""
    if snippet_compiles(snippet):
        return snippet
    flattened = "\n".join(line.lstrip() for line in snippet.split("\n"))
    if snippet_compiles(flattened):
        return flattened
    return snippet


def indent_completion_block(block, base_indent=8):
    cleaned = sanitize_completion(block)
    if not cleaned.strip():
        return ""
    dedented = textwrap.dedent(cleaned.replace("\t", "    "))

    def strip_base_indent(lines):
        stripped = []
        for line in lines:
            if line.strip() == "":
                stripped.append("")
                continue
            indent_len = len(line) - len(line.lstrip(" "))
            if indent_len >= base_indent:
                stripped.append(line[base_indent:])
            else:
                stripped.append(line)
        return stripped

    stripped_lines = strip_base_indent(dedented.split("\n"))
    normalized = normalize_snippet("\n".join(stripped_lines))
    lines = normalized.split("\n")
    indented = []
    for line in lines:
        if line.strip() == "":
            indented.append("")
        else:
            indented.append(" " * base_indent + line.rstrip())
    return "\n".join(indented)


def build_complete_code(base_code, completion_snippet):
    snippet = indent_completion_block(completion_snippet)
    normalized_base = base_code.rstrip()
    if not normalized_base.endswith("\n"):
        normalized_base += "\n"
    return f"{normalized_base}{snippet}\n"


def run_tests(complete_code, test_code, module_name="task_module"):
    module = types.ModuleType(module_name)
    namespace = module.__dict__
    try:
        exec(complete_code, namespace)
    except Exception:
        return {
            "status": "code_error",
            "num_tests": 0,
            "num_failures": 0,
            "num_errors": 1,
            "details": [
                {"test": "code_compilation", "traceback": traceback.format_exc()}
            ],
            "log": "",
        }

    try:
        exec(test_code, namespace)
    except Exception:
        return {
            "status": "test_error",
            "num_tests": 0,
            "num_failures": 0,
            "num_errors": 1,
            "details": [
                {"test": "test_compilation", "traceback": traceback.format_exc()}
            ],
            "log": "",
        }

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    details = [
        {"test": str(case), "traceback": err}
        for case, err in result.failures + result.errors
    ]
    status = "passed" if result.wasSuccessful() else "failed"
    return {
        "status": status,
        "num_tests": result.testsRun,
        "num_failures": len(result.failures),
        "num_errors": len(result.errors),
        "details": details,
        "log": stream.getvalue(),
    }


def _evaluation_worker(work_queue, code, test):
    work_queue.put(run_tests(code, test))


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset, args.split)
    completions = load_completions(args.completions)

    output_dir = args.output_dir
    result_dir = output_dir / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    complete_code_path = output_dir / "complete_code.jsonl"
    detailed_result_path = result_dir / "detailed_result.json"
    summary_path = result_dir / "pass_at_k_result.json"

    total_rows = len(dataset["task_id"])
    detailed_results = []
    passed = 0
    code_errors = 0
    test_errors = 0
    timeouts = 0
    worker_errors = 0

    start_index = max(0, args.start_index)
    end_index = total_rows if args.end_index is None else min(args.end_index, total_rows)
    if start_index >= end_index:
        raise ValueError("start-index must be less than end-index.")
    segment_total = end_index - start_index

    with complete_code_path.open("w") as code_file:
        for offset, idx in enumerate(range(start_index, end_index), start=1):
            task_id = dataset["task_id"][idx]
            new_task_id = dataset["new_task_id"][idx]
            method_name = dataset["method_name"][idx]
            base_code = dataset["completion_code"][idx]
            test_code = dataset["test"][idx]
            completion = completions.get(new_task_id)
            if completion is None:
                result = {
                    "task_id": task_id,
                    "new_task_id": new_task_id,
                    "method_name": method_name,
                    "status": "missing_completion",
                    "num_tests": 0,
                    "num_failures": 0,
                    "num_errors": 0,
                    "details": [{"test": "completion_lookup", "traceback": ""}],
                    "log": "",
                }
                detailed_results.append(result)
                continue

            complete_code = build_complete_code(base_code, completion)
            json.dump(
                {
                    "task_id": task_id,
                    "new_task_id": new_task_id,
                    "method_name": method_name,
                    "complete_code": complete_code,
                },
                code_file,
            )
            code_file.write("\n")

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_evaluation_worker, args=(result_queue, complete_code, test_code)
            )
            process.start()
            process.join(args.per_task_timeout)
            if process.is_alive():
                process.terminate()
                process.join()
                timeouts += 1
                test_result = {
                    "status": "timeout",
                    "num_tests": 0,
                    "num_failures": 0,
                    "num_errors": 0,
                    "details": [
                        {
                            "test": "timeout",
                            "traceback": f"Timed out after {args.per_task_timeout} seconds.",
                        }
                    ],
                    "log": "",
                }
            else:
                try:
                    test_result = result_queue.get_nowait()
                except queue.Empty:
                    test_result = {
                        "status": "worker_error",
                        "num_tests": 0,
                        "num_failures": 0,
                        "num_errors": 0,
                        "details": [
                            {
                                "test": "worker_error",
                                "traceback": "Worker finished without returning a result.",
                            }
                        ],
                        "log": "",
                    }
                finally:
                    result_queue.close()
                    result_queue.join_thread()

            status = test_result["status"]
            if status == "passed":
                passed += 1
            elif status == "code_error":
                code_errors += 1
            elif status == "test_error":
                test_errors += 1
            elif status == "worker_error":
                worker_errors += 1

            detailed_results.append(
                {
                    "task_id": task_id,
                    "new_task_id": new_task_id,
                    "method_name": method_name,
                    **test_result,
                }
            )
            if offset % 25 == 0 or idx + 1 == end_index:
                global_pos = idx + 1
                print(
                    f"Processed {offset}/{segment_total} tasks (global {global_pos}/{total_rows})",
                    flush=True,
                )

    total = len(detailed_results)
    summary = {
        "total_tasks": total,
        "passed_tasks": passed,
        "failed_tasks": total
        - passed
        - code_errors
        - test_errors
        - timeouts
        - worker_errors,
        "code_errors": code_errors,
        "test_errors": test_errors,
        "timeouts": timeouts,
        "worker_errors": worker_errors,
        "pass_rate": passed / total if total else 0.0,
    }

    with detailed_result_path.open("w") as fh:
        json.dump(detailed_results, fh, indent=2)

    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
