#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from datasets import load_dataset as hf_load_dataset

DATASET_ID_CANDIDATES = ("new_task_id", "task_id", "id")
PREDICTION_ID_CANDIDATES = ("task_id", "new_task_id", "id")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ClassEval-LineInfilling predictions with exact match."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="annachaaang/ClassEval-LineInfilling",
        help="Hugging Face dataset name for ClassEval-LineInfilling.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load from Hugging Face.",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="JSONL file containing predictions.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ClassEval-LineInfilling-results.csv"),
        help="CSV file to write per-sample exact-match results.",
    )
    parser.add_argument(
        "--truncate-at",
        type=str,
        default=None,
        help="Optional delimiter. If provided, text after its first occurrence is discarded before comparison.",
    )
    return parser.parse_args()


def normalize_text(value, truncate_at=None):
    text = "" if value is None else str(value).strip()
    if truncate_at:
        text = text.split(truncate_at, 1)[0].strip()
    return text


def resolve_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_ground_truth(dataset_name, split):
    dataset = hf_load_dataset(dataset_name, split=split)
    if "solution" not in dataset.column_names:
        raise ValueError("Dataset is missing the 'solution' column required for exact-match evaluation.")

    id_column = resolve_column(dataset.column_names, DATASET_ID_CANDIDATES)
    rows = []
    for index in range(len(dataset)):
        rows.append(
            {
                "sample": index + 1,
                "task_id": str(dataset[id_column][index]) if id_column else "",
                "ground_truth": normalize_text(dataset["solution"][index]),
            }
        )
    return rows, id_column


def load_predictions(jsonl_path, truncate_at):
    rows = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "completion" not in record:
                raise ValueError(f"Missing 'completion' field in {jsonl_path} at line {line_number}.")

            pred_id = ""
            for candidate in PREDICTION_ID_CANDIDATES:
                if candidate in record:
                    pred_id = str(record[candidate])
                    break

            rows.append(
                {
                    "task_id": pred_id,
                    "prediction": normalize_text(record["completion"], truncate_at=truncate_at),
                }
            )
    return rows


def compare_by_order(ground_truth_rows, prediction_rows):
    if len(ground_truth_rows) != len(prediction_rows):
        raise ValueError(
            f"Mismatch: GT={len(ground_truth_rows)}, Pred={len(prediction_rows)}. "
            "Provide task IDs in the prediction JSONL or ensure the rows are aligned."
        )

    results = []
    for gt_row, pred_row in zip(ground_truth_rows, prediction_rows):
        status = "PASS" if gt_row["ground_truth"] == pred_row["prediction"] else "FAIL"
        results.append(
            {
                "sample": gt_row["sample"],
                "task_id": gt_row["task_id"],
                "ground_truth": gt_row["ground_truth"],
                "prediction": pred_row["prediction"],
                "result": status,
            }
        )
    return results


def compare_by_task_id(ground_truth_rows, prediction_rows):
    prediction_map = {}
    for row in prediction_rows:
        task_id = row["task_id"]
        if not task_id:
            raise ValueError("Prediction rows must all contain task IDs for task-based matching.")
        if task_id in prediction_map:
            raise ValueError(f"Duplicate prediction task_id found: {task_id}")
        prediction_map[task_id] = row["prediction"]

    results = []
    for gt_row in ground_truth_rows:
        prediction = prediction_map.get(gt_row["task_id"], "")
        status = "PASS" if gt_row["ground_truth"] == prediction else "FAIL"
        results.append(
            {
                "sample": gt_row["sample"],
                "task_id": gt_row["task_id"],
                "ground_truth": gt_row["ground_truth"],
                "prediction": prediction,
                "result": status,
            }
        )
    return results


def write_results_csv(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample", "task_id", "ground_truth", "prediction", "result"]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    ground_truth_rows, dataset_id_column = load_ground_truth(args.dataset, args.split)
    prediction_rows = load_predictions(args.pred, args.truncate_at)

    prediction_id_count = sum(1 for row in prediction_rows if row["task_id"])
    if 0 < prediction_id_count < len(prediction_rows):
        raise ValueError("Prediction JSONL contains task IDs for only some rows; use all or none.")

    use_task_ids = bool(dataset_id_column) and prediction_id_count == len(prediction_rows)
    if use_task_ids:
        results = compare_by_task_id(ground_truth_rows, prediction_rows)
        comparison_mode = "task_id"
    else:
        results = compare_by_order(ground_truth_rows, prediction_rows)
        comparison_mode = "order"

    write_results_csv(args.output_csv, results)

    total = len(results)
    passed = sum(1 for row in results if row["result"] == "PASS")
    exact_match = passed / total if total else 0.0

    print(f"GT samples: {len(ground_truth_rows)}")
    print(f"Prediction samples: {len(prediction_rows)}")
    print(f"Comparison mode: {comparison_mode}")
    print(f"Exact Match: {exact_match:.4%}")
    print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
