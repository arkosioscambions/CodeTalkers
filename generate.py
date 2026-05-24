# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import ast
import gzip
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def write_jsonl(path, data):
    """Write a list of dicts to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def append_jsonl(path, data):
    """Append a list of dicts to JSONL."""
    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_completion(text: str) -> str:
    """
    Post-process raw model completion for code-only output (for *completion* datasets, not single-line infilling).

    - Cut off at model-specific separators (Qwen, FIM tokens, markdown fences).
    - Strip whitespace.
    """
    stop_markers = [
        "<|file_sep|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "```",          # in case model wraps code in fences
        "# Qwen",       # common README start
    ]
    cut_positions = []
    for m in stop_markers:
        idx = text.find(m)
        if idx != -1:
            cut_positions.append(idx)
    if cut_positions:
        text = text[: min(cut_positions)]
    return text.strip()


def extract_fim_middle(decoded: str, prompt: str, suffix: str) -> str:
    """
    Extract the text that belongs in the infill hole by cutting:
      decoded = <prompt> + <MIDDLE> + <suffix> + (maybe more junk)
    """
    suffix = suffix or ""

    p = decoded.find(prompt)
    start = p + len(prompt) if p != -1 else 0

    if suffix:
        end = decoded.find(suffix, start)
        if end != -1:
            return decoded[start:end].strip()

    return decoded[start:].strip()


def first_non_empty_line(s: str) -> str:
    """
    Return the first line that is not empty / not whitespace-only.
    Preserve indentation and content exactly.
    """
    if s is None or s == "":
        return ""
    for line in s.splitlines():
        if line.strip() != "":
            return line
    return ""


# === Dataset loaders ===
def load_dataset_hefim(name: str, data_dir="../human-eval-infilling/data"):
    if name.lower() == "hei":
        filename = os.path.join(data_dir, "HumanEval-SingleLineInfilling.jsonl.gz")
        problems = {}
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                task = json.loads(line)
                problems[task["task_id"]] = task
        return [
            dict(task_id=tid, prompt=problems[tid]["prompt"], suffix=problems[tid]["suffix"])
            for tid in problems
        ]
    raise ValueError(f"Unknown HEFIM dataset: {name}")


def load_dataset_ds1000(name: str):
    return pd.read_csv(name)


def load_fewshot_examples(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_classeval_lineinfilling(
    hf_name: str = "annachaaang/ClassEval-LineInfilling",
    split: str = "test",
):
    ds = load_dataset(hf_name, split=split)
    prompts = []
    for ex in ds:
        prefix = ex.get("prefix")
        suffix = ex.get("suffix")
        if prefix is None or suffix is None:
            raise ValueError(f"Missing prefix/suffix in HF example: {ex}")
        prompts.append(
            dict(
                task_id=ex["new_task_id"],
                prompt=prefix,
                suffix=suffix,
            )
        )
    return prompts


def load_classeval_completion(
    hf_name: str = "annachaaang/ClassEval-Completion",
    split: str = "test",
):
    ds = load_dataset(hf_name, split=split)
    prompts = []
    for ex in ds:
        code_prefix = ex.get("completion_code")
        if code_prefix is None:
            raise ValueError(f"Missing completion_code in HF example: {ex}")
        prompts.append(
            dict(
                task_id=ex["new_task_id"],
                prompt=code_prefix,
            )
        )
    return prompts


# === Model generator functions ===
def get_model_and_tokenizer(
    model_name: str,
    model_id: str,
    dataset_type: str,
    few_shot_examples=None,
):
    model_name_l = model_name.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Ensure pad_token exists
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False

    single_line_infilling = dataset_type in ["hei", "classeval_fim"]

    # === FIM / infilling datasets ===
    if dataset_type in ["hei", "classeval_fim"]:
        if model_name_l == "qwen":
            def generate(prompt, suffix):
                input_text = f"<|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>"
                inputs = tokenizer([input_text], return_tensors="pt").to(device)
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )[0]
                out = tokenizer.decode(
                    outputs[len(inputs.input_ids[0]):],
                    skip_special_tokens=True,
                )
                return first_non_empty_line(out) if single_line_infilling else out

        elif model_name_l == "dscoder":
            def generate(prompt, suffix):
                input_text = f"<｜fim▁begin｜>{prompt}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"
                inputs = tokenizer([input_text], return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                out = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):]
                return first_non_empty_line(out) if single_line_infilling else out

        else:
            raise ValueError(f"Unknown model: {model_name}")

    # === DS-1000 ===
    elif dataset_type == "ds1000":
        shots = ""
        if few_shot_examples:
            for ex in few_shot_examples:
                shots += ex["problem"].rstrip() + ex["solution"].strip()

        if model_name_l in ["qwen", "dscoder"]:
            def generate(prompt, _suffix=None):
                user_prompt = shots + prompt

                # Prefer chat template if present (Qwen/DSCoder)
                if hasattr(tokenizer, "apply_chat_template") and model_name_l in ["qwen", "dscoder"]:
                    messages = [{"role": "user", "content": user_prompt}]
                    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                else:
                    text = user_prompt

                inputs = tokenizer([text], return_tensors="pt").to(device)

                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated_ids = [out[len(inp):] for out, inp in zip(outputs, inputs.input_ids)]
                return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            raise ValueError(f"Unknown model for DS-1000: {model_name}")

    # === ClassEval plain completion ===
    elif dataset_type == "classeval_completion":
        if model_name_l in ["qwen", "dscoder"]:
            def generate(prompt, _suffix=None):
                if model_name_l == "qwen":
                    prompt_with_hint = (
                        prompt.rstrip()
                        + "\n# NOTE: Continue the function above. Return ONLY valid and minimal Python code. Do not add explanations, tests, or extra files.\n"
                    )
                else:
                    prompt_with_hint = prompt.rstrip() + "\n# The implementation is as follows:\n"
                inputs = tokenizer([prompt_with_hint], return_tensors="pt").to(device)

                generate_kwargs = {
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                }

                if model_name_l == "qwen":
                    eos_ids = []
                    if tokenizer.eos_token_id is not None:
                        eos_ids.append(tokenizer.eos_token_id)

                    special_stops = ["<|file_sep|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
                    vocab = tokenizer.get_vocab()
                    for tok in special_stops:
                        if tok in vocab:
                            eos_ids.append(vocab[tok])

                    generate_kwargs.update(
                        temperature=0.25,
                        top_p=0.9,
                        top_k=40,
                        repetition_penalty=1.1,
                        eos_token_id=eos_ids if eos_ids else None,
                    )
                else:
                    generate_kwargs.update(temperature=0.1)

                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generate_kwargs,
                )

                generated_ids = [out[len(inp):] for out, inp in zip(outputs, inputs.input_ids)]
                raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return clean_completion(raw)
        else:
            raise ValueError(f"Unknown model for ClassEval-Completion: {model_name}")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return generate


# === Main function ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name: qwen | dscoder")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset: 'hei', a DS-1000 CSV file, or an HF dataset name like 'annachaaang/ClassEval-LineInfilling'",
    )
    parser.add_argument("--fewshot_file", type=str, default=None, help="JSON file containing few-shot examples for DS-1000")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per task")

    # Subset control
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated task_ids to run only (e.g. tid1,tid2,tid3). If omitted, run all tasks.",
    )
    args = parser.parse_args()

    # Detect dataset type
    if args.dataset.lower() == "hei":
        dataset_type = "hei"
        prompts = load_dataset_hefim(args.dataset)
        # target_ids = {
        #     "SingleLineInfilling/HumanEval/40/L4",
        #     "SingleLineInfilling/HumanEval/46/L6",
        #     "SingleLineInfilling/HumanEval/59/L11",
        #     "SingleLineInfilling/HumanEval/60/L0",
        #     "SingleLineInfilling/HumanEval/61/L7",
        # }
        
        # prompts = [p for p in prompts if p["task_id"] in target_ids]
        # print(f"Filtered to {len(prompts)} tasks:", [p["task_id"] for p in prompts])

    elif args.dataset.lower().endswith(".csv") and "ds-1000" in args.dataset.lower():
        df = load_dataset_ds1000(args.dataset)
        dataset_type = "ds1000"

    elif "classeval-lineinfilling" in args.dataset.lower() or "annachaaang/classeval-lineinfilling" in args.dataset.lower():
        dataset_type = "classeval_fim"
        hf_name = args.dataset if "/" in args.dataset else "annachaaang/ClassEval-LineInfilling"
        prompts = load_classeval_lineinfilling(hf_name=hf_name, split="test")

    elif "classeval-completion" in args.dataset.lower() or "annachaaang/classeval-completion" in args.dataset.lower():
        dataset_type = "classeval_completion"
        hf_name = args.dataset if "/" in args.dataset else "annachaaang/ClassEval-Completion"
        prompts = load_classeval_completion(hf_name=hf_name, split="test")

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Optional filtering to specific task_ids
    if args.task_ids:
        target_ids = {x.strip() for x in args.task_ids.split(",") if x.strip()}
        prompts = [p for p in prompts if p["task_id"] in target_ids]
        print(f"Filtered to {len(prompts)} tasks:", [p["task_id"] for p in prompts])

    few_shot_examples = load_fewshot_examples(args.fewshot_file) if args.fewshot_file else None
    generate_fn = get_model_and_tokenizer(
        args.model,
        args.model_id,
        dataset_type,
        few_shot_examples=few_shot_examples,
    )

    samples = []
    if dataset_type in ["hei", "classeval_fim", "classeval_completion"]:
        for p in tqdm(prompts, desc="Generating samples", leave=False):
            for _ in range(args.num_samples):
                answer = generate_fn(p["prompt"], p.get("suffix"))
                samples.append({"task_id": p["task_id"], "completion": answer})
                print(samples[-1])

        model_name = args.model_id.split("/")[-1]
        if dataset_type == "classeval_fim":
            output_file = f"{model_name}-ClassEval-LineInfilling.jsonl"
        elif dataset_type == "classeval_completion":
            output_file = f"{model_name}-ClassEval-Completion-0.25-1st.jsonl"
        else:
            output_file = f"{model_name}_{args.dataset.replace('.csv','')}.jsonl"
    
        write_jsonl(output_file, samples)
        print(f"✅ Saved results to {output_file}")

    else:  # ds1000
        model_name = args.model_id.split("/")[-1]
        output_file = f"{model_name}_{args.dataset.replace('.csv','')}_instruct.jsonl"
    
        buffer = []
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            prompt = row["prompt"]
            metadata = ast.literal_eval(row["metadata"])
            try:
                insert = generate_fn(prompt)
                buffer.append({"code": insert + "\n", "metadata": metadata})
            except Exception as e:
                print(f"⚠️ Error on prompt: {prompt[:50]}... -> {e}")
    
            # 🔹 Save every 10 samples
            if len(buffer) >= 10:
                append_jsonl(output_file, buffer)
                buffer.clear()
    
        # 🔹 Save any remaining samples
        if buffer:
            append_jsonl(output_file, buffer)



if __name__ == "__main__":
    main()
