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
from human_eval_infilling.data import write_jsonl, read_problems

# === Dataset loaders ===
def load_data_safim(name: str):
    df = pd.read_csv(f"{name}.csv")
    task_id = list(df["task_id"])
    prompt = [x.split("{{completion}}") for x in list(df["eval_prompt"])]
    return [
        dict(task_id=task_id[i], prompt=prompt[i][0], suffix=prompt[i][1])
        for i in range(len(task_id))
    ]

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
    else:
        raise ValueError(f"Unknown HEFIM dataset: {name}")


def load_dataset_ds1000(name: str):
    df = pd.read_csv(name)
    return df

def load_fewshot_examples(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Model generator functions ===
def get_model_and_tokenizer(model_name: str, model_id: str, dataset_type: str, few_shot_examples=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto").eval()

    if dataset_type in ["safim", "hei"]:
        if model_name.lower() == "qwen":
            def generate(prompt, suffix):
                input_text = f"<|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>"
                inputs = tokenizer([input_text], return_tensors="pt").to(device)
                outputs = model.generate(inputs.input_ids, max_new_tokens=512, temperature=0.0)[0]
                return tokenizer.decode(outputs[len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        elif model_name.lower() == "dscoder":
            def generate(prompt, suffix):
                input_text = f"<｜fim▁begin｜>{prompt}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"
                inputs = tokenizer([input_text], return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.0)
                return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):]
        else:
            raise ValueError(f"Unknown model: {model_name}")
    elif dataset_type == "ds1000":
        # Prepare few-shot concatenation
        shots = ""
        if few_shot_examples:
            for ex in few_shot_examples:
                shots += ex["problem"].rstrip() + ex["solution"].strip()

        if model_name.lower() in ["qwen", "dscoder"]:
            def generate(prompt, _suffix=None):
                user_prompt = shots + prompt
                messages = [{"role": "user", "content": user_prompt}]
                # Apply chat template if tokenizer supports it
                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                else:
                    text = user_prompt
                inputs = tokenizer([text], return_tensors="pt").to(device)
                outputs = model.generate(inputs.input_ids, max_new_tokens=512, temperature=0.0)
                generated_ids = [out[len(inp):] for out, inp in zip(outputs, inputs.input_ids)]
                return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            raise ValueError(f"Unknown model for DS-1000: {model_name}")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return generate

# === Main function ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name: qwen | dscoder")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset file (CSV) or 'hei'")
    parser.add_argument("--fewshot_file", type=str, default=None, help="JSON file containing few-shot examples for DS-1000")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per task")
    args = parser.parse_args()

    # Detect dataset type
    if args.dataset.lower() == "hei":
        dataset_type = "hei"
        prompts = load_dataset_hefim(args.dataset)
    elif args.dataset.lower().endswith(".csv"):
        df = load_dataset_ds1000(args.dataset)
        dataset_type = "ds1000" if "ds-1000" in args.dataset.lower() else "safim"
        if dataset_type == "safim":
            prompts = load_data_safim(args.dataset.replace(".csv", ""))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load few-shot examples if provided
    few_shot_examples = load_fewshot_examples(args.fewshot_file) if args.fewshot_file else None

    generate_fn = get_model_and_tokenizer(args.model, args.model_id, dataset_type, few_shot_examples)

    samples = []
    if dataset_type in ["safim", "hei"]:
        for p in tqdm(prompts, desc="Generating samples", leave=False):
            for _ in range(args.num_samples):
                answer = generate_fn(p["prompt"], p.get("suffix"))
                samples.append({"task_id": p["task_id"], "completion": answer})
                print(samples[-1])
        output_file = f"{args.model}_{args.dataset.replace('.csv','')}.jsonl"
    else:  # ds1000
        for _, row in tqdm(df.iterrows(), total=len(df)):
            prompt = row["prompt"]
            metadata = ast.literal_eval(row["metadata"])
            try:
                insert = generate_fn(prompt)
                samples.append({"code": insert + "\n", "metadata": metadata})
            except Exception as e:
                print(f"⚠️ Error on prompt: {prompt[:50]}... -> {e}")
        output_file = f"{args.model}_{args.dataset.replace('.csv','')}_instruct.jsonl"

    write_jsonl(output_file, samples)
    print(f"✅ Saved results to {output_file}")

if __name__ == "__main__":
    main()
