# Lost in the Flow with Code Talkers: Unveiling the Instruction-Tuning Tax of Large Language Models in Code Tasks

This repository contains the artifacts and experiments for the paper **“Lost in the Flow with Code Talkers: Unveiling the Instruction-Tuning Tax of Large Language Models in Code Tasks.”**

## Requirements

Clone the repository and install dependencies:

```bash
git clone https://github.com/arkosioscambions/CodeTalkers.git
cd CodeTalkers
pip install -r requirements.txt
```

## Reproduction Overview

`RQ1` and `RQ3` use the same benchmark generation and evaluation commands.

- `RQ1` evaluates pretrained base and instruction-tuned models.
- `RQ3` evaluates fine-tuned checkpoints produced from `Qwen2.5-Coder-7B` using the Magicoder pipeline.
- `RQ2` is a separate behavioral analysis over generated outputs.

## Common Benchmark Workflow for RQ1 and RQ3

### 1. Generate model outputs

Use the same generation command for both `RQ1` and `RQ3`:

```bash
python generate.py --model <qwen|dscoder> --model_id <model_id> --dataset <dataset_name>
```

`--model_id` can be a Hugging Face model ID or a local checkpoint path.

### 2. Example generation commands

**HumanEval-Infilling**

```bash
git clone https://github.com/openai/human-eval-infilling.git
cd human-eval-infilling
pip install -e .
python ../generate.py --model <qwen|dscoder> --model_id <model_id> --dataset hei
```

**ClassEval-LineInfilling**

Dataset: [annachaaang/ClassEval-LineInfilling](https://huggingface.co/datasets/annachaaang/ClassEval-LineInfilling)

```bash
python generate.py --model <qwen|dscoder> --model_id <model_id> --dataset annachaaang/ClassEval-LineInfilling
```

**ClassEval-Completion**

Dataset: [annachaaang/ClassEval-Completion](https://huggingface.co/datasets/annachaaang/ClassEval-Completion)

```bash
python generate.py --model <qwen|dscoder> --model_id <model_id> --dataset annachaaang/ClassEval-Completion
```

**DS-1000 (Instruct)**

```bash
python generate.py \
  --model <qwen|dscoder> \
  --model_id <model_id> \
  --dataset ds-1000.csv \
  --fewshot_file fewshot_ds1000.json
```

**Other benchmarks**

- `SAFIM`: follow [SAFIM](https://github.com/gonglinyuan/safim)
- `BigCodeBench`: follow [BigCodeBench](https://github.com/bigcode-project/bigcodebench)
- `HumanEval(+)` and `MBPP(+)`: follow [Magicoder experiments](https://github.com/ise-uiuc/magicoder/tree/main/experiments)

### 3. Evaluate generated outputs

**HumanEval-FIM**

Follow [human-eval-infilling](https://github.com/openai/human-eval-infilling).

**ClassEval-Completion**

```bash
python evaluate_classeval_completion.py \
  --completions filename.jsonl \
  --output-dir ClassEval/output/ \
  --per-task-timeout 5
```

**ClassEval-LineInfilling**

```bash
python evaluate_classeval_lineinfilling.py \
  --pred filename.jsonl \
  --output-csv ClassEval-LineInfilling-results.csv
```

Use `--truncate-at "//"` if you want to ignore trailing comments before comparison.

**Other benchmark evaluators**

- `SAFIM`: follow [SAFIM](https://github.com/gonglinyuan/safim)
- `DS-1000`: use the [official DS-1000 repository](https://github.com/xlang-ai/DS-1000)
- `BigCodeBench`: use the [official implementation](https://github.com/bigcode-project/bigcodebench)
- `HumanEval(+)` and `MBPP(+)`: use [Magicoder experiments](https://github.com/ise-uiuc/magicoder/tree/main/experiments)

## RQ1: Pretrained and Instruction-Tuned Models

`RQ1` uses the common workflow above with pretrained base and instruction-tuned models.

The experiments in this repository cover:

- `Qwen2.5-Coder`: `1.5B`, `7B`, `14B`, `32B`
- `DeepSeek-Coder`: `1.3B`, `6.7B`, `33B`

Use the corresponding base or instruct model as `--model_id`, then reuse the common generation and evaluation commands.

## RQ3: Fine-Tuned Qwen2.5-Coder-7B Checkpoints

`RQ3` reuses the same benchmark workflow as `RQ1`. The only difference is the model checkpoint passed to `--model_id`.

In `RQ3`, the evaluated model is a fine-tuned `Qwen2.5-Coder-7B` checkpoint produced with the Magicoder pipeline.

### Setup

```bash
bash setup.sh
```

### Replicate intermediate checkpoints

1. Generate subset datasets:

```bash
python sample_magicoder_subsets.py
```

2. Perform instruction tuning following [Magicoder README-DEV](https://github.com/ise-uiuc/magicoder/blob/main/README-DEV.md).

### Evaluate fine-tuned checkpoints

After the checkpoints are produced, reuse the common benchmark workflow and replace `--model_id` with the checkpoint path.

Example:

```bash
python generate.py \
  --model qwen \
  --model_id /path/to/magicoder-checkpoint \
  --dataset annachaaang/ClassEval-Completion
```

Then evaluate the generated file with the same evaluation command used in `RQ1`.

## RQ2: Behavioral Metrics (Table 7)

Generate the behavioral metrics reported in Table 7 as CSV and JSON:

```bash
python3 generate_rq2_table7.py
```

This writes:

```text
results/rq2_table7.csv
results/rq2_table7.json
```

The script expects the raw benchmark artifacts listed in `MODEL_SPECS` inside `generate_rq2_table7.py`.

If your local artifact layout is different, edit `MODEL_SPECS` in [generate_rq2_table7.py](generate_rq2_table7.py).
