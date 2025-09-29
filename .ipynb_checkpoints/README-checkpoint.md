
  

# Code Talkers Can’t Complete: Unveiling the Instruction-Tuning Tax of Large Language Models in Code Tasks

  

  

This repository contains the artifacts and experiments for the paper **“Code Talkers Can’t Complete: Unveiling the Instruction-Tuning Tax of Large Language Models in Code Tasks.”**

  

  

----------

  

  

## 🔧 Requirements

  

  

Clone the repository and install dependencies:

  

  

```bash

git  clone  https://github.com/arkosioscambions/CodeTalkers.git
pip  install  -r  requirements.txt

```

  

  

----------

  

  

## 🚀 Experiments

  

  

To generate model responses for evaluation, run:

  

  

```bash

python  generate.py  --model <model_name> --dataset <dataset_name>

```

  

  

### Examples

  

  

-  **SAFIM**

  

```bash

python  generate.py  --model  qwen  --model_id <model_id> --dataset  api.csv

```

  

-  **HumanEval-Infilling**

  

```bash
git clone https://github.com/openai/human-eval-infilling.git
cd human-eval-infilling
pip install -e .
python  ../generate.py  --model  dscoder  --model_id <model_id> --dataset  hei
```

  

-  **DS-1000 (Instruct)**

  

```bash

python  generate.py  --model  qwen  --model_id <model_id> --dataset  ds-1000.csv  --fewshot_file  fewshot_ds1000.json

```

> We run the fine-tuned model on Magicoder pipeline [Magicoder pipeline](https://chatgpt.com/c/68d03603-4258-8331-8eb7-d67a60171141#-magicoder-integration) to align with the chat template input.

  

-  **Other Benchmarks**

	-  **BigCodeBench** → Follow [BigCodeBench](https://github.com/bigcode-project/bigcodebench).
	-  **HumanEval(+) & MBPP(+)** → See [Magicoder experiments](https://github.com/ise-uiuc/magicoder/tree/main/experiments).

  

  

----------

  

  

## 🪄 Usage related to Magicoder

  

  

We build on [Magicoder](https://github.com/ise-uiuc/magicoder) for subset sampling of the dataset OSS-Instruct-75k and Evol-Instruct-110k, instruction tuning, and generating the response for questions from the Instruct version of DS-1000.

  

The modifications are applied to achieve **custom fine-tuned checkpoints, data mixing, and DS-1000 integration** in our studies, in which the implementation details will be guided below.

  

  

### Setup

  

  

```bash
bash  setup.sh
```

  

  

### Replicating Intermediate Models

  

  

1. Generate subset datasets:

  

```bash

python  sample_magicoder_subsets.py

```

  

2. Perform instruction tuning following [Magicoder README-DEV](https://github.com/ise-uiuc/magicoder/blob/main/README-DEV.md).

  

  

### Data Mixing Models

  

Similarly, perform instruction tuning following [Magicoder README-DEV](https://github.com/ise-uiuc/magicoder/blob/main/README-DEV.md) and change the `num_epoch` to 1.

  

Specifically, for sequential data mixing, update `datafile_paths` to `general_75k.jsonl` or `magicoder_oss_instruct_75k.jsonl` for whichever sequences needed (i.e., **Code-NL** or **NL-Code**), whereas for the **Mix** strategy, update `datafile_paths` with a merged JSONL of the two.

  
  

  

----------

  

  

## 📊 Evaluation

  

  

-  **HumanEval-FIM**

  

Follow [human-eval-infilling](https://github.com/openai/human-eval-infilling).

  

-  **SAFIM**

  

```bash

python  eval.py  --pred  <file_name.jsonl>  --gt  api.csv

```

  

-  **DS-1000**

  

Use the [official DS-1000 repository](https://github.com/xlang-ai/DS-1000).

  

We adapt their `test_ds1000.py` script for our `codex002-answers.jsonl`.

  

-  **BigCodeBench**

  

Follow the [official implementation](https://github.com/bigcode-project/bigcodebench).

  

-  **HumanEval(+) & MBPP(+)**

  

Refer to [Magicoder experiments](https://github.com/ise-uiuc/magicoder/tree/main/experiments).