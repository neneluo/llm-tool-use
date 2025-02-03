# llm-tool-use
[NAACL'25 Findings] Analysis of Augmenting Large Language Models with Tools in Zero-Shot

## Environment
The details of dependencies are in `environment.yml`

## Usage

### QA datasets collection
Downloading dataset from HuggingFace and sampling subsets for experiments:
```
mkdir -p data
python toolusellm/generate_dataset.py
```

### Data generation & filtering

To generate training data for Supervised Fine-Tuning or Preference Fine-Tuning, below is an example of dataset `TriviaQA`:

1. Data generation: specify dataset name, model name, and subset in `prompt.sh`, conduct model inference on the training set of `TriviaQA`, and then save the output JSON file to `results/triviaqa-subset-train.jsonl`;
```
sh prompt.sh
```
2. Data filtering: filter out "correct" tool-using traces with specified metric

```
python toolusellm/prepare_data.py \
    --input_json results/triviaqa-subset-train.jsonl \
    --output_json training_data/sft.triviaqa.train.acc.jsonl \
    --data_type sft \
    --metric acc \
    --dataset triviaqa
```

### Model inference:
```
sh prompt.sh
```

### Model training
Supervised Fine-Tuning experiments:
```
sh sft.sh
```

Preference Fine-Tuning experiments:
```
sh pft.sh
```

Note: before running the shell script, specify the variables (e.g., model name, dataset name, etc) in the scripts accordingly.

### Model evaluation
Compute Exact Match and Accuracy:
```
python evaluation/compute_score.py \
    --json ${result_jsonl} \
    --dataset ${dataset}
```

Compute Invoke Rate, Pass Rate and Answerable Rate:
```
python evaluation/compute_rate.py \
    --json ${result_jsonl}
```
Note: specify the `${result_jsonl}` and `${dataset}` as needed.

## Acknowledgement
This repo's model inference and training codes are supported by HuggingFace `trl`,  `transformers`, and `peft`.
The evaluation implementation of the repo incorporated codes from `mandarjoshi90/triviaqa`, `nelson-liu/lost-in-the-middle`, `EleutherAI/lm-evaluation-harness`. 
The tool's implementation of the repo adapted codes from `ernie-research/Tool-Augmented-Reward-Model`, `lucidrains/toolformer-pytorch`.

A heartfelt thank you to the authors and contributors of these projects for their invaluable work and open-source contributions! 
