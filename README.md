# PRISM

Code and data for **"Do LLMs Understand Personality? Rethinking Persona Fidelity Evaluation through Structured Behavioral Inference"**, accepted to **EMNLP 2026 Main Conference**.

**Authors:** Mengfan Li, Zesheng Wei, Xuanhua Shi, Yang Deng

PRISM is a persona-fidelity evaluation method for persona-grounded dialogue responses.

PRISM estimates whether a candidate response is consistent with a target profile by probing a language model's latent evidence over three profile-consistency dimensions:

- `Task`: agenda, priorities, goals, or action pattern expressed by the response.
- `Stance`: interpersonal and affective posture toward the interlocutor.
- `Style`: wording, register, rhythm, and surface realization.

The repository also includes scripts for several baseline evaluators used in the paper.

## Repository Layout

```text
PRISM/
  data/
    big5_persona_easy.json
    big5_persona_hard.json
    social_persona.json
  scripts/
    evaluate/
      PRISM_big5.py
      PRISM_socialpersona.py
    baselines/
      selene_single.py
      alignscore_single.py
      pandalm_pairwise.py
    direct_LLM_as_a_judge.py
  requirements.txt
  README.md
```

## Datasets

The released data files are JSON lists. Each row contains a target profile, dialogue context, candidate response, binary label, and grouping metadata for pairwise/group-level evaluation.

- `data/big5_persona_easy.json`: Big5-Persona-EASY.
- `data/big5_persona_hard.json`: Big5-Persona-HARD.
- `data/social_persona.json`: Social-Persona.

The main fields are:

- `profile`: target persona/profile description.
- `context` or `dialogue`: dialogue context before the candidate response.
- `response`: candidate response to evaluate.
- `label`: `1` for profile-consistent response, `0` for inconsistent response.
- `case_id` / `group_id` / `source_case_id`: metadata used to group positive and negative variants of the same case.

## Installation

We recommend using a fresh Python environment.

```bash
conda create -n prism python=3.10 -y
conda activate prism
pip install -r requirements.txt
```

For gated Hugging Face models such as Llama, make sure you have access and are logged in:

```bash
huggingface-cli login
```

## Model Weights

This repository does not include model weights. When a script is run with a Hugging Face model name, for example `Qwen/Qwen2.5-7B-Instruct`, `transformers` or `vLLM` will automatically download the model to the local Hugging Face cache if network access is available.

For gated or restricted models, such as some Llama checkpoints, users must first accept the model license and authenticate with Hugging Face. If models have already been downloaded locally, pass the local checkpoint directory with `--model_name` or the corresponding baseline argument such as `--judge`.

## Running PRISM

Run commands from the repository root, i.e. the `PRISM/` directory.

### Big5-Persona-EASY

```bash
python scripts/evaluate/PRISM_big5.py \
  --dataset easy \
  --cases_file data/big5_persona_easy.json \
  --cases_tag big5_persona_easy \
  --model_preset qwen25 \
  --out_dir outputs/prism_big5
```

### Big5-Persona-HARD

```bash
python scripts/evaluate/PRISM_big5.py \
  --dataset hard \
  --cases_file data/big5_persona_hard.json \
  --cases_tag big5_persona_hard \
  --model_preset qwen25 \
  --out_dir outputs/prism_big5
```

### Social-Persona

```bash
python scripts/evaluate/PRISM_socialpersona.py \
  --cases_file data/social_persona.json \
  --cases_tag social_persona \
  --model_preset qwen25 \
  --out_dir outputs/prism_socialpersona
```

Available model presets are:

- `qwen25`: `Qwen/Qwen2.5-7B-Instruct`
- `qwen25_14b`: `Qwen/Qwen2.5-14B-Instruct`
- `llama3`: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `mistral`: `mistralai/Mistral-7B-Instruct-v0.3`

You can also pass a Hugging Face model name or local model path directly:

```bash
python scripts/evaluate/PRISM_big5.py \
  --dataset easy \
  --cases_file data/big5_persona_easy.json \
  --model_name Qwen/Qwen2.5-7B-Instruct
```

## Outputs

PRISM writes two files per run:

- `*_results.json`: per-example predictions, prompts, label mappings, and scores.
- `*_analysis.json`: aggregate metrics and run configuration.

The main reported metrics are:

- `auc`: standard AUC over all positive/negative examples.
- `pair_auc`: pairwise AUC comparing positive and negative variants within the same case.
- `strict_group_acc`: group accuracy requiring the positive response to score above all negatives in the same group.

## Baselines

### Selene Single-Score Baseline

This baseline uses vLLM with `AtlaAI/Selene-1-Mini-Llama-3.1-8B`.

```bash
python scripts/baselines/selene_single.py \
  --dataset hard \
  --input_file data/big5_persona_hard.json \
  --out_dir outputs/selene_single
```

To run all supported datasets:

```bash
python scripts/baselines/selene_single.py \
  --dataset all \
  --out_dir outputs/selene_single
```

### PandaLM Pairwise Baseline

This baseline adapts PandaLM to a pairwise profile-consistency task. Install or place PandaLM under:

```text
scripts/baselines/PandaLM/
```

Then run:

```bash
python scripts/baselines/pandalm_pairwise.py \
  --dataset hard \
  --input_file data/big5_persona_hard.json \
  --out_dir outputs/pandalm_pairwise
```

PandaLM is used as a pairwise judge, so it reports pairwise metrics such as `pair_auc` and `strict_group_acc`; standard score-based AUC is not reported for this baseline.

### AlignScore Single-Score Baseline

Install or place AlignScore under:

```text
scripts/baselines/AlignScore/
```

Then provide the AlignScore checkpoint path:

```bash
python scripts/baselines/alignscore_single.py \
  --dataset hard \
  --input_file data/big5_persona_hard.json \
  --ckpt_path /path/to/alignscore/checkpoint.ckpt \
  --out_dir outputs/alignscore_single
```

### API-Based Direct LLM-as-a-Judge

The API judge script reads credentials from environment variables or command-line arguments. Do not hard-code private API keys in the repository.

```bash
export OPENROUTER_API_KEY="your_api_key"
export OPENROUTER_API_BASE="your_api_base"

python scripts/direct_LLM_as_a_judge.py \
  --cases_preset big5_hard \
  --cases_file data/big5_persona_hard.json \
  --model_name google/gemini-3-flash-preview \
  --out_dir outputs/api_judge
```

## Notes

- The scripts support `--start` and `--end` for running subsets by grouped case index.
- Output filenames include dataset/model/range/timestamp information.

## Citation

BibTeX will be added once the official proceedings version is available.

If you use this repository before then, please cite the EMNLP 2026 paper:

**Do LLMs Understand Personality? Rethinking Persona Fidelity Evaluation through Structured Behavioral Inference.**
