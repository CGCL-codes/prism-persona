#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Selene-style single-score judging over three datasets with a local vLLM judge.

This script evaluates each (profile, context, response) example on a 1-5
consistency scale, then derives:
- auc               : absolute-score AUC
- pair_acc          : pair accuracy induced from single scores
- pair_auc          : pair AUC induced from single scores
- strict_group_acc  : strict group accuracy induced from single scores

Supported datasets:
1. Big5 hard
2. Big5 swap-within-trait
3. Self-awareness / SocialBench-style role consistency
"""

import argparse
import datetime
import json
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "selene_single"
DEFAULT_JUDGE = "AtlaAI/Selene-1-Mini-Llama-3.1-8B"

DATASET_SPECS = {
    "hard": {
        "display_name": "Big5-Persona-HARD",
        "input_file": PROJECT_ROOT / "data" / "big5_persona_hard.json",
        "context_key": "context",
        "context_tail_chars": None,
    },
    "swapwithintrait": {
        "display_name": "Big5-Persona-EASY",
        "input_file": PROJECT_ROOT / "data" / "big5_persona_easy.json",
        "context_key": "context",
        "context_tail_chars": None,
    },
    "self_awareness": {
        "display_name": "Social-Persona",
        "input_file": PROJECT_ROOT / "data" / "social_persona.json",
        "context_key": "dialogue",
        "context_tail_chars": 1200,
    },
}

RE_REASONING = re.compile(r"\*\*Reasoning:\*\*(.*?)(\*\*Result:\*\*|$)", re.IGNORECASE | re.DOTALL)
RE_RESULT_NUMERIC = re.compile(r"\*\*Result:\*\*\s*([1-5])\b", re.IGNORECASE)

DIRECT_JUDGE_SCALE_5 = {
    "1": "clearly inconsistent with the target profile or role",
    "2": "more inconsistent than consistent",
    "3": "mixed, borderline, or genuinely uncertain",
    "4": "more consistent than inconsistent",
    "5": "clearly consistent with the target profile or role",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_case_group_id(case_id):
    if not isinstance(case_id, str):
        return case_id
    if "_NEG" in case_id:
        case_id = case_id.split("_NEG", 1)[0]
    if "_neg_" in case_id:
        case_id = case_id.split("_neg_", 1)[0]
    if case_id.endswith("_pos"):
        case_id = case_id[: -len("_pos")]
    if case_id.endswith("_neg_swap"):
        case_id = case_id[: -len("_neg_swap")]
    return case_id


def normalize_row(dataset_name, row, spec):
    item = dict(row)
    context_key = spec["context_key"]
    item["dataset"] = dataset_name
    item["dataset_display_name"] = spec["display_name"]
    if "context" not in item:
        item["context"] = item.get(context_key, "")
    if "dialogue" not in item and context_key == "dialogue":
        item["dialogue"] = item.get("context", "")
    return item


def normalize_rows(dataset_name, rows, spec):
    return [normalize_row(dataset_name, row, spec) for row in rows]


def get_group_id(row):
    if row.get("label") == 0 and row.get("source_case_id"):
        return normalize_case_group_id(row["source_case_id"])
    case_id = normalize_case_group_id(row.get("case_id"))
    group_id = normalize_case_group_id(row.get("group_id"))
    return group_id or row.get("id") or case_id


def group_cases(rows):
    grouped = defaultdict(list)
    ordered_ids = []
    seen = set()
    positives = [r for r in rows if r.get("label") == 1]
    negatives = [r for r in rows if r.get("label") == 0]
    for row in positives:
        gid = get_group_id(row)
        if gid not in seen:
            seen.add(gid)
            ordered_ids.append(gid)
        grouped[gid].append(row)
    for row in negatives:
        gid = get_group_id(row)
        if gid in grouped:
            grouped[gid].append(row)
    return grouped, ordered_ids


def slice_grouped_cases(rows, start, end):
    grouped, ordered_ids = group_cases(rows)
    if end is None:
        end = len(ordered_ids)
    end = min(end, len(ordered_ids))
    selected_ids = ordered_ids[start:end]
    sliced = []
    for gid in selected_ids:
        sliced.extend(grouped[gid])
    return sliced, selected_ids


def truncate_tail(text, keep_chars):
    text = str(text or "").strip()
    if keep_chars is None or keep_chars <= 0 or len(text) <= keep_chars:
        return text
    return "[Earlier dialogue omitted]\n...\n" + text[-keep_chars:]


def render_scale_block(scale_dict):
    return "\n".join(f"- {k}: {v}" for k, v in scale_dict.items())


def infer_negative_type(row, dataset_name):
    neg_type = row.get("negative_type")
    if isinstance(neg_type, str) and neg_type:
        return neg_type
    option = row.get("option")
    if isinstance(option, str) and option:
        return f"option_{option}"
    if dataset_name == "swapwithintrait":
        return "swap_within_trait"
    return "negative"


def build_single_prompt(row, context_tail_chars=None):
    profile_block = row.get("profile", "")
    context = truncate_tail(row.get("dialogue") or row.get("context") or "", context_tail_chars)
    task = "Judge the overall consistency of the candidate response with the target profile or role in the given dialogue context."
    return f"""You are tasked with evaluating a response for persona consistency.

Your reply should strictly follow this format:
**Reasoning:** <brief feedback>

**Result:** <an integer between 1 and 5>

Task:
{task}

Use this scale:
{render_scale_block(DIRECT_JUDGE_SCALE_5)}

Instructions:
1. Evaluate whether the candidate response is consistent with the target profile in the given dialogue context.
2. Output exactly one final integer score from 1 to 5.
3. Do not explain.
4. Do not output any words, punctuation, or justification.


Target profile or role:
{profile_block}

Dialogue context:
{context}

Candidate response:
{row['response']}
"""


def parse_reasoning(text):
    match = RE_REASONING.search(text or "")
    if not match:
        return None
    reasoning = match.group(1).strip()
    return reasoning or None


def parse_single_result(text):
    match = RE_RESULT_NUMERIC.search(text or "")
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\b([1-5])\b", str(text or ""))
    if numbers:
        return int(numbers[-1])
    return None


def compute_scores_from_rating(rating, max_rating=5):
    score_01 = (float(rating) - 1.0) / float(max_rating - 1)
    midpoint = (max_rating + 1.0) / 2.0
    half_span = (max_rating - 1.0) / 2.0
    centered = (float(rating) - midpoint) / half_span
    return {
        "single_rating": float(rating),
        "single_score_01": score_01,
        "single_centered_score": centered,
    }


def rank_auc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y == 1 and isinstance(s, (int, float))]
    neg = [s for y, s in zip(labels, scores) if y == 0 and isinstance(s, (int, float))]
    if not pos or not neg:
        return None
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def pair_accuracy(results, score_key):
    grouped, _ = group_cases(results)
    wins = 0
    total = 0
    for _, rows in grouped.items():
        pos_scores = [r["scores"].get(score_key) for r in rows if r["label"] == 1]
        neg_scores = [r["scores"].get(score_key) for r in rows if r["label"] == 0]
        pos_scores = [x for x in pos_scores if isinstance(x, (int, float))]
        neg_scores = [x for x in neg_scores if isinstance(x, (int, float))]
        if not pos_scores or not neg_scores:
            continue
        p = max(pos_scores)
        for n in neg_scores:
            total += 1
            if p > n:
                wins += 1
    return (wins / total if total else None), total


def pair_auc(results, score_key):
    grouped, _ = group_cases(results)
    wins = 0.0
    total = 0
    for _, rows in grouped.items():
        pos_scores = [r["scores"].get(score_key) for r in rows if r["label"] == 1]
        neg_scores = [r["scores"].get(score_key) for r in rows if r["label"] == 0]
        pos_scores = [x for x in pos_scores if isinstance(x, (int, float))]
        neg_scores = [x for x in neg_scores if isinstance(x, (int, float))]
        if not pos_scores or not neg_scores:
            continue
        p = max(pos_scores)
        for n in neg_scores:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return (wins / total if total else None), total


def strict_group_accuracy(results, score_key):
    grouped, _ = group_cases(results)
    hits = 0
    total = 0
    for _, rows in grouped.items():
        pos_scores = [r["scores"].get(score_key) for r in rows if r["label"] == 1]
        neg_scores = [r["scores"].get(score_key) for r in rows if r["label"] == 0]
        pos_scores = [x for x in pos_scores if isinstance(x, (int, float))]
        neg_scores = [x for x in neg_scores if isinstance(x, (int, float))]
        if not pos_scores or not neg_scores:
            continue
        total += 1
        if min(pos_scores) > max(neg_scores):
            hits += 1
    return hits / total if total else None


def bucket_stats(bucket_rows, score_key):
    parsed_bucket = [r for r in bucket_rows if isinstance(r.get("scores", {}).get(score_key), (int, float))]
    labels_bucket = [r["label"] for r in parsed_bucket]
    vals_bucket = [r["scores"][score_key] for r in parsed_bucket]
    pair_acc_bucket, pair_n_bucket = pair_accuracy(parsed_bucket, score_key)
    pair_auc_bucket, _ = pair_auc(parsed_bucket, score_key)
    return {
        "n": len(bucket_rows),
        "n_parsed": len(parsed_bucket),
        "auc": rank_auc(labels_bucket, vals_bucket) if len(set(labels_bucket)) > 1 else None,
        "pair_acc": pair_acc_bucket,
        "pair_auc": pair_auc_bucket,
        "strict_group_acc": strict_group_accuracy(parsed_bucket, score_key),
        "pair_n": pair_n_bucket,
    }


def summarize(results, dataset_name, model_name, score_key="single_score_01"):
    valid = [r for r in results if r.get("parse_ok") and isinstance(r.get("scores", {}).get(score_key), (int, float))]
    labels = [r["label"] for r in valid]
    vals = [r["scores"][score_key] for r in valid]
    p_acc, pair_n = pair_accuracy(valid, score_key)
    p_auc, _ = pair_auc(valid, score_key)

    by_negative_type = defaultdict(list)
    by_trait = defaultdict(list)
    by_level = defaultdict(list)
    for row in results:
        by_negative_type[row.get("negative_type")].append(row)
        if row.get("trait") is not None:
            by_trait[row.get("trait")].append(row)
        if row.get("level") is not None:
            by_level[row.get("level")].append(row)

    return {
        "schema": "selene_single_vllm",
        "dataset": dataset_name,
        "model_name": model_name,
        "score_key": score_key,
        "n_total": len(results),
        "n_valid": len(valid),
        "n_parse_failed": len(results) - len(valid),
        "parse_rate": (len(valid) / len(results)) if results else None,
        "auc": rank_auc(labels, vals),
        "pair_acc": p_acc,
        "pair_auc": p_auc,
        "strict_group_acc": strict_group_accuracy(valid, score_key),
        "pair_n": pair_n,
        "by_negative_type": {str(k): bucket_stats(v, score_key) for k, v in sorted(by_negative_type.items(), key=lambda kv: str(kv[0]))},
        "by_trait": {str(k): bucket_stats(v, score_key) for k, v in sorted(by_trait.items(), key=lambda kv: str(kv[0]))},
        "by_level": {str(k): bucket_stats(v, score_key) for k, v in sorted(by_level.items(), key=lambda kv: str(kv[0]))},
    }


def run_single(llm, rows, sampling_params, batch_size, context_tail_chars):
    results = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [[{"role": "user", "content": build_single_prompt(row, context_tail_chars=context_tail_chars)}] for row in chunk]
        outputs = llm.chat(prompts, sampling_params=sampling_params)
        for row, output in zip(chunk, outputs):
            text = output.outputs[0].text
            rating = parse_single_result(text)
            parse_ok = rating is not None
            scores = compute_scores_from_rating(rating, max_rating=5) if parse_ok else {}
            results.append(
                {
                    **row,
                    "schema": "selene_single_vllm",
                    "judge_output": text,
                    "reasoning": parse_reasoning(text),
                    "parse_ok": parse_ok,
                    "scores": scores,
                    "negative_type": infer_negative_type(row, row["dataset"]),
                }
            )
    return results


def prepare_dataset_run(dataset_name, input_path, start, end):
    spec = DATASET_SPECS[dataset_name]
    rows = normalize_rows(dataset_name, load_json(input_path), spec)
    selected_rows, selected_group_ids = slice_grouped_cases(rows, start, end)
    return {
        "dataset": dataset_name,
        "display_name": spec["display_name"],
        "input_path": str(input_path),
        "selected_rows": selected_rows,
        "selected_group_ids": selected_group_ids,
        "context_tail_chars": spec["context_tail_chars"],
    }


def dataset_output_stem(dataset_name, start, end, timestamp):
    end_tag = "end" if end is None else str(end)
    return f"{dataset_name}_selene_single_{start}-{end_tag}_{timestamp}"


def main():
    parser = argparse.ArgumentParser(description="Run Selene single-score baseline on three datasets.")
    parser.add_argument("--dataset", choices=["hard", "swapwithintrait", "self_awareness", "all"], default="all")
    parser.add_argument("--input_file", type=str, default=None, help="Optional override for a single dataset run.")
    parser.add_argument("--start", type=int, default=0, help="Group start index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Group end index (exclusive).")
    parser.add_argument("--judge", type=str, default=DEFAULT_JUDGE)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    if args.dataset == "all" and args.input_file:
        raise ValueError("--input_file can only be used when --dataset is a single dataset.")

    import vllm
    from utils.seeds import initialize_seeds
    from vllm.sampling_params import SamplingParams

    initialize_seeds()
    dataset_names = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    llm = vllm.LLM(
        model=args.judge,
        enable_prefix_caching=True,
        dtype=args.dtype,
        tensor_parallel_size=args.gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_cascade_attn=True,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    console_runs = []
    for dataset_name in dataset_names:
        input_path = Path(args.input_file) if args.input_file else DATASET_SPECS[dataset_name]["input_file"]
        run = prepare_dataset_run(dataset_name, input_path, args.start, args.end)
        results = run_single(
            llm=llm,
            rows=run["selected_rows"],
            sampling_params=sampling_params,
            batch_size=args.batch_size,
            context_tail_chars=run["context_tail_chars"],
        )
        summary = summarize(results, dataset_name=dataset_name, model_name=args.judge)
        summary["config"] = {
            "dataset": dataset_name,
            "display_name": run["display_name"],
            "input_path": run["input_path"],
            "start": args.start,
            "end": args.end,
            "n_selected_rows": len(run["selected_rows"]),
            "n_selected_groups": len(run["selected_group_ids"]),
            "judge": args.judge,
            "gpus": args.gpus,
            "dtype": args.dtype,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "context_tail_chars": run["context_tail_chars"],
        }

        stem = dataset_output_stem(dataset_name, args.start, args.end, timestamp)
        results_path = out_dir / f"{stem}_results.jsonl"
        summary_path = out_dir / f"{stem}_summary.json"
        dump_jsonl(results, results_path)
        dump_json(summary, summary_path)

        console_runs.append(
            {
                "dataset": dataset_name,
                "display_name": run["display_name"],
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "auc": summary["auc"],
                "pair_acc": summary["pair_acc"],
                "pair_auc": summary["pair_auc"],
                "strict_group_acc": summary["strict_group_acc"],
                "parse_rate": summary["parse_rate"],
                "n_valid": summary["n_valid"],
                "n_total": summary["n_total"],
            }
        )

    print(json.dumps({"runs": console_runs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
