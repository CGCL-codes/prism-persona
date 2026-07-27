#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run PandaLM as an auxiliary pairwise judge baseline on three datasets:
1. Big5 hard
2. Big5 swap-within-trait
3. Self-awareness / SocialBench-style role consistency

This script adapts PandaLM to a profile-consistency pairwise task and reports:
- pair_acc
- pair_auc (P-AUC)
- strict_group_acc (G-ACC)

Absolute-score AUC is not reported because PandaLM is used here strictly as a
pairwise evaluator.
"""

import argparse
import datetime
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANDALM_ROOT = SCRIPT_DIR / "PandaLM"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "pandalm_pairwise"
DEFAULT_JUDGE = "WeOpenML/PandaLM-7B-v1"

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


def choose_candidate_order(pair_index, strategy, rng):
    if strategy == "positive_a":
        return True
    if strategy == "positive_b":
        return False
    if strategy == "alternate":
        return pair_index % 2 == 0
    return rng.random() < 0.5


def build_pairwise_rows(dataset_name, rows, selected_group_ids, position_strategy, seed):
    grouped, _ = group_cases(rows)
    rng = random.Random(seed)
    pair_rows = []
    pair_index = 0

    for gid in selected_group_ids:
        group_rows = grouped[gid]
        pos_rows = [r for r in group_rows if r.get("label") == 1]
        neg_rows = [r for r in group_rows if r.get("label") == 0]
        if not pos_rows or not neg_rows:
            continue

        pos = pos_rows[0]
        context = pos.get("context") or pos.get("dialogue") or ""
        profile = pos.get("profile", "")

        for neg in neg_rows:
            positive_is_a = choose_candidate_order(pair_index, position_strategy, rng)
            negative_type = infer_negative_type(neg, dataset_name)
            negative_option = neg.get("option")

            if positive_is_a:
                candidate_a = pos["response"]
                candidate_b = neg["response"]
                candidate_a_case_id = pos.get("case_id")
                candidate_b_case_id = neg.get("case_id")
                positive_candidate = "A"
                negative_candidate = "B"
            else:
                candidate_a = neg["response"]
                candidate_b = pos["response"]
                candidate_a_case_id = neg.get("case_id")
                candidate_b_case_id = pos.get("case_id")
                positive_candidate = "B"
                negative_candidate = "A"

            pair_rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_display_name": pos.get("dataset_display_name"),
                    "group_id": gid,
                    "pair_index": pair_index,
                    "pair_id": f"{gid}__PAIR__{negative_type}__{neg.get('case_id')}",
                    "source_case_id": pos.get("case_id"),
                    "original_index": pos.get("original_index"),
                    "trait": pos.get("trait"),
                    "level": pos.get("level"),
                    "profile": profile,
                    "context": context,
                    "dialogue": pos.get("dialogue", context),
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "candidate_a_case_id": candidate_a_case_id,
                    "candidate_b_case_id": candidate_b_case_id,
                    "candidate_a_label": 1 if positive_is_a else 0,
                    "candidate_b_label": 0 if positive_is_a else 1,
                    "positive_candidate": positive_candidate,
                    "negative_candidate": negative_candidate,
                    "negative_type": negative_type,
                    "negative_option": negative_option,
                    "position_strategy": position_strategy,
                }
            )
            pair_index += 1

    return pair_rows


def default_context_tail_chars(dataset_name, override):
    if override is not None:
        return override if override > 0 else None
    return DATASET_SPECS[dataset_name]["context_tail_chars"]


def prepare_dataset_run(dataset_name, input_path, start, end, position_strategy, shuffle_seed):
    spec = DATASET_SPECS[dataset_name]
    rows = normalize_rows(dataset_name, load_json(input_path), spec)
    selected_rows, selected_group_ids = slice_grouped_cases(rows, start, end)
    pair_rows = build_pairwise_rows(
        dataset_name=dataset_name,
        rows=selected_rows,
        selected_group_ids=selected_group_ids,
        position_strategy=position_strategy,
        seed=shuffle_seed,
    )
    return {
        "dataset": dataset_name,
        "display_name": spec["display_name"],
        "input_path": str(input_path),
        "selected_rows": selected_rows,
        "selected_group_ids": selected_group_ids,
        "pair_rows": pair_rows,
    }


def dataset_output_stem(dataset_name, start, end, timestamp):
    end_tag = "end" if end is None else str(end)
    return f"{dataset_name}_pandalm_pairwise_{start}-{end_tag}_{timestamp}"


def build_pandalm_instruction():
    return (
        "Choose the response that is more consistent with the target profile in the given dialogue context. "
        "If both responses are equally consistent or equally inconsistent, output Tie."
    )


def build_pandalm_input(row, context_tail_chars=None):
    context = truncate_tail(row.get("dialogue") or row.get("context") or "", context_tail_chars)
    return (
        f"Target profile:\n{row['profile']}\n\n"
        f"Dialogue context:\n{context}\n"
    )


def build_pandalm_prompt(row, context_tail_chars=None):
    instruction = build_pandalm_instruction()
    input_text = build_pandalm_input(row, context_tail_chars=context_tail_chars)
    response1 = str(row["candidate_a"]).strip()
    response2 = str(row["candidate_b"]).strip()
    return (
        "Below are two responses for a given task. The task is defined by the Instruction with an Input that "
        "provides further context. Evaluate which response is more consistent with the target profile in context.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n"
        f"### Response 1:\n{response1}\n\n"
        f"### Response 2:\n{response2}\n\n"
        "### Evaluation:\n"
        "Output exactly one of: 1, 2, or Tie.\n"
    )


def import_pandalm_provider():
    if str(PANDALM_ROOT) not in sys.path:
        sys.path.insert(0, str(PANDALM_ROOT))
    from pandalm.utils.pandalm_inference import PandaLMBatchInferenceProvider, seed_everything

    return PandaLMBatchInferenceProvider, seed_everything


def prepare_pandalm_prompts(handler, rows, context_tail_chars):
    handler.prepared = []
    for row in rows:
        prompt = build_pandalm_prompt(row, context_tail_chars=context_tail_chars)
        handler.prepared.append(handler.tokenizer(prompt, return_tensors="pt", padding=True))


def parse_pandalm_result(text):
    first_line = str(text or "").strip().splitlines()
    first = first_line[0].strip() if first_line else ""
    if first == "1":
        return "A"
    if first == "2":
        return "B"
    if first.lower() == "tie":
        return "tie"

    match = re.search(r"\b(1|2|Tie)\b", str(text or ""), re.IGNORECASE)
    if not match:
        return None
    value = match.group(1)
    return "tie" if value.lower() == "tie" else ("A" if value == "1" else "B")


def run_pairwise(handler, rows, context_tail_chars, temperature, top_p, top_k, num_beams, max_new_tokens, repetition_penalty):
    prepare_pandalm_prompts(handler, rows, context_tail_chars=context_tail_chars)
    generated = handler.inference(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    results = []
    for row, text in zip(rows, generated):
        preferred_response = parse_pandalm_result(text)
        positive_candidate = row.get("positive_candidate", "A")
        if preferred_response == "tie":
            prefers_positive = "tie"
        elif preferred_response in {"A", "B"}:
            prefers_positive = preferred_response == positive_candidate
        else:
            prefers_positive = None
        results.append(
            {
                **row,
                "judge_output": text,
                "preferred_response": preferred_response,
                "prefers_positive": prefers_positive,
            }
        )
    return results


def bucket_pair_stats(rows):
    parsed = [r for r in rows if r.get("prefers_positive") in {True, False, "tie"}]
    n = len(parsed)
    if not n:
        return {
            "n": len(rows),
            "n_parsed": 0,
            "pair_acc": None,
            "pair_auc": None,
            "non_tie_accuracy": None,
            "win_count": 0,
            "tie_count": 0,
            "fail_count": 0,
        }
    wins = sum(r["prefers_positive"] is True for r in parsed)
    fails = sum(r["prefers_positive"] is False for r in parsed)
    ties = sum(r["prefers_positive"] == "tie" for r in parsed)
    return {
        "n": len(rows),
        "n_parsed": n,
        "pair_acc": wins / n,
        "pair_auc": (wins + 0.5 * ties) / n,
        "non_tie_accuracy": (wins / (wins + fails)) if (wins + fails) else None,
        "win_count": wins,
        "tie_count": ties,
        "fail_count": fails,
    }


def strict_group_accuracy(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["group_id"]].append(row)

    hits = 0
    total = 0
    skipped = 0
    for rows in grouped.values():
        parsed = [r for r in rows if r.get("prefers_positive") in {True, False, "tie"}]
        if len(parsed) != len(rows):
            skipped += 1
            continue
        total += 1
        if all(r["prefers_positive"] is True for r in parsed):
            hits += 1
    return (hits / total if total else None), total, skipped


def summarize_pairwise(results):
    parsed = [r for r in results if r.get("prefers_positive") in {True, False, "tie"}]
    wins = sum(r["prefers_positive"] is True for r in parsed)
    fails = sum(r["prefers_positive"] is False for r in parsed)
    ties = sum(r["prefers_positive"] == "tie" for r in parsed)
    strict_acc, strict_total, skipped_groups = strict_group_accuracy(results)

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
        "mode": "pairwise",
        "auc": None,
        "n_pairs": len(results),
        "n_parsed": len(parsed),
        "parse_rate": (len(parsed) / len(results)) if results else None,
        "pair_acc": (wins / len(parsed)) if parsed else None,
        "pair_auc": ((wins + 0.5 * ties) / len(parsed)) if parsed else None,
        "strict_group_acc": strict_acc,
        "n_groups_scored_for_strict_acc": strict_total,
        "n_groups_skipped_for_strict_acc": skipped_groups,
        "non_tie_accuracy": (wins / (wins + fails)) if (wins + fails) else None,
        "win_count": wins,
        "tie_count": ties,
        "fail_count": fails,
        "by_negative_type": {
            str(key): bucket_pair_stats(rows) for key, rows in sorted(by_negative_type.items(), key=lambda kv: str(kv[0]))
        },
        "by_trait": {
            str(key): bucket_pair_stats(rows) for key, rows in sorted(by_trait.items(), key=lambda kv: str(kv[0]))
        },
        "by_level": {
            str(key): bucket_pair_stats(rows) for key, rows in sorted(by_level.items(), key=lambda kv: str(kv[0]))
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run PandaLM pairwise baseline on three datasets.")
    parser.add_argument(
        "--dataset",
        choices=["hard", "swapwithintrait", "self_awareness", "all"],
        default="all",
    )
    parser.add_argument("--input_file", type=str, default=None, help="Optional override for a single dataset run.")
    parser.add_argument("--start", type=int, default=0, help="Group start index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Group end index (exclusive).")
    parser.add_argument(
        "--position_strategy",
        choices=["balanced", "alternate", "positive_a", "positive_b"],
        default="balanced",
        help="How to place the positive candidate in A/B for fairness.",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument(
        "--context_tail_chars",
        type=int,
        default=None,
        help="Optional tail truncation for dialogue/context. <=0 disables truncation.",
    )
    parser.add_argument("--prepare_only", action="store_true", help="Only build and save prepared pairwise files.")
    parser.add_argument("--judge", type=str, default=DEFAULT_JUDGE, help="PandaLM model name or local path.")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    if args.dataset == "all" and args.input_file:
        raise ValueError("--input_file can only be used when --dataset is a single dataset.")

    dataset_names = list(DATASET_SPECS.keys()) if args.dataset == "all" else [args.dataset]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    prepared_runs = []
    for dataset_name in dataset_names:
        input_path = Path(args.input_file) if args.input_file else DATASET_SPECS[dataset_name]["input_file"]
        run = prepare_dataset_run(
            dataset_name=dataset_name,
            input_path=input_path,
            start=args.start,
            end=args.end,
            position_strategy=args.position_strategy,
            shuffle_seed=args.shuffle_seed,
        )
        stem = dataset_output_stem(dataset_name, args.start, args.end, timestamp)
        prepared_path = out_dir / f"{stem}_prepared_pairs.json"
        dump_json(run["pair_rows"], prepared_path)
        run["prepared_pairs_path"] = str(prepared_path)
        prepared_runs.append(run)

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepare_only": True,
                    "runs": [
                        {
                            "dataset": run["dataset"],
                            "display_name": run["display_name"],
                            "input_path": run["input_path"],
                            "prepared_pairs_path": run["prepared_pairs_path"],
                            "n_selected_groups": len(run["selected_group_ids"]),
                            "n_selected_rows": len(run["selected_rows"]),
                            "n_pairs": len(run["pair_rows"]),
                        }
                        for run in prepared_runs
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    PandaLMBatchInferenceProvider, seed_everything = import_pandalm_provider()
    seed_everything(args.seed)
    handler = PandaLMBatchInferenceProvider(model_path=args.judge)

    console_runs = []
    for run in prepared_runs:
        dataset_name = run["dataset"]
        context_tail_chars = default_context_tail_chars(dataset_name, args.context_tail_chars)
        results = run_pairwise(
            handler=handler,
            rows=run["pair_rows"],
            context_tail_chars=context_tail_chars,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        summary = summarize_pairwise(results)
        summary["config"] = {
            "dataset": dataset_name,
            "display_name": run["display_name"],
            "input_path": run["input_path"],
            "prepared_pairs_path": run["prepared_pairs_path"],
            "judge": args.judge,
            "start": args.start,
            "end": args.end,
            "n_selected_groups": len(run["selected_group_ids"]),
            "n_selected_rows": len(run["selected_rows"]),
            "n_pairs": len(run["pair_rows"]),
            "position_strategy": args.position_strategy,
            "shuffle_seed": args.shuffle_seed,
            "context_tail_chars": context_tail_chars,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
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
                "prepared_pairs_path": run["prepared_pairs_path"],
                "results_path": str(results_path),
                "summary_path": str(summary_path),
                "auc": summary["auc"],
                "pair_acc": summary["pair_acc"],
                "pair_auc": summary["pair_auc"],
                "strict_group_acc": summary["strict_group_acc"],
                "parse_rate": summary["parse_rate"],
                "n_pairs": summary["n_pairs"],
                "n_groups_scored_for_strict_acc": summary["n_groups_scored_for_strict_acc"],
            }
        )

    print(json.dumps({"runs": console_runs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
