import argparse
import datetime
import json
import os
import re
import time
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CASES_FILE_PRESETS = {
    "big5_swap": "data/big5_persona_easy.json",
    "big5_hard": "data/big5_persona_hard.json",
    "socialbench": "data/social_persona.json",
}

DIRECT_JUDGE_SCALE_5 = {
    "1": "clearly inconsistent with the target profile or role",
    "2": "more inconsistent than consistent",
    "3": "mixed, borderline, or genuinely uncertain",
    "4": "more consistent than inconsistent",
    "5": "clearly consistent with the target profile or role",
}

DIRECT_JUDGE_SCALE_7 = {
    "1": "clearly and strongly inconsistent with the target profile or role",
    "2": "strongly inconsistent with the target profile or role",
    "3": "somewhat more inconsistent than consistent",
    "4": "mixed, borderline, or genuinely uncertain",
    "5": "somewhat more consistent than inconsistent",
    "6": "strongly consistent with the target profile or role",
    "7": "clearly and strongly consistent with the target profile or role",
}


def resolve_input_path(path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    for base in (REPO_ROOT, SCRIPT_DIR, os.getcwd()):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(REPO_ROOT, path)


def resolve_output_dir(path):
    if not path:
        path = "api_results"
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def infer_cases_tag(path, explicit_tag=None):
    if explicit_tag:
        return explicit_tag
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(rows):
    normalized = []
    for row in rows:
        item = dict(row)
        if "context" not in item and "dialogue" in item:
            item["context"] = item["dialogue"]
        if "group_id" not in item and "id" in item:
            item["group_id"] = item["id"]
        normalized.append(item)
    return normalized


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_group_id(row):
    if row.get("label") == 0 and row.get("source_case_id"):
        source_case_id = row["source_case_id"]
        if isinstance(source_case_id, str):
            if "_NEG" in source_case_id:
                source_case_id = source_case_id.split("_NEG", 1)[0]
            if source_case_id.endswith("_pos"):
                source_case_id = source_case_id[: -len("_pos")]
            if source_case_id.endswith("_neg_swap"):
                source_case_id = source_case_id[: -len("_neg_swap")]
        return source_case_id
    case_id = row.get("case_id")
    if isinstance(case_id, str):
        if "_NEG" in case_id:
            return case_id.split("_NEG", 1)[0]
        if "_neg_" in case_id:
            return case_id.split("_neg_", 1)[0]
        if case_id.endswith("_pos"):
            return case_id[: -len("_pos")]
        if case_id.endswith("_neg_swap"):
            return case_id[: -len("_neg_swap")]
    return row.get("group_id") or row.get("id") or case_id


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
    end = min(end, len(ordered_ids))
    selected_ids = ordered_ids[start:end]
    sliced = []
    for gid in selected_ids:
        sliced.extend(grouped[gid])
    return sliced, selected_ids


def render_scale_block(scale_dict):
    return "\n".join(f"- {k}: {v}" for k, v in scale_dict.items())


def normalize_score_scales(raw_scales):
    scales = []
    for scale in raw_scales:
        value = int(scale)
        if value not in (5, 7):
            raise ValueError(f"Unsupported score scale: {scale}")
        if value not in scales:
            scales.append(value)
    return sorted(scales)


def build_mode_tag(score_scales):
    suffix = "_".join(str(scale) for scale in score_scales)
    return f"direct_cot_{suffix}"


def get_method_prefixes_for_scales(score_scales):
    prefixes = []
    if 5 in score_scales:
        prefixes.extend(["direct_judge", "direct_cot"])
    if 7 in score_scales:
        prefixes.extend(["direct_judge_7", "direct_cot_7"])
    return prefixes


def get_score_keys_for_scales(score_scales):
    score_keys = []
    if 5 in score_scales:
        score_keys.extend(
            [
                "direct_judge_rating",
                "direct_judge_score_01",
                "direct_judge_centered_score",
                "direct_cot_rating",
                "direct_cot_score_01",
                "direct_cot_centered_score",
            ]
        )
    if 7 in score_scales:
        score_keys.extend(
            [
                "direct_judge_7_rating",
                "direct_judge_7_score_01",
                "direct_judge_7_centered_score",
                "direct_cot_7_rating",
                "direct_cot_7_score_01",
                "direct_cot_7_centered_score",
            ]
        )
    return score_keys


def get_absolute_score_keys_for_scales(score_scales):
    keys = set()
    if 5 in score_scales:
        keys.update({"direct_judge_score_01", "direct_cot_score_01"})
    if 7 in score_scales:
        keys.update({"direct_judge_7_score_01", "direct_cot_7_score_01"})
    return keys


def infer_available_scales_from_results(results):
    has_5 = False
    has_7 = False
    for row in results:
        scores = row.get("scores", {})
        if isinstance(scores.get("direct_judge_score_01"), (int, float)) or isinstance(
            scores.get("direct_cot_score_01"), (int, float)
        ):
            has_5 = True
        if isinstance(scores.get("direct_judge_7_score_01"), (int, float)) or isinstance(
            scores.get("direct_cot_7_score_01"), (int, float)
        ):
            has_7 = True
    scales = []
    if has_5:
        scales.append(5)
    if has_7:
        scales.append(7)
    return scales


def build_direct_judge_prompt(row):
    profile_block = row["profile"]
    task = "Judge the overall consistency of the candidate response with the target profile or role in the given dialogue context."
    return f"""You are doing overall consistency judgment for a candidate response.

Task:
{task}

Use this scale:
{render_scale_block(DIRECT_JUDGE_SCALE_5)}

Instructions:
1. Output only one integer: 1, 2, 3, 4, or 5.
2. Do not explain.
3. Do not output any words, punctuation, or justification.

Target profile or role:
{profile_block}

Dialogue context:
{row['context']}

Candidate response:
{row['response']}

Answer with exactly one integer: 1, 2, 3, 4, or 5.
Answer:"""


def build_direct_cot_judge_prompt(row):
    profile_block = row["profile"]
    task = "Before scoring, think briefly about three dimensions, then output a final 1-5 consistency score."
    return f"""You are doing overall consistency judgment for a candidate response.

Task:
{task}

Use this scale:
{render_scale_block(DIRECT_JUDGE_SCALE_5)}

You may consider consistency from these dimensions:
- Interactional Task
- Interpersonal stance pattern
- Expression style pattern

Instructions:
1. Think privately before scoring.
2. Do not explain.
3. Do not output reasoning.
4. Output only one integer: 1, 2, 3, 4, or 5.

Target profile or role:
{profile_block}

Dialogue context:
{row['context']}

Candidate response:
{row['response']}

Answer with exactly one integer: 1, 2, 3, 4, or 5.
Answer:"""


def build_direct_judge_prompt_7(row):
    profile_block = row["profile"]
    task = "Judge the overall consistency of the candidate response with the target profile or role in the given dialogue context."
    return f"""You are doing overall consistency judgment for a candidate response.

Task:
{task}

Use this scale:
{render_scale_block(DIRECT_JUDGE_SCALE_7)}

Instructions:
1. Output only one integer: 1, 2, 3, 4, 5, 6, or 7.
2. Do not explain.
3. Do not output any words, punctuation, or justification.

Target profile or role:
{profile_block}

Dialogue context:
{row['context']}

Candidate response:
{row['response']}

Answer with exactly one integer: 1, 2, 3, 4, 5, 6, or 7.
Answer:"""


def build_direct_cot_judge_prompt_7(row):
    profile_block = row["profile"]
    task = "Before scoring, think briefly about three dimensions, then output a final 1-7 consistency score."
    return f"""You are doing overall consistency judgment for a candidate response.

Task:
{task}

Use this scale:
{render_scale_block(DIRECT_JUDGE_SCALE_7)}

You may consider consistency from these dimensions:
- Interactional Goal
- Interpersonal stance pattern
- Expression style pattern

Instructions:
1. Think privately before scoring.
2. Do not explain.
3. Do not output reasoning.
4. Output only one integer: 1, 2, 3, 4, 5, 6, or 7.

Target profile or role:
{profile_block}

Dialogue context:
{row['context']}

Candidate response:
{row['response']}

Answer with exactly one integer: 1, 2, 3, 4, 5, 6, or 7.
Answer:"""


def make_openai_client(api_base=None, api_key=None):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Python package 'openai' is required for API evaluation.") from exc

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    resolved_api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENROUTER_API_BASE")
    kwargs = {"api_key": resolved_api_key}
    if resolved_api_base:
        kwargs["base_url"] = resolved_api_base
    return OpenAI(**kwargs)


def extract_message_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts).strip()
    return str(content or "").strip()


def call_chat_completion(client, model_name, prompt, temperature=0.0, max_tokens=16, timeout=120, max_retries=5, retry_sleep=3.0):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            choice = (getattr(response, "choices", None) or [None])[0]
            message = getattr(choice, "message", None)
            content = getattr(message, "content", "")
            return {
                "ok": True,
                "raw_response": response.model_dump() if hasattr(response, "model_dump") else str(response),
                "text": extract_message_text(content),
            }
        except Exception as e:
            last_error = str(e)
            print(
                json.dumps(
                    {
                        "event": "api_retry",
                        "model_name": model_name,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "error": last_error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if attempt < max_retries - 1:
            time.sleep(retry_sleep * (attempt + 1))

    return {
        "ok": False,
        "raw_response": None,
        "text": "",
        "error": last_error or "unknown_api_error",
    }


def parse_rating(raw_output, max_rating=5):
    matches = re.findall(r"\b(\d+)\b", str(raw_output or ""))
    rating = None
    for match in reversed(matches):
        value = int(match)
        if 1 <= value <= max_rating:
            rating = value
            break
    return rating


def compute_direct_scores(rating, max_rating):
    score_01 = (float(rating) - 1.0) / float(max_rating - 1)
    midpoint = (max_rating + 1.0) / 2.0
    half_span = (max_rating - 1.0) / 2.0
    centered = (float(rating) - midpoint) / half_span
    return {
        "rating": float(rating),
        "score_01": score_01,
        "centered_score": centered,
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


def fixed_threshold_stats(results, score_key, threshold):
    labels = []
    preds = []
    for row in results:
        score = row.get("scores", {}).get(score_key)
        if not isinstance(score, (int, float)):
            continue
        labels.append(int(row["label"]))
        preds.append(1 if score >= threshold else 0)
    if not labels:
        return None
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "n_valid": len(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def negative_only_stats(results, score_key, thresholds):
    neg_scores = [
        row.get("scores", {}).get(score_key)
        for row in results
        if row.get("label") == 0 and isinstance(row.get("scores", {}).get(score_key), (int, float))
    ]
    if not neg_scores:
        return None
    out = {
        "n_negatives": len(neg_scores),
        "mean_negative_score": sum(neg_scores) / len(neg_scores),
        "max_negative_score": max(neg_scores),
    }
    for t in thresholds:
        out[f"fp_rate_at_{t:.2f}"] = sum(1 for s in neg_scores if s >= t) / len(neg_scores)
    return out


def positive_only_stats(results, score_key, thresholds):
    pos_scores = [
        row.get("scores", {}).get(score_key)
        for row in results
        if row.get("label") == 1 and isinstance(row.get("scores", {}).get(score_key), (int, float))
    ]
    if not pos_scores:
        return None
    out = {
        "n_positives": len(pos_scores),
        "mean_positive_score": sum(pos_scores) / len(pos_scores),
        "min_positive_score": min(pos_scores),
    }
    for t in thresholds:
        out[f"fn_rate_at_{t:.2f}"] = sum(1 for s in pos_scores if s < t) / len(pos_scores)
    return out


def pair_records(results, score_key):
    grouped, _ = group_cases(results)
    records = []
    for gid, rows in grouped.items():
        pos_rows = [r for r in rows if r.get("label") == 1]
        neg_rows = [r for r in rows if r.get("label") == 0]
        if not pos_rows or not neg_rows:
            continue
        pos = max(pos_rows, key=lambda r: r.get("scores", {}).get(score_key, float("-inf")))
        for neg in neg_rows:
            pos_score = pos.get("scores", {}).get(score_key)
            neg_score = neg.get("scores", {}).get(score_key)
            if not isinstance(pos_score, (int, float)) or not isinstance(neg_score, (int, float)):
                continue
            records.append(
                {
                    "group_id": gid,
                    "negative_type": neg.get("negative_type"),
                    "pos_score": pos_score,
                    "neg_score": neg_score,
                    "margin": pos_score - neg_score,
                    "correct": pos_score > neg_score,
                }
            )
    return records


def wrong_high_confidence_rate(pair_recs, threshold=0.8):
    wrong = [r for r in pair_recs if not r["correct"]]
    if not wrong:
        return None
    high = sum(1 for r in wrong if r["neg_score"] >= threshold)
    return {
        "threshold": threshold,
        "n_wrong_pairs": len(wrong),
        "wrong_high_confidence_rate": high / len(wrong),
    }


def method_failure_summary(results, score_key, threshold):
    fixed = fixed_threshold_stats(results, score_key, threshold)
    pair_acc, pair_n = pair_accuracy(results, score_key)
    neg = negative_only_stats(results, score_key, thresholds=(threshold,))
    if fixed is None or pair_acc is None:
        return None
    out = {
        "threshold": threshold,
        "fixed_threshold_total_error_rate": 1.0 - fixed["accuracy"],
        "pairwise_failure_rate": 1.0 - pair_acc,
        "n_pairs": pair_n,
    }
    if neg is not None:
        out["hard_negative_false_accept_rate"] = neg[f"fp_rate_at_{threshold:.2f}"]
        out["n_hard_negatives"] = neg["n_negatives"]
    return out


def summarize(results, dataset_name, thresholds, score_scales=None):
    valid = [r for r in results if r.get("parse_ok")]
    active_scales = normalize_score_scales(score_scales) if score_scales else infer_available_scales_from_results(valid)
    out = {
        "schema": "api_direct_only",
        "dataset": dataset_name,
        "n_total": len(results),
        "n_valid": len(valid),
        "n_parse_failed": len(results) - len(valid),
        "available_score_scales": active_scales,
    }

    score_keys = get_score_keys_for_scales(active_scales)
    absolute_threshold_score_keys = get_absolute_score_keys_for_scales(active_scales)
    for key in score_keys:
        vals = [r["scores"].get(key) for r in valid if isinstance(r["scores"].get(key), (int, float))]
        if not vals:
            continue
        labels = [r["label"] for r in valid if isinstance(r["scores"].get(key), (int, float))]
        out[f"{key}_auc"] = rank_auc(labels, vals)
        p_acc, pair_n = pair_accuracy(valid, key)
        p_auc, _ = pair_auc(valid, key)
        out[f"{key}_pair_acc"] = p_acc
        out[f"{key}_pair_auc"] = p_auc
        out[f"{key}_strict_group_acc"] = strict_group_accuracy(valid, key)
        out[f"{key}_pair_n"] = pair_n
        if key in absolute_threshold_score_keys:
            out[f"{key}_negative_only"] = negative_only_stats(valid, key, thresholds)
            out[f"{key}_positive_only"] = positive_only_stats(valid, key, thresholds)
            out[f"{key}_threshold_stats"] = {
                f"{t:.2f}": fixed_threshold_stats(valid, key, t) for t in thresholds
            }
            out[f"{key}_failure_by_threshold"] = {
                f"{t:.2f}": method_failure_summary(valid, key, t) for t in thresholds
            }
            pair_recs = pair_records(valid, key)
            out[f"{key}_wrong_high_confidence_by_threshold"] = {
                f"{t:.2f}": wrong_high_confidence_rate(pair_recs, threshold=t) for t in thresholds
            }

    return out


def process_rows(rows, client, model_name, request_timeout, max_retries, retry_sleep, score_scales):
    results = []
    active_scales = normalize_score_scales(score_scales)
    for idx, row in enumerate(rows, start=1):
        print(
            json.dumps(
                {
                    "event": "row_start",
                    "row_index": idx,
                    "total_rows": len(rows),
                    "case_id": row.get("case_id"),
                    "label": row.get("label"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        scores = {}
        parse_flags = []
        prompt_records = {}

        if 5 in active_scales:
            direct_prompt = build_direct_judge_prompt(row)
            direct_resp = call_chat_completion(
                client=client,
                model_name=model_name,
                prompt=direct_prompt,
                temperature=0.0,
                max_tokens=16,
                timeout=request_timeout,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
            direct_rating = parse_rating(direct_resp.get("text", ""), max_rating=5)
            direct_parse_ok = direct_rating is not None
            parse_flags.append(direct_parse_ok)
            if direct_parse_ok:
                direct_scores = compute_direct_scores(direct_rating, max_rating=5)
                scores.update(
                    {
                        "direct_judge_rating": direct_scores["rating"],
                        "direct_judge_score_01": direct_scores["score_01"],
                        "direct_judge_centered_score": direct_scores["centered_score"],
                    }
                )
            prompt_records["direct_judge"] = {
                "prompt": direct_prompt,
                "raw_output": direct_resp.get("text", ""),
                "api_ok": direct_resp.get("ok", False),
                "api_error": direct_resp.get("error"),
                "rating": direct_rating,
                "parse_ok": direct_parse_ok,
            }

            direct_cot_prompt = build_direct_cot_judge_prompt(row)
            direct_cot_resp = call_chat_completion(
                client=client,
                model_name=model_name,
                prompt=direct_cot_prompt,
                temperature=0.0,
                max_tokens=16,
                timeout=request_timeout,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
            direct_cot_rating = parse_rating(direct_cot_resp.get("text", ""), max_rating=5)
            direct_cot_parse_ok = direct_cot_rating is not None
            parse_flags.append(direct_cot_parse_ok)
            if direct_cot_parse_ok:
                direct_cot_scores = compute_direct_scores(direct_cot_rating, max_rating=5)
                scores.update(
                    {
                        "direct_cot_rating": direct_cot_scores["rating"],
                        "direct_cot_score_01": direct_cot_scores["score_01"],
                        "direct_cot_centered_score": direct_cot_scores["centered_score"],
                    }
                )
            prompt_records["direct_cot"] = {
                "prompt": direct_cot_prompt,
                "raw_output": direct_cot_resp.get("text", ""),
                "api_ok": direct_cot_resp.get("ok", False),
                "api_error": direct_cot_resp.get("error"),
                "rating": direct_cot_rating,
                "parse_ok": direct_cot_parse_ok,
            }

        if 7 in active_scales:
            direct_prompt_7 = build_direct_judge_prompt_7(row)
            direct_resp_7 = call_chat_completion(
                client=client,
                model_name=model_name,
                prompt=direct_prompt_7,
                temperature=0.0,
                max_tokens=16,
                timeout=request_timeout,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
            direct_rating_7 = parse_rating(direct_resp_7.get("text", ""), max_rating=7)
            direct_parse_ok_7 = direct_rating_7 is not None
            parse_flags.append(direct_parse_ok_7)
            if direct_parse_ok_7:
                direct_scores_7 = compute_direct_scores(direct_rating_7, max_rating=7)
                scores.update(
                    {
                        "direct_judge_7_rating": direct_scores_7["rating"],
                        "direct_judge_7_score_01": direct_scores_7["score_01"],
                        "direct_judge_7_centered_score": direct_scores_7["centered_score"],
                    }
                )
            prompt_records["direct_judge_7"] = {
                "prompt": direct_prompt_7,
                "raw_output": direct_resp_7.get("text", ""),
                "api_ok": direct_resp_7.get("ok", False),
                "api_error": direct_resp_7.get("error"),
                "rating": direct_rating_7,
                "parse_ok": direct_parse_ok_7,
            }

            direct_cot_prompt_7 = build_direct_cot_judge_prompt_7(row)
            direct_cot_resp_7 = call_chat_completion(
                client=client,
                model_name=model_name,
                prompt=direct_cot_prompt_7,
                temperature=0.0,
                max_tokens=16,
                timeout=request_timeout,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
            direct_cot_rating_7 = parse_rating(direct_cot_resp_7.get("text", ""), max_rating=7)
            direct_cot_parse_ok_7 = direct_cot_rating_7 is not None
            parse_flags.append(direct_cot_parse_ok_7)
            if direct_cot_parse_ok_7:
                direct_cot_scores_7 = compute_direct_scores(direct_cot_rating_7, max_rating=7)
                scores.update(
                    {
                        "direct_cot_7_rating": direct_cot_scores_7["rating"],
                        "direct_cot_7_score_01": direct_cot_scores_7["score_01"],
                        "direct_cot_7_centered_score": direct_cot_scores_7["centered_score"],
                    }
                )
            prompt_records["direct_cot_7"] = {
                "prompt": direct_cot_prompt_7,
                "raw_output": direct_cot_resp_7.get("text", ""),
                "api_ok": direct_cot_resp_7.get("ok", False),
                "api_error": direct_cot_resp_7.get("error"),
                "rating": direct_cot_rating_7,
                "parse_ok": direct_cot_parse_ok_7,
            }

        result = dict(row)
        result["schema"] = "api_direct_only"
        result["parse_ok"] = all(parse_flags) if parse_flags else False
        result["scores"] = scores
        result.update(prompt_records)
        results.append(result)

        print(
            json.dumps(
                {
                    "event": "row_done",
                    "processed_rows": idx,
                    "total_rows": len(rows),
                    "parse_ok": result["parse_ok"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return results


def infer_dataset_name(cases_preset, input_path):
    del input_path
    if cases_preset == "socialbench":
        return "socialbench"
    if cases_preset == "big5_swap":
        return "big5_swap"
    return "big5_hard"


def build_summary_payload(analysis, score_scales):
    payload = {
        "n_valid_rows": analysis["n_valid"],
    }
    if 5 in score_scales:
        payload.update(
            {
                "direct_judge_score_01_auc": analysis.get("direct_judge_score_01_auc"),
                "direct_judge_score_01_pair_acc": analysis.get("direct_judge_score_01_pair_acc"),
                "direct_judge_score_01_pair_auc": analysis.get("direct_judge_score_01_pair_auc"),
                "direct_judge_score_01_strict_group_acc": analysis.get("direct_judge_score_01_strict_group_acc"),
                "direct_cot_score_01_auc": analysis.get("direct_cot_score_01_auc"),
                "direct_cot_score_01_pair_acc": analysis.get("direct_cot_score_01_pair_acc"),
                "direct_cot_score_01_pair_auc": analysis.get("direct_cot_score_01_pair_auc"),
                "direct_cot_score_01_strict_group_acc": analysis.get("direct_cot_score_01_strict_group_acc"),
            }
        )
    if 7 in score_scales:
        payload.update(
            {
                "direct_judge_7_score_01_auc": analysis.get("direct_judge_7_score_01_auc"),
                "direct_judge_7_score_01_pair_acc": analysis.get("direct_judge_7_score_01_pair_acc"),
                "direct_judge_7_score_01_pair_auc": analysis.get("direct_judge_7_score_01_pair_auc"),
                "direct_judge_7_score_01_strict_group_acc": analysis.get("direct_judge_7_score_01_strict_group_acc"),
                "direct_cot_7_score_01_auc": analysis.get("direct_cot_7_score_01_auc"),
                "direct_cot_7_score_01_pair_acc": analysis.get("direct_cot_7_score_01_pair_acc"),
                "direct_cot_7_score_01_pair_auc": analysis.get("direct_cot_7_score_01_pair_auc"),
                "direct_cot_7_score_01_strict_group_acc": analysis.get("direct_cot_7_score_01_strict_group_acc"),
            }
        )
    return payload


def main():
    parser = argparse.ArgumentParser(description="API-based direct/coT evaluation for Big5 and SocialBench datasets.")
    parser.add_argument("--cases_preset", choices=sorted(CASES_FILE_PRESETS.keys()), required=True)
    parser.add_argument("--cases_file", type=str, default=None)
    parser.add_argument("--cases_tag", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    # parser.add_argument("--out_dir", type=str, default="api_GPT_5.4_baselines")
    parser.add_argument("--out_dir", type=str, default="api_Gemini_3_baselines")
    # parser.add_argument("--model_name", type=str, default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.4"))
    parser.add_argument("--model_name", type=str, default=os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview"))
    parser.add_argument("--api_base", type=str, default=os.environ.get("OPENROUTER_API_BASE", ""))
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY") ,
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=3.0)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.71, 0.75, 0.80])
    parser.add_argument("--score_scales", nargs="+", choices=["5", "7"], default=["5"])
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set --api_key or DEEPSEEK_API_KEY.")

    score_scales = normalize_score_scales(args.score_scales)

    cases_file = args.cases_file or CASES_FILE_PRESETS[args.cases_preset]
    input_path = resolve_input_path(cases_file)
    out_dir = resolve_output_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = normalize_rows(load_cases(input_path))
    selected_rows, selected_group_ids = slice_grouped_cases(rows, args.start, args.end)
    dataset_name = infer_dataset_name(args.cases_preset, input_path)
    client = make_openai_client(api_base=args.api_base, api_key=args.api_key)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cases_tag = infer_cases_tag(input_path, args.cases_tag)
    mode_tag = build_mode_tag(score_scales)
    stem = f"{dataset_name}_{args.model_name}_{mode_tag}_{args.start}-{args.end}_{timestamp}"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    results_path = os.path.join(out_dir, f"{safe_stem}_results.json")
    analysis_path = os.path.join(out_dir, f"{safe_stem}_analysis.json")

    results = process_rows(
        selected_rows,
        client=client,
        model_name=args.model_name,
        request_timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        score_scales=score_scales,
    )
    analysis = summarize(results, dataset_name, args.thresholds, score_scales=score_scales)
    analysis["config"] = {
        "dataset": dataset_name,
        "cases_preset": args.cases_preset,
        "input_path": input_path,
        "cases_tag": cases_tag,
        "start": args.start,
        "end": args.end,
        "n_selected_rows": len(selected_rows),
        "n_selected_groups": len(selected_group_ids),
        "model_name": args.model_name,
        "api_base": args.api_base,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "timeout": args.timeout,
        "thresholds": args.thresholds,
        "score_scales": score_scales,
        "mode_tag": mode_tag,
        "methods": get_method_prefixes_for_scales(score_scales),
    }

    dump_json(results, results_path)
    dump_json(analysis, analysis_path)

    summary_payload = {
        "results_path": results_path,
        "analysis_path": analysis_path,
        "n_selected_rows": len(selected_rows),
    }
    summary_payload.update(build_summary_payload(analysis, score_scales))
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
