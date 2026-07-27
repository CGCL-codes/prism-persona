#!/usr/bin/env python3
"""Unified PRISM evaluation for Big5-Persona-EASY and Big5-Persona-HARD.

    prism = mean_d q_d(aligned | c, r)

where each dimension-specific inverse prompt estimates the posterior over
A/B/C latent states. The A/B/C option order is deterministically shuffled per
case and dimension to reduce positional bias.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
from collections import defaultdict

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # type: ignore
        return x


MODEL_PRESETS = {
    "llama3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
    "qwen25_14b": "Qwen/Qwen2.5-14B-Instruct",
}

DATASET_PRESETS = {
    "easy": {
        "display": "Big5-Persona-EASY",
        "file": "data/big5_persona_easy.json",
    },
    "hard": {
        "display": "Big5-Persona-HARD",
        "file": "data/big5_persona_hard.json",
    },
}

DIMENSIONS = {
    "agenda_pattern": {
        "display": "Task",
        "label_order": ["A", "B", "C"],
        "inverse_task": "Given the current dialogue context and the candidate response, infer which agenda pattern state is actually expressed by the response. Do not output a consistency judgment.",
        "fallback_b": "indeterminate_agenda_pattern: weakly marked, mixed, generic, or underspecified agenda.",
    },
    "interpersonal_stance_pattern": {
        "display": "Stance",
        "label_order": ["A", "B", "C"],
        "inverse_task": "Given the current dialogue context and the candidate response, infer which interpersonal stance pattern state is actually expressed by the response. Do not output a consistency judgment.",
        "fallback_b": "indeterminate_stance_pattern: weakly marked, mixed, flat, generic, or hard-to-read interpersonal stance.",
    },
    "expression_style_pattern": {
        "display": "Style",
        "label_order": ["A", "B", "C"],
        "inverse_task": "Given the current dialogue context and the candidate response, infer which expression style pattern state is actually expressed by the response. Do not output a consistency judgment.",
        "fallback_b": "indeterminate_style_pattern: ordinary, mixed, weakly marked, generic, or underspecified wording.",
    },
}

TRAIT_PATTERN_PROTOTYPES = {
    "agenda_pattern": {
        "openness": {
            "high": ["exploring possibilities", "reframing toward novelty", "expanding options", "entertaining unconventional directions"],
            "low": ["grounding the exchange", "staying conventional", "narrowing toward practical next steps", "keeping to familiar routines"],
        },
        "conscientiousness": {
            "high": ["planning next steps", "taking ownership", "sequencing action clearly", "signaling follow-through"],
            "low": ["keeping commitment open-ended", "handling things casually", "deferring structure", "leaving execution loosely specified"],
        },
        "extraversion": {
            "high": ["initiating contact", "energizing the interaction", "moving the exchange outward", "actively engaging the other person"],
            "low": ["keeping the exchange contained", "minimizing outward engagement", "handling things quietly", "not expanding the interaction"],
        },
        "agreeableness": {
            "high": ["preserving harmony", "smoothing tension", "cooperative repair", "keeping the interaction relationally aligned"],
            "low": ["prioritizing blunt handling", "low-accommodation problem handling", "letting friction stand", "not centering harmony preservation"],
        },
        "neuroticism": {
            "high": ["seeking reassurance", "monitoring threat or risk", "self-protective regulation", "managing uncertainty defensively"],
            "low": ["handling things matter-of-factly", "keeping alarm low", "staying steady under uncertainty", "using low-defensive regulation"],
        },
    },
    "interpersonal_stance_pattern": {
        "openness": {
            "high": ["curious and receptive", "idea-open toward the other person", "willing to entertain unfamiliar perspectives"],
            "low": ["conventional and less exploratory", "more closed to novelty", "interpersonally grounded rather than idea-open"],
        },
        "conscientiousness": {
            "high": ["reliable and deliberate", "responsible toward the other person", "task-accountable in stance"],
            "low": ["casual and low-ownership", "less accountable in stance", "interpersonally relaxed about responsibility"],
        },
        "extraversion": {
            "high": ["engaged and socially forward", "contact-seeking", "animated and outward-facing"],
            "low": ["reserved and low-contact", "socially contained", "less outwardly engaging"],
        },
        "agreeableness": {
            "high": ["warm and accommodating", "affiliative", "harmony-managing"],
            "low": ["blunt and less accommodating", "low-harmony", "resistant or hard-edged"],
        },
        "neuroticism": {
            "high": ["tense or guarded", "reassurance-sensitive", "affectively vulnerable"],
            "low": ["calm and steady", "emotionally contained", "low-reactive in stance"],
        },
    },
    "expression_style_pattern": {
        "openness": {
            "high": ["exploratory wording", "vivid or associative phrasing", "imaginative expression"],
            "low": ["plain wording", "conventional or concrete phrasing", "grounded expression"],
        },
        "conscientiousness": {
            "high": ["structured wording", "precise expression", "orderly, well-shaped phrasing"],
            "low": ["loose wording", "under-structured phrasing", "casual, minimally shaped expression"],
        },
        "extraversion": {
            "high": ["expansive wording", "energetic phrasing", "socially expressive realization"],
            "low": ["restrained wording", "concise, low-display phrasing", "contained realization"],
        },
        "agreeableness": {
            "high": ["tactful wording", "softening phrasing", "relationally attentive expression"],
            "low": ["blunt wording", "sharp or minimally cushioning phrasing", "less relationally padded expression"],
        },
        "neuroticism": {
            "high": ["hedged wording", "tense or affect-laden phrasing", "uncertainty-marking expression"],
            "low": ["steady wording", "plain low-reactive phrasing", "emotionally even expression"],
        },
    },
}

VALID_TRAITS = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def require_torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def resolve_out_dir(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_cases_tag(path, explicit=None):
    if explicit:
        return explicit
    return os.path.splitext(os.path.basename(path))[0]


def parse_big5_focus(profile_text):
    match = re.search(r"Big Five focus:\s*([A-Za-z]+)\s*=\s*(high|low)\b", str(profile_text or ""), flags=re.I)
    if not match:
        return None, None
    trait = match.group(1).strip().lower()
    level = match.group(2).strip().lower()
    if trait in VALID_TRAITS and level in {"high", "low"}:
        return trait, level
    return None, None


def normalize_case_group_id(case_id):
    if isinstance(case_id, str):
        if "_NEG" in case_id:
            case_id = case_id.split("_NEG", 1)[0]
        if "_neg_" in case_id:
            case_id = case_id.split("_neg_", 1)[0]
        for suffix in ["_pos", "_neg_swap"]:
            if case_id.endswith(suffix):
                case_id = case_id[: -len(suffix)]
    return case_id


def get_group_id(row):
    if row.get("label") == 0 and row.get("source_case_id"):
        return normalize_case_group_id(row["source_case_id"])
    return row.get("group_id") or row.get("id") or normalize_case_group_id(row.get("case_id"))


def group_cases(rows):
    grouped = defaultdict(list)
    ordered_ids = []
    seen = set()
    for row in [r for r in rows if r.get("label") == 1]:
        gid = get_group_id(row)
        if gid not in seen:
            seen.add(gid)
            ordered_ids.append(gid)
        grouped[gid].append(row)
    for row in [r for r in rows if r.get("label") == 0]:
        gid = get_group_id(row)
        if gid in grouped:
            grouped[gid].append(row)
    return grouped, ordered_ids


def slice_grouped_cases(rows, start, end):
    grouped, ordered_ids = group_cases(rows)
    end = len(ordered_ids) if end is None else min(end, len(ordered_ids))
    selected_ids = ordered_ids[start:end]
    selected = []
    for gid in selected_ids:
        selected.extend(grouped[gid])
    return selected, selected_ids


def join_examples(items):
    return ", ".join(items)


def get_role_meanings(row, dim_key):
    trait, level = parse_big5_focus(row.get("profile"))
    if trait is None or level is None:
        return None
    prototypes = TRAIT_PATTERN_PROTOTYPES[dim_key]
    opposite_level = "low" if level == "high" else "high"
    return {
        "aligned": f"profile_aligned_pattern: {join_examples(prototypes[trait][level])}.",
        "indeterminate": DIMENSIONS[dim_key]["fallback_b"],
        "contradictory": f"profile_contradictory_pattern: {join_examples(prototypes[trait][opposite_level])}.",
    }


def deterministic_label_mapping(row, dim_key):
    roles = ["aligned", "indeterminate", "contradictory"]
    key = f"{row.get('case_id') or row.get('id') or get_group_id(row)}::{dim_key}"
    digest = hashlib.md5(key.encode("utf-8")).digest()
    shuffled_roles = sorted(roles, key=lambda role: digest[roles.index(role)])
    letters = ["A", "B", "C"]
    letter_to_role = dict(zip(letters, shuffled_roles))
    role_to_letter = {role: letter for letter, role in letter_to_role.items()}
    return letter_to_role, role_to_letter


def label_meanings_for_prompt(row, dim_key):
    role_meanings = get_role_meanings(row, dim_key)
    letter_to_role, role_to_letter = deterministic_label_mapping(row, dim_key)
    label_meanings = {
        letter: role_meanings[role]
        for letter, role in letter_to_role.items()
    }
    return label_meanings, letter_to_role, role_to_letter


def format_label_block(label_meanings, label_order):
    return "\n".join(f"- {label}: {label_meanings[label]}" for label in label_order)


def render_structured_prompt(task, dim_key, label_order, label_meanings, subject_blocks):
    label_list = ", ".join(label_order)
    rules = "\n".join([
        f"- Output exactly one label: {label_list}.",
        "- Do not explain.",
        f"- We read the model's internal {label_list} probabilities, so the next token should be the single best label.",
    ])
    subject = "\n\n".join(subject_blocks)
    return f"""Choose the best latent label for one consistency dimension.

Task:
{task}

Dimension:
{dim_key}

Labels:
{format_label_block(label_meanings, label_order)}

Rules:
{rules}

{subject}

Answer:"""


def build_inverse_prompt(row, dim_key, label_meanings):
    spec = DIMENSIONS[dim_key]
    return render_structured_prompt(
        spec["inverse_task"],
        dim_key,
        spec["label_order"],
        label_meanings,
        [f"Dialogue context:\n{row['context']}", f"Candidate response:\n{row['response']}"],
    )


def build_chat_text(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt + "\nAnswer:"


def sequence_logprob(model, tokenizer, prompt_text, completion_text, device):
    torch, F = require_torch()
    full_text = prompt_text + completion_text
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return float("-inf")
    with torch.no_grad():
        logits = model(full_ids).logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
    target_ids = full_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs[:, prompt_len - 1:].sum().item())


def label_distribution_from_logits(model, tokenizer, prompt_text, label_order, device):
    torch, F = require_torch()
    label_logps = {label: sequence_logprob(model, tokenizer, prompt_text, " " + label, device) for label in label_order}
    values = torch.tensor([label_logps[label] for label in label_order], dtype=torch.float32)
    probs = F.softmax(values, dim=0).tolist()
    return {
        "log_probs": {label: label_logps[label] for label in label_order},
        "probs": {label: prob for label, prob in zip(label_order, probs)},
        "argmax_label": label_order[int(torch.tensor(probs).argmax().item())],
    }


def score_dimension(inverse_probs, role_to_letter):
    aligned_letter = role_to_letter["aligned"]
    indeterminate_letter = role_to_letter["indeterminate"]
    contradictory_letter = role_to_letter["contradictory"]
    return {
        "aligned_label": aligned_letter,
        "indeterminate_label": indeterminate_letter,
        "contradictory_label": contradictory_letter,
        "inverse_aligned_prob": float(inverse_probs.get(aligned_letter, 0.0)),
        "inverse_indeterminate_prob": float(inverse_probs.get(indeterminate_letter, 0.0)),
        "inverse_contradictory_prob": float(inverse_probs.get(contradictory_letter, 0.0)),
    }


def make_model_and_tokenizer(model_name, dtype_name):
    torch, _ = require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = {"float16": torch.float16, "float32": torch.float32}.get(dtype_name, torch.bfloat16)
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=None)
    model.to(device)
    model.eval()
    return model, tokenizer, device, str(dtype).replace("torch.", "")


def process_rows(rows, model, tokenizer, device):
    results = []
    for row in tqdm(rows, desc="Big5 PRISM"):
        trait, level = parse_big5_focus(row.get("profile"))
        result = dict(row)
        result["group_id"] = get_group_id(row)
        result["parsed_profile_trait"] = trait
        result["parsed_profile_level"] = level
        if trait is None or level is None:
            result.update({"parse_ok": False, "parse_error": "profile_big5_focus_parse_failed", "scores": {}, "per_dim": {}})
            results.append(result)
            continue

        per_dim = {}
        for dim_key, spec in DIMENSIONS.items():
            label_meanings, letter_to_role, role_to_letter = label_meanings_for_prompt(row, dim_key)
            inverse_prompt = build_chat_text(tokenizer, build_inverse_prompt(row, dim_key, label_meanings))
            inverse_logits = label_distribution_from_logits(model, tokenizer, inverse_prompt, spec["label_order"], device)
            per_dim[dim_key] = {
                "display": spec["display"],
                "label_meanings": label_meanings,
                "letter_to_role": letter_to_role,
                "inverse_prompt": inverse_prompt,
                "inverse_logits": inverse_logits,
                **score_dimension(inverse_logits["probs"], role_to_letter),
            }

        dim_scores = {dim_key: per_dim[dim_key]["inverse_aligned_prob"] for dim_key in DIMENSIONS}
        prism = sum(dim_scores.values()) / len(dim_scores)
        result.update({
            "parse_ok": True,
            "schema": "big5_prism_profile_conditioned",
            "per_dim": per_dim,
            "scores": {
                "prism": prism,
                "task": dim_scores["agenda_pattern"],
                "stance": dim_scores["interpersonal_stance_pattern"],
                "style": dim_scores["expression_style_pattern"],
            },
        })
        results.append(result)
    return results


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def rank_auc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y == 1 and is_number(s)]
    neg = [s for y, s in zip(labels, scores) if y == 0 and is_number(s)]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def metric_rows(results, score_key):
    return [r for r in results if r.get("parse_ok") and is_number((r.get("scores") or {}).get(score_key))]


def pair_auc(results, score_key):
    grouped, _ = group_cases(results)
    wins = total = 0.0
    for rows in grouped.values():
        pos = [(r.get("scores") or {}).get(score_key) for r in rows if r.get("label") == 1]
        neg = [(r.get("scores") or {}).get(score_key) for r in rows if r.get("label") == 0]
        pos = [x for x in pos if is_number(x)]
        neg = [x for x in neg if is_number(x)]
        if not pos or not neg:
            continue
        p = max(pos)
        for n in neg:
            total += 1
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / total if total else None, int(total)


def strict_group_accuracy(results, score_key):
    grouped, _ = group_cases(results)
    hits = total = 0
    for rows in grouped.values():
        pos = [(r.get("scores") or {}).get(score_key) for r in rows if r.get("label") == 1]
        neg = [(r.get("scores") or {}).get(score_key) for r in rows if r.get("label") == 0]
        pos = [x for x in pos if is_number(x)]
        neg = [x for x in neg if is_number(x)]
        if not pos or not neg:
            continue
        total += 1
        hits += min(pos) > max(neg)
    return hits / total if total else None


def summarize(results, dataset_name, selected_group_ids):
    valid = metric_rows(results, "prism")
    labels = [int(r["label"]) for r in valid]
    scores = [r["scores"]["prism"] for r in valid]
    p_auc, pair_n = pair_auc(valid, "prism")
    return {
        "dataset": dataset_name,
        "score_name": "prism",
        "score_definition": "mean inverse aligned posterior over task, stance, and style dimensions",
        "n_total": len(results),
        "n_valid": len(valid),
        "n_parse_failed": len(results) - len(valid),
        "n_selected_groups": len(selected_group_ids),
        "auc": rank_auc(labels, scores),
        "pair_auc": p_auc,
        "strict_group_acc": strict_group_accuracy(valid, "prism"),
        "pair_n": pair_n,
        "per_dimension": {
            name: {
                "display": DIMENSIONS[name]["display"],
                "mean_positive": sum(r["scores"][short] for r in valid if r["label"] == 1) / max(1, sum(1 for r in valid if r["label"] == 1)),
                "mean_negative": sum(r["scores"][short] for r in valid if r["label"] == 0) / max(1, sum(1 for r in valid if r["label"] == 0)),
            }
            for name, short in [
                ("agenda_pattern", "task"),
                ("interpersonal_stance_pattern", "stance"),
                ("expression_style_pattern", "style"),
            ]
        },
    }


def run_dataset(dataset_key, args, model, tokenizer, device, resolved_dtype):
    spec = DATASET_PRESETS[dataset_key]
    input_path = resolve_path(args.cases_file or spec["file"])
    rows = load_json(input_path)
    selected_rows, selected_group_ids = slice_grouped_cases(rows, args.start, args.end)
    results = process_rows(selected_rows, model, tokenizer, device)
    analysis = summarize(results, spec["display"], selected_group_ids)
    analysis["config"] = {
        "dataset_key": dataset_key,
        "dataset_display": spec["display"],
        "input_path": input_path,
        "model_preset": args.model_preset,
        "model_name": args.model_name or MODEL_PRESETS[args.model_preset],
        "start": args.start,
        "end": args.end,
        "dtype": resolved_dtype,
        "device": str(device),
        "dimensions": {k: v["display"] for k, v in DIMENSIONS.items()},
        "profile_conditioning": "parse Big Five focus from current profile text",
    }
    return results, analysis, infer_cases_tag(input_path, args.cases_tag)


def main():
    parser = argparse.ArgumentParser(description="Unified Big5 PRISM evaluation for EASY and HARD datasets.")
    parser.add_argument("--dataset", choices=["easy", "hard", "all"], default="easy")
    parser.add_argument("--cases_file", type=str, default=None, help="Optional override; only valid for a single dataset run.")
    parser.add_argument("--cases_tag", type=str, default=None)
    parser.add_argument("--model_preset", choices=sorted(MODEL_PRESETS.keys()), default="qwen25")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--out_dir", type=str, default="big5_prism_unified")
    args = parser.parse_args()

    if args.dataset == "all" and args.cases_file:
        raise ValueError("--cases_file can only be used when --dataset is easy or hard.")

    out_dir = resolve_out_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model_name = args.model_name or MODEL_PRESETS[args.model_preset]
    model, tokenizer, device, resolved_dtype = make_model_and_tokenizer(model_name, args.dtype)

    datasets = ["easy", "hard"] if args.dataset == "all" else [args.dataset]
    manifest = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for dataset_key in datasets:
        results, analysis, cases_tag = run_dataset(dataset_key, args, model, tokenizer, device, resolved_dtype)
        stem = f"big5_{dataset_key}_{cases_tag}_prism_{args.model_preset}_{args.start}-{args.end or 'end'}_{timestamp}"
        results_path = os.path.join(out_dir, f"{stem}_results.json")
        analysis_path = os.path.join(out_dir, f"{stem}_analysis.json")
        dump_json(results, results_path)
        dump_json(analysis, analysis_path)
        manifest.append({
            "dataset": DATASET_PRESETS[dataset_key]["display"],
            "results_path": results_path,
            "analysis_path": analysis_path,
            "auc": analysis["auc"],
            "pair_auc": analysis["pair_auc"],
            "strict_group_acc": analysis["strict_group_acc"],
            "n_valid": analysis["n_valid"],
        })

    print(json.dumps({"runs": manifest}, indent=2))


if __name__ == "__main__":
    main()
