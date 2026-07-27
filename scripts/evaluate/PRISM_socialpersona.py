#!/usr/bin/env python3
"""Clean Social-Persona PRISM evaluation.

    prism = mean_d q_d(cue_evidence | c, r)

where d ranges over task framing, interpersonal stance, and linguistic style.
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

DATASET_PRESET = "data/social_persona.json"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DIMENSIONS = {
    "agenda_pattern": {
        "display": "Task",
        "label_order": ["A", "B", "C"],
        "inverse_task": "Given the dialogue context and candidate response, judge what evidence the response expresses on the Agenda pattern dimension according to the label definitions.",
        "base": {
            "aligned": "cue_evidence: The response agenda expresses the provided personality cues for this dimension.",
            "indeterminate": "weak_or_generic: The response agenda is weakly marked, mixed, generic, or underspecified.",
            "contradictory": "other_state: The response agenda centers on priorities or motives outside the provided cue set.",
        },
    },
    "interpersonal_stance_pattern": {
        "display": "Stance",
        "label_order": ["A", "B", "C"],
        "inverse_task": "Given the dialogue context and candidate response, judge what evidence the response expresses on the Interpersonal stance pattern dimension according to the label definitions.",
        "base": {
            "aligned": "cue_evidence: The response's interpersonal stance expresses the provided personality cues for this dimension.",
            "indeterminate": "weak_or_generic: The response's interpersonal stance is weakly marked, mixed, flat, generic, or hard to read.",
            "contradictory": "other_state: The response's interpersonal stance centers on social or emotional cues outside the provided cue set.",
        },
    },
    "expression_style_pattern": {
        "display": "Style",
        "label_order": ["A", "B", "C"],
        "inverse_task": "Given the dialogue context and candidate response, judge what evidence the response expresses on the Expression style pattern dimension according to the label definitions.",
        "base": {
            "aligned": "cue_evidence: The response's wording, register, rhythm, or nonverbal realization expresses the provided dialogue-style cues for this dimension.",
            "indeterminate": "weak_or_generic: The response's wording is ordinary, weakly marked, mixed, generic, or underspecified.",
            "contradictory": "other_state: The response's wording, register, rhythm, or realization centers on manner cues outside the provided cue set.",
        },
    },
}


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


def infer_cases_tag(path, explicit=None):
    if explicit:
        return explicit
    return os.path.splitext(os.path.basename(path))[0]


def get_group_id(row):
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


def compact_text(text, max_chars=360):
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:") + "..."


def profile_section(profile, section_name):
    pattern = rf"{re.escape(section_name)}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:|$)"
    match = re.search(pattern, profile or "", flags=re.S)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def stable_row_id(row):
    for key in ("case_id", "id", "source_case_id", "group_id"):
        value = row.get(key)
        if value:
            return str(value)
    base = f"{row.get('profile', '')}|{row.get('context', '')}|{row.get('response', '')}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def deterministic_label_mapping(row, dim_key):
    roles = ["aligned", "indeterminate", "contradictory"]
    key = f"{stable_row_id(row)}::{dim_key}"
    digest = hashlib.md5(key.encode("utf-8")).digest()
    shuffled_roles = sorted(roles, key=lambda role: digest[roles.index(role)])
    letters = ["A", "B", "C"]
    letter_to_role = dict(zip(letters, shuffled_roles))
    role_to_letter = {role: letter for letter, role in letter_to_role.items()}
    return letter_to_role, role_to_letter


def role_meanings(row, dim_key):
    profile = row.get("profile", "")
    personality = profile_section(profile, "Personality")
    dialogue_style = profile_section(profile, "Dialogue Style")
    spec = DIMENSIONS[dim_key]

    if dim_key == "agenda_pattern":
        cue = compact_text(personality or profile, 360)
        return {
            "aligned": f"cue_evidence: The response agenda expresses the target profile's characteristic priorities, motives, or recurring concerns. Profile cues: {cue}",
            "indeterminate": spec["base"]["indeterminate"],
            "contradictory": spec["base"]["contradictory"],
        }
    if dim_key == "interpersonal_stance_pattern":
        cue = compact_text(personality or profile, 360)
        return {
            "aligned": f"cue_evidence: The response's interpersonal stance expresses the target profile's characteristic social orientation or emotional posture. Profile cues: {cue}",
            "indeterminate": spec["base"]["indeterminate"],
            "contradictory": spec["base"]["contradictory"],
        }
    if dim_key == "expression_style_pattern":
        cue = compact_text(dialogue_style or personality or profile, 380)
        return {
            "aligned": f"cue_evidence: The response's wording, register, rhythm, or nonverbal realization expresses the target profile's dialogue-style cues. Style cues: {cue}",
            "indeterminate": spec["base"]["indeterminate"],
            "contradictory": spec["base"]["contradictory"],
        }
    return spec["base"]


def label_meanings_for_prompt(row, dim_key):
    meanings = role_meanings(row, dim_key)
    letter_to_role, role_to_letter = deterministic_label_mapping(row, dim_key)
    label_meanings = {letter: meanings[role] for letter, role in letter_to_role.items()}
    return label_meanings, letter_to_role, role_to_letter


def format_label_block(label_order, label_meanings):
    return "\n".join(f"- {label}: {label_meanings[label]}" for label in label_order)


def build_inverse_prompt(row, dim_key, label_meanings):
    spec = DIMENSIONS[dim_key]
    label_order = spec["label_order"]
    label_list = ", ".join(label_order)
    return f"""Choose the best latent label for one consistency dimension.

Task:
{spec['inverse_task']}

Dimension:
{dim_key}

Labels:
{format_label_block(label_order, label_meanings)}

Rules:
- Output exactly one label: {label_list}.
- Do not explain.
- We read the model's internal label probabilities, so the next token should be the single best label.

Dialogue context:
{row['context']}

Candidate response:
{row['response']}

Answer:"""


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
    aligned = role_to_letter["aligned"]
    indeterminate = role_to_letter["indeterminate"]
    contradictory = role_to_letter["contradictory"]
    return {
        "aligned_label": aligned,
        "indeterminate_label": indeterminate,
        "contradictory_label": contradictory,
        "inverse_evidence_prob": float(inverse_probs.get(aligned, 0.0)),
        "inverse_generic_prob": float(inverse_probs.get(indeterminate, 0.0)),
        "inverse_other_prob": float(inverse_probs.get(contradictory, 0.0)),
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
    for row in tqdm(rows, desc="Social-Persona PRISM"):
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

        dim_scores = {dim: per_dim[dim]["inverse_evidence_prob"] for dim in DIMENSIONS}
        prism = sum(dim_scores.values()) / len(dim_scores)
        result = dict(row)
        result.update({
            "group_id": get_group_id(row),
            "schema": "socialbench_prism_inverse_evidence",
            "parse_ok": True,
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


def summarize(results, selected_group_ids):
    valid = metric_rows(results, "prism")
    labels = [int(r["label"]) for r in valid]
    scores = [r["scores"]["prism"] for r in valid]
    p_auc, pair_n = pair_auc(valid, "prism")
    return {
        "dataset": "Social-Persona",
        "score_name": "prism",
        "score_definition": "mean inverse cue-evidence posterior over task, stance, and style dimensions",
        "n_total": len(results),
        "n_valid": len(valid),
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


def main():
    parser = argparse.ArgumentParser(description="Clean Social-Persona PRISM evaluation.")
    parser.add_argument("--cases_file", type=str, default=DATASET_PRESET)
    parser.add_argument("--cases_tag", type=str, default=None)
    parser.add_argument("--model_preset", choices=sorted(MODEL_PRESETS.keys()), default="qwen25")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--out_dir", type=str, default="socialbench_prism")
    args = parser.parse_args()

    input_path = resolve_path(args.cases_file)
    out_dir = resolve_out_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = normalize_rows(load_json(input_path))
    selected_rows, selected_group_ids = slice_grouped_cases(rows, args.start, args.end)
    model_name = args.model_name or MODEL_PRESETS[args.model_preset]
    model, tokenizer, device, resolved_dtype = make_model_and_tokenizer(model_name, args.dtype)
    results = process_rows(selected_rows, model, tokenizer, device)
    analysis = summarize(results, selected_group_ids)
    analysis["config"] = {
        "input_path": input_path,
        "cases_tag": infer_cases_tag(input_path, args.cases_tag),
        "model_preset": args.model_preset,
        "model_name": model_name,
        "start": args.start,
        "end": args.end,
        "dtype": resolved_dtype,
        "device": str(device),
        "dimensions": {k: v["display"] for k, v in DIMENSIONS.items()},
        "label_order": "deterministically shuffled A/B/C per case and dimension",
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cases_tag = infer_cases_tag(input_path, args.cases_tag)
    stem = f"socialbench_{cases_tag}_prism_{args.model_preset}_{args.start}-{args.end or 'end'}_{timestamp}"
    results_path = os.path.join(out_dir, f"{stem}_results.json")
    analysis_path = os.path.join(out_dir, f"{stem}_analysis.json")
    dump_json(results, results_path)
    dump_json(analysis, analysis_path)
    print(json.dumps({
        "results_path": results_path,
        "analysis_path": analysis_path,
        "auc": analysis["auc"],
        "pair_auc": analysis["pair_auc"],
        "strict_group_acc": analysis["strict_group_acc"],
        "n_valid": analysis["n_valid"],
    }, indent=2))


if __name__ == "__main__":
    main()
