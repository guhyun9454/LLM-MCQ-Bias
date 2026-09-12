# eval_clm.py
# -*- coding: utf-8 -*-

import os
import sys
import gc
import json
import copy
import logging
import random
import math
import time
import atexit
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
from transformers import logging as hf_logging

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_clm_utils import (
    build_results_dir,
    parse_arguments,
    prepare_eval,
    select_api_probe_subjects,
)
from eval_clm_online import (
    _recall_std,
    _run_cyclic_random_fraction,
    _run_cyclic_random_fraction_with_preds,
    _run_online_avggap_policy,
    _run_online_avggap_policy_with_preds,
    _run_online_avggap_policy_with_stats,
    _run_online_sqrt_policy,
    _run_online_sqrt_policy_lowconf_update,
    _run_online_sqrt_policy_with_preds,
    _run_online_sqrt_policy_with_stats,
    _run_online_switch_cyclic_with_preds,
    _run_online_switch_cyclic_with_stats,
    _run_online_th1_quantile_th2_from_th1_rule,
    _run_online_th1_quantile_th2_from_th1_rule_with_preds,
    _run_online_th1_quantile_th2_from_th1_rule_with_stats,
    _run_online_top2flip_policy,
    _run_online_top2flip_policy_with_preds,
    _run_online_top2flip_policy_with_stats,
    _run_prefix_cyclic_postfix_base,
)
from eval_clm_plots import _plot_three_curves_acc_recall_std
from eval_clm_reporting import _log_baseline_report, _log_named_report
from api_inference import CommercialAPIClient, OnlinePercentileRouter

from utils import (
    _orange, _blue, _purple,
    eval_all_samples,
    get_accuracy,
    get_bootstrap_accuracy_std,
    save_results,
    patch_open,
)

# PriDe (PRIDE) helper: estimates option-token prior
from debias_utils import simple as debias_simple

# -------------------------
# [FIX] Safe NVML init
# -------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

logger = logging.getLogger(__name__)

PRIMARY_OURS_LABEL = "th1/sqrt2"
LEGACY_OURS_LABEL = "th1/2"
EMPIRICAL_PRIDE_LABEL = "empirical_pride_primary"


def _rule_th1_half(th1_val: float) -> float:
    return float(th1_val) / 2.0


def _rule_th1_sqrt2(th1_val: float) -> float:
    return float(th1_val) / math.sqrt(2.0)


def _gap_of_distribution(probs: np.ndarray) -> float:
    vals = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    if vals.size <= 1:
        return 0.0
    return float(vals[0] - vals[1])


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(xa) & np.isfinite(ya)
    if np.sum(m) < 2:
        return float("nan")
    xa = xa[m]
    ya = ya[m]
    if float(np.std(xa)) <= 1e-12 or float(np.std(ya)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _build_sigma_analysis_record(
    subject: str,
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    cyclic_gap_mean: np.ndarray,
    cyclic_gap_std: np.ndarray,
    flip_mask: np.ndarray,
) -> Dict[str, float]:
    dc = np.asarray(default_conf, dtype=np.float64)
    mc = np.asarray(mean_conf, dtype=np.float64)
    mu = np.asarray(cyclic_gap_mean, dtype=np.float64)
    sg = np.asarray(cyclic_gap_std, dtype=np.float64)
    flip = np.asarray(flip_mask, dtype=bool)

    resid_single = dc - mu
    resid_two_view = mc - mu
    sigma_single = float(np.std(resid_single)) if resid_single.size > 0 else float("nan")
    sigma_two_view = float(np.std(resid_two_view)) if resid_two_view.size > 0 else float("nan")
    sigma_ratio = float(sigma_two_view / sigma_single) if np.isfinite(sigma_single) and sigma_single > 1e-12 else float("nan")

    q_low = float(np.quantile(dc, 0.30)) if dc.size > 0 else 0.0
    q_high = float(np.quantile(dc, 0.70)) if dc.size > 0 else 0.0
    low_conf_mask = dc <= q_low
    high_conf_mask = dc >= q_high

    q_sigma_low = float(np.quantile(sg, 0.30)) if sg.size > 0 else 0.0
    q_sigma_high = float(np.quantile(sg, 0.70)) if sg.size > 0 else 0.0
    low_sigma_mask = sg <= q_sigma_low
    high_sigma_mask = sg >= q_sigma_high

    def _mean_or_nan(arr: np.ndarray, mask: np.ndarray) -> float:
        if arr.size == 0 or np.sum(mask) == 0:
            return float("nan")
        return float(np.mean(arr[mask]))

    flip_float = flip.astype(np.float64)
    return {
        "subject": str(subject),
        "n": int(dc.size),
        "default_gap_mean": float(np.mean(dc)) if dc.size > 0 else float("nan"),
        "two_view_gap_mean": float(np.mean(mc)) if mc.size > 0 else float("nan"),
        "cyclic_gap_mean": float(np.mean(mu)) if mu.size > 0 else float("nan"),
        "sigma_mean": float(np.mean(sg)) if sg.size > 0 else float("nan"),
        "sigma_std": float(np.std(sg)) if sg.size > 0 else float("nan"),
        "sigma_single": sigma_single,
        "sigma_two_view": sigma_two_view,
        "sigma_ratio": sigma_ratio,
        "sigma_ratio_target": float(1.0 / math.sqrt(2.0)),
        "corr_default_gap_sigma": _safe_corr(dc, sg),
        "corr_flip_sigma": _safe_corr(flip_float, sg),
        "sigma_low_conf_mean": _mean_or_nan(sg, low_conf_mask),
        "sigma_high_conf_mean": _mean_or_nan(sg, high_conf_mask),
        "flip_low_conf": _mean_or_nan(flip_float, low_conf_mask),
        "flip_high_conf": _mean_or_nan(flip_float, high_conf_mask),
        "flip_low_sigma": _mean_or_nan(flip_float, low_sigma_mask),
        "flip_high_sigma": _mean_or_nan(flip_float, high_sigma_mask),
    }

def _pride_correct_row(row: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """PriDe correction: divide by prior then renormalize."""
    r = np.asarray(row, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64)
    adj = r / (pr + eps)
    adj = adj / (adj.sum() + eps)
    return adj


def _stable_u32_seed(s: str, base_seed: int = 0) -> int:
    return (int(zlib.crc32(s.encode("utf-8"))) + int(base_seed)) & 0xFFFFFFFF


def _estimate_pride_prior_random_prefix_mean(
    per_sample_probs: List[np.ndarray],
    cyclic_indices: List[int],
    k: int,
    prefix_ratio: float,
    seed: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Estimate global prior over option letters using a random prefix subset (ratio),
    WITHOUT EMA: compute per-sample prior_i and take their mean.
    """
    N = len(per_sample_probs)
    if N <= 0:
        prior = np.ones((k,), dtype=np.float64) / float(k)
        return prior, {"N": 0, "m": 0, "used": 0, "ratio": float(prefix_ratio), "seed": int(seed), "prefix_ids": []}

    ratio = float(max(0.0, min(1.0, prefix_ratio)))
    m = int(max(1, int(round(N * ratio))))

    rng = np.random.default_rng(int(seed))
    prefix_ids = rng.choice(np.arange(N, dtype=np.int64), size=m, replace=False)
    prefix_ids = [int(x) for x in prefix_ids.tolist()]

    priors = []
    used = 0
    for i in prefix_ids:
        ps = np.asarray(per_sample_probs[i], dtype=np.float64)
        observed = np.asarray([ps[j] for j in cyclic_indices], dtype=np.float64)  # (k,k)
        try:
            _, _, prior_i = debias_simple(observed)
        except Exception:
            continue
        prior_i = np.asarray(prior_i, dtype=np.float64)
        prior_i = prior_i / (prior_i.sum() + eps)
        priors.append(prior_i)
        used += 1

    if len(priors) == 0:
        prior = np.ones((k,), dtype=np.float64) / float(k)
    else:
        prior = np.mean(np.asarray(priors, dtype=np.float64), axis=0)
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior / (prior.sum() + eps)

    meta = {"N": int(N), "m": int(m), "used": int(used), "ratio": float(ratio), "seed": int(seed), "prefix_ids": prefix_ids}
    return prior, meta


def _estimate_pride_prior_random_prefix_details(
    per_sample_probs: List[np.ndarray],
    cyclic_indices: List[int],
    k: int,
    prefix_ratio: float,
    seed: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns (mean_prior, per_sample_priors, meta) for the sampled random prefix.
    per_sample_priors has shape (used, k).
    """
    N = len(per_sample_probs)
    if N <= 0:
        prior = np.ones((k,), dtype=np.float64) / float(k)
        empty = np.zeros((0, k), dtype=np.float64)
        meta = {"N": 0, "m": 0, "used": 0, "ratio": float(prefix_ratio), "seed": int(seed), "prefix_ids": []}
        return prior, empty, meta

    ratio = float(max(0.0, min(1.0, prefix_ratio)))
    m = int(max(1, int(round(N * ratio))))
    rng = np.random.default_rng(int(seed))
    prefix_ids = rng.choice(np.arange(N, dtype=np.int64), size=m, replace=False)
    prefix_ids = [int(x) for x in prefix_ids.tolist()]

    priors = []
    used_ids = []
    for i in prefix_ids:
        ps = np.asarray(per_sample_probs[i], dtype=np.float64)
        observed = np.asarray([ps[j] for j in cyclic_indices], dtype=np.float64)
        try:
            _, _, prior_i = debias_simple(observed)
        except Exception:
            continue
        prior_i = np.asarray(prior_i, dtype=np.float64)
        prior_i = prior_i / (prior_i.sum() + eps)
        priors.append(prior_i)
        used_ids.append(int(i))

    if len(priors) == 0:
        prior = np.ones((k,), dtype=np.float64) / float(k)
        priors_arr = np.zeros((0, k), dtype=np.float64)
    else:
        priors_arr = np.asarray(priors, dtype=np.float64)
        prior = np.mean(priors_arr, axis=0)
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior / (prior.sum() + eps)

    meta = {
        "N": int(N),
        "m": int(m),
        "used": int(len(used_ids)),
        "ratio": float(ratio),
        "seed": int(seed),
        "prefix_ids": used_ids,
    }
    return prior, priors_arr, meta


def _estimate_empirical_pride_bank(
    per_sample_probs: List[np.ndarray],
    cyclic_indices: List[int],
    k: int,
    prefix_ratio: float,
    seed: int,
    logit_delta: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns (mean_prior, mu_hat, residual_bank, meta).
    residual_bank has shape (used, k); each row is centered and sums to ~0.
    """
    mean_prior, priors_arr, meta = _estimate_pride_prior_random_prefix_details(
        per_sample_probs=per_sample_probs,
        cyclic_indices=cyclic_indices,
        k=k,
        prefix_ratio=prefix_ratio,
        seed=seed,
        eps=max(float(logit_delta), 1e-18),
    )
    if priors_arr.size == 0:
        mu_hat = np.zeros((k,), dtype=np.float64)
        residual_bank = np.zeros((1, k), dtype=np.float64)
        return mean_prior, mu_hat, residual_bank, meta

    delta = max(float(logit_delta), 1e-18)
    log_priors = np.log(priors_arr + delta)
    centered_logits = log_priors - np.mean(log_priors, axis=1, keepdims=True)
    mu_hat = np.mean(centered_logits, axis=0)
    residual_bank = centered_logits - mu_hat[None, :]
    return mean_prior, np.asarray(mu_hat, dtype=np.float64), np.asarray(residual_bank, dtype=np.float64), meta


def _estimate_logistic_normal_pride_bank(
    per_sample_probs: List[np.ndarray],
    cyclic_indices: List[int],
    k: int,
    prefix_ratio: float,
    seed: int,
    logit_delta: float = 1e-12,
    mc_samples: int = 64,
    shrinkage_lambda: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns (mean_prior, mu_hat, residual_samples, covariance, meta).
    residual_samples has shape (mc_samples, k), centered to sum ~0 per row.
    """
    mean_prior, priors_arr, meta = _estimate_pride_prior_random_prefix_details(
        per_sample_probs=per_sample_probs,
        cyclic_indices=cyclic_indices,
        k=k,
        prefix_ratio=prefix_ratio,
        seed=seed,
        eps=max(float(logit_delta), 1e-18),
    )
    delta = max(float(logit_delta), 1e-18)
    n_mc = max(1, int(mc_samples))
    lam = min(max(float(shrinkage_lambda), 0.0), 1.0)

    if priors_arr.size == 0:
        mu_hat = np.zeros((k,), dtype=np.float64)
        residual_samples = np.zeros((n_mc, k), dtype=np.float64)
        covariance = np.zeros((k, k), dtype=np.float64)
        meta = dict(meta)
        meta.update({"residual_model": "logistic_normal", "mc_samples": int(n_mc), "cov_shrinkage": float(lam), "used_priors": 0})
        return mean_prior, mu_hat, residual_samples, covariance, meta

    log_priors = np.log(priors_arr + delta)
    centered_logits = log_priors - np.mean(log_priors, axis=1, keepdims=True)
    mu_hat = np.mean(centered_logits, axis=0)
    residual_bank = centered_logits - mu_hat[None, :]

    if residual_bank.shape[0] <= 1:
        covariance = np.zeros((k, k), dtype=np.float64)
    else:
        covariance = (residual_bank.T @ residual_bank) / float(residual_bank.shape[0] - 1)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)

    projector = np.eye(k, dtype=np.float64) - np.ones((k, k), dtype=np.float64) / float(k)
    sigma2 = float(np.trace(covariance)) / float(max(k - 1, 1))
    shrunk_cov = (1.0 - lam) * covariance + lam * sigma2 * projector
    shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)
    shrunk_cov = projector @ shrunk_cov @ projector
    shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)

    eigvals, eigvecs = np.linalg.eigh(shrunk_cov)
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64), 0.0)
    shrunk_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)

    rng = np.random.default_rng(int(seed) + 99173)
    if float(np.max(eigvals)) <= 1e-18:
        residual_samples = np.zeros((n_mc, k), dtype=np.float64)
    else:
        gaussian = rng.normal(size=(n_mc, k))
        transform = eigvecs @ np.diag(np.sqrt(eigvals))
        residual_samples = gaussian @ transform.T
        residual_samples = residual_samples - np.mean(residual_samples, axis=1, keepdims=True)

    meta = dict(meta)
    meta.update({
        "residual_model": "logistic_normal",
        "mc_samples": int(n_mc),
        "cov_shrinkage": float(lam),
        "used_priors": int(priors_arr.shape[0]),
    })
    return mean_prior, np.asarray(mu_hat, dtype=np.float64), np.asarray(residual_samples, dtype=np.float64), np.asarray(shrunk_cov, dtype=np.float64), meta


def _extract_question_from_user_prompt(user_prompt: str) -> str:
    marker = "\nOptions:\n"
    if user_prompt.startswith("Question: ") and marker in user_prompt:
        return user_prompt[len("Question: "): user_prompt.index(marker)]
    return str(user_prompt)


def _build_option_user_prompt(question: str, options: List[str], option_ids: List[str]) -> str:
    return (
        f"Question: {question.strip()}\nOptions:\n"
        + "\n".join(f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options))
        + "\nAnswer:"
    )


def _invert_slot_to_content_perm(slot_to_content: Tuple[int, ...]) -> np.ndarray:
    inv = np.zeros(len(slot_to_content), dtype=np.int64)
    for slot_idx, content_idx in enumerate(slot_to_content):
        inv[int(content_idx)] = int(slot_idx)
    return inv


def _content_to_slot_assignment_to_perm(content_to_slot: List[int]) -> Tuple[int, ...]:
    slot_to_content = [0] * len(content_to_slot)
    for content_idx, slot_idx in enumerate(content_to_slot):
        slot_to_content[int(slot_idx)] = int(content_idx)
    return tuple(slot_to_content)


def _search_latin_assignment(
    k: int,
    allowed_slots: List[List[int]],
    content_idx: int,
    used_slots: set,
    current: List[int],
) -> Optional[List[int]]:
    if content_idx >= k:
        return list(current)
    for slot_idx in allowed_slots[content_idx]:
        if slot_idx in used_slots:
            continue
        current[content_idx] = int(slot_idx)
        used_slots.add(int(slot_idx))
        out = _search_latin_assignment(k, allowed_slots, content_idx + 1, used_slots, current)
        if out is not None:
            return out
        used_slots.remove(int(slot_idx))
        current[content_idx] = -1
    return None


def _build_targeted_latin_schedule(k: int, top1_idx: int, runner_idx: int) -> List[Tuple[int, ...]]:
    """
    Returns slot->content permutations. The first is identity; the second swaps
    the two targeted contents and rotates the remainder; later stages are filled
    deterministically so each content visits every slot exactly once.
    """
    if k <= 1:
        return [tuple(range(k))]

    schedules_content_to_slot: List[List[int]] = [list(range(k))]
    if top1_idx == runner_idx:
        runner_idx = (int(top1_idx) + 1) % int(k)

    remaining = [idx for idx in range(k) if idx not in (int(top1_idx), int(runner_idx))]
    targeted = [-1] * k
    targeted[int(top1_idx)] = int(runner_idx)
    targeted[int(runner_idx)] = int(top1_idx)
    if remaining:
        for offset, content_idx in enumerate(remaining):
            targeted[int(content_idx)] = int(remaining[(offset + 1) % len(remaining)])
    schedules_content_to_slot.append(targeted)

    for _stage in range(2, k):
        used_by_content = [set() for _ in range(k)]
        for sched in schedules_content_to_slot:
            for content_idx, slot_idx in enumerate(sched):
                used_by_content[content_idx].add(int(slot_idx))
        allowed_slots = [sorted(set(range(k)) - used_by_content[i]) for i in range(k)]
        candidate = _search_latin_assignment(
            k=k,
            allowed_slots=allowed_slots,
            content_idx=0,
            used_slots=set(),
            current=[-1] * k,
        )
        if candidate is None:
            raise RuntimeError(f"Failed to complete Latin schedule for k={k}, top1={top1_idx}, runner={runner_idx}")
        schedules_content_to_slot.append(candidate)

    return [_content_to_slot_assignment_to_perm(sched) for sched in schedules_content_to_slot]


def _build_incremental_cyclic_schedule(
    k: int,
    top1_idx: int,
    runner_idx: int,
    mode: str,
    seed: int,
) -> List[Tuple[int, ...]]:
    """
    Returns slot->content permutations restricted to cyclic rotations.
    identity is always first. Remaining cyclic shifts are either:
    - cyclic_random: random order
    - cyclic_targeted: first choose the shift that moves runner_idx into top1_idx's slot,
      then randomize the remaining cyclic shifts.
    """
    if k <= 1:
        return [tuple(range(k))]

    mode = str(mode or "cyclic_random").strip().lower()
    shifts = list(range(1, int(k)))
    rng = np.random.default_rng(int(seed))

    if mode == "cyclic_targeted":
        preferred_shift = (int(runner_idx) - int(top1_idx)) % int(k)
        if preferred_shift == 0:
            preferred_shift = 1
        remaining = [s for s in shifts if int(s) != int(preferred_shift)]
        rng.shuffle(remaining)
        ordered_shifts = [int(preferred_shift)] + remaining
    else:
        ordered_shifts = list(shifts)
        rng.shuffle(ordered_shifts)

    schedule = [tuple(range(k))]
    for shift in ordered_shifts:
        schedule.append(tuple((slot_idx + int(shift)) % int(k) for slot_idx in range(int(k))))
    return schedule


def _candidate_relative_action_sequences() -> List[Tuple[str, Tuple[Tuple[int, int], ...]]]:
    return [
        ("omega1", ((2, 1), (3, 1), (3, 2), (1, 3))),
        ("omega2", ((2, 1), (3, 2), (3, 1), (1, 3))),
        ("omega3", ((2, 1), (1, 2), (3, 1), (3, 2))),
        ("omega4", ((2, 1), (2, 3), (3, 1), (1, 3))),
        ("omega5", ((2, 1), (1, 3), (3, 1), (3, 2))),
    ]


def _format_relative_action_sequence(actions: Tuple[Tuple[int, int], ...]) -> List[str]:
    return [f"A_{int(u)}to{int(v)}" for (u, v) in actions]


def _build_relative_action_cyclic_schedule(
    k: int,
    initial_rank: List[int],
    actions: Tuple[Tuple[int, int], ...],
) -> Tuple[List[int], List[Tuple[int, ...]]]:
    """
    Build a cyclic-only schedule from relative rank actions.
    initial_rank is a list of content indices ordered by the stage-1 corrected posterior.
    """
    if k <= 1:
        return [0], [tuple(range(k))]

    rank = [int(x) for x in initial_rank[: max(3, int(k))]]
    shifts = [0]
    used = {0}
    for (u_rank, v_rank) in actions:
        u_idx = int(u_rank) - 1
        v_idx = int(v_rank) - 1
        if u_idx < 0 or v_idx < 0 or u_idx >= len(rank) or v_idx >= len(rank):
            continue
        shift = int((rank[v_idx] - rank[u_idx]) % int(k))
        if shift == 0 or shift in used:
            continue
        shifts.append(int(shift))
        used.add(int(shift))
    for shift in range(1, int(k)):
        if int(shift) in used:
            continue
        shifts.append(int(shift))
        used.add(int(shift))
    schedule = [tuple((slot_idx + int(shift)) % int(k) for slot_idx in range(int(k))) for shift in shifts]
    return shifts, schedule


def _select_best_relative_cyclic_sequence(
    sample_indices: List[int],
    per_sample_probs: List[np.ndarray],
    cyclic_indices: List[int],
    cyc_perms: List[Tuple[int, ...]],
    mu_hat: np.ndarray,
    residual_bank: np.ndarray,
    labels_idx: List[int],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Pick the best cyclic action sequence on a small validation subset using
    soft low-confidence weighted NLL gain.
    """
    k = int(len(cyc_perms))
    candidates = _candidate_relative_action_sequences()
    if k <= 1 or len(sample_indices) <= 0:
        fallback_name, fallback_actions = candidates[0]
        return {
            "selection_policy": "initial_rank_relative_bruteforce_cyclic_top3",
            "selected_sequence_name": str(fallback_name),
            "selected_action_sequence": _format_relative_action_sequence(fallback_actions),
            "candidate_scores": [{"name": str(name), "actions": _format_relative_action_sequence(actions), "score": float("nan")} for name, actions in candidates],
            "n_validation": int(len(sample_indices)),
        }

    identity_perm = tuple(range(int(k)))
    selection_rows: List[Dict[str, Any]] = []
    for cand_name, cand_actions in candidates:
        stage_num = np.zeros((max(k - 1, 1),), dtype=np.float64)
        stage_den = np.zeros((max(k - 1, 1),), dtype=np.float64)
        for sample_idx in sample_indices:
            sample_idx_i = int(sample_idx)
            base_row = np.asarray(per_sample_probs[sample_idx_i][cyclic_indices[0]], dtype=np.float64)
            base_posteriors, _, _ = _compute_empirical_stage_posteriors(
                stage_probs=base_row.reshape(1, -1),
                slot_to_content_schedule=[identity_perm],
                mu_hat=mu_hat,
                residual_bank=residual_bank,
            )
            stage1_post = np.asarray(base_posteriors[0], dtype=np.float64)
            initial_rank = [int(x) for x in np.argsort(stage1_post)[::-1]]
            shifts, schedule = _build_relative_action_cyclic_schedule(int(k), initial_rank, cand_actions)
            stage_probs = np.asarray([per_sample_probs[sample_idx_i][cyclic_indices[int(s)]] for s in shifts], dtype=np.float64)
            posteriors, _, confs = _compute_empirical_stage_posteriors(
                stage_probs=stage_probs,
                slot_to_content_schedule=schedule,
                mu_hat=mu_hat,
                residual_bank=residual_bank,
            )
            label_idx = int(labels_idx[sample_idx_i])
            for stage_idx in range(len(posteriors) - 1):
                p_t = float(np.clip(posteriors[stage_idx][label_idx], eps, 1.0))
                p_next = float(np.clip(posteriors[stage_idx + 1][label_idx], eps, 1.0))
                gain = float(np.log(p_next) - np.log(p_t))
                weight = float(1.0 - float(confs[stage_idx]))
                stage_num[stage_idx] += weight * gain
                stage_den[stage_idx] += weight
        stage_terms = []
        for stage_idx in range(max(k - 1, 1)):
            if stage_den[stage_idx] > eps:
                stage_terms.append(float(stage_num[stage_idx] / stage_den[stage_idx]))
            else:
                stage_terms.append(0.0)
        score = float(np.sum(np.asarray(stage_terms, dtype=np.float64)))
        selection_rows.append({
            "name": str(cand_name),
            "actions": _format_relative_action_sequence(cand_actions),
            "score": float(score),
            "stage_scores": [float(x) for x in stage_terms],
        })

    selection_rows = sorted(selection_rows, key=lambda row: float(row.get("score", float("-inf"))), reverse=True)
    best_row = selection_rows[0]
    return {
        "selection_policy": "initial_rank_relative_bruteforce_cyclic_top3",
        "selected_sequence_name": str(best_row.get("name")),
        "selected_action_sequence": list(best_row.get("actions") or []),
        "candidate_scores": selection_rows,
        "n_validation": int(len(sample_indices)),
    }


_EMPIRICAL_RESIDUAL_WEIGHTING = "uniform"
_EMPIRICAL_RESIDUAL_IDENT = False
_EMPIRICAL_RESIDUAL_IDENT_SHRINK = 1.0
_EMPIRICAL_RESIDUAL_IDENT_PROJECT = False


def _identify_question_residual(
    stage_probs: np.ndarray,
    slot_to_content_schedule: List[Tuple[int, ...]],
    mu_hat: np.ndarray,
    logit_clip: float = 1e-6,
    project: bool = False,
) -> np.ndarray:
    """
    LS-estimate this question's own residual eps(q) from its observed views.

    Model (centered logits, slot space): y_t = b + A_t c, b = mu + eps(q),
    A_t[slot, content] = 1 iff schedule[t][slot] == content. View differences
    y_0 - y_t = (A_0 - A_t) c cancel b; c is solved on the zero-sum subspace
    (pinv), then eps(q) = mean_t(y_t - A_t c) - mu_hat. Identified (rank k-1)
    from 3 views under the production targeted-latin schedules.

    project=True additionally handles the rank-deficient case (2 views): the
    min-norm pinv solution leaks the content component along ker(M) into b,
    and that leakage lies *exactly* along ker(M) (since A_t v = v for v in the
    kernel of the stage-2 double-transposition). Projecting eps onto the
    identified subspace removes it exactly, yielding the (k-2)-dim component
    of eps(q) with no content leakage. At >=3 views M is full rank and the
    projection is a no-op, so results match plain identify bit-for-bit.
    """
    arr = np.asarray(stage_probs, dtype=np.float64)
    n_views, k = arr.shape
    ys = np.log(np.clip(arr, logit_clip, None))
    ys = ys - ys.mean(axis=1, keepdims=True)
    As = []
    for perm in slot_to_content_schedule[:n_views]:
        A = np.zeros((k, k), dtype=np.float64)
        for slot_idx, content_idx in enumerate(perm):
            A[slot_idx, int(content_idx)] = 1.0
        As.append(A)
    M = np.vstack([As[0] - A for A in As[1:]])
    rhs = np.concatenate([ys[0] - y for y in ys[1:]])
    c = np.linalg.pinv(M) @ rhs
    c -= c.mean()
    b = np.mean([y - A @ c for y, A in zip(ys, As)], axis=0)
    b -= b.mean()
    eps = b - np.asarray(mu_hat, dtype=np.float64).reshape(-1)
    if project:
        eps = _project_onto_identified(eps, M, k)
    return eps


def _project_onto_identified(eps: np.ndarray, M: np.ndarray, k: int) -> np.ndarray:
    """
    Drop the components of eps that the view-difference system cannot identify.

    The identifiable subspace is row-space(M) restricted to the zero-sum
    subspace; ker(M) on that subspace carries pure content leakage (see
    _identify_question_residual). Full-rank M => returns eps unchanged.
    """
    Q = np.eye(k, dtype=np.float64) - np.ones((k, k), dtype=np.float64) / float(k)
    MQ = np.asarray(M, dtype=np.float64) @ Q
    _, sv, Vt = np.linalg.svd(MQ)
    tol = 1e-9 * max(1.0, float(sv[0]) if sv.size else 1.0)
    rank = int((sv > tol).sum())
    if rank >= k - 1:
        return eps
    null_dirs = Q @ Vt[rank:].T                      # zero-sum part of ker(MQ)
    Un, sn, _ = np.linalg.svd(null_dirs, full_matrices=False)
    N = Un[:, sn > tol]
    if N.size == 0:
        return eps
    return eps - N @ (N.T @ eps)


def _compute_empirical_stage_posteriors(
    stage_probs: np.ndarray,
    slot_to_content_schedule: List[Tuple[int, ...]],
    mu_hat: np.ndarray,
    residual_bank: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """
    Returns (posterior_by_stage, pred_idx_by_stage, conf_by_stage).
    """
    probs = np.asarray(stage_probs, dtype=np.float64)
    mu = np.asarray(mu_hat, dtype=np.float64).reshape(-1)
    residuals = np.asarray(residual_bank, dtype=np.float64)
    if residuals.ndim == 1:
        residuals = residuals.reshape(1, -1)
    if residuals.size == 0:
        residuals = np.zeros((1, probs.shape[1]), dtype=np.float64)

    k = probs.shape[1]
    inverse_assignments = [_invert_slot_to_content_perm(p) for p in slot_to_content_schedule]
    posterior_by_stage: List[np.ndarray] = []
    pred_idx_by_stage: List[int] = []
    conf_by_stage: List[float] = []

    for stage_idx in range(len(slot_to_content_schedule)):
        stage_residuals = residuals
        if _EMPIRICAL_RESIDUAL_IDENT:
            min_views = 2 if _EMPIRICAL_RESIDUAL_IDENT_PROJECT else 3
            if stage_idx + 1 >= min_views:
                stage_residuals = (
                    _EMPIRICAL_RESIDUAL_IDENT_SHRINK
                    * _identify_question_residual(
                        probs[: stage_idx + 1], slot_to_content_schedule[: stage_idx + 1], mu,
                        project=_EMPIRICAL_RESIDUAL_IDENT_PROJECT,
                    )
                ).reshape(1, -1)
            else:
                stage_residuals = np.zeros((1, k), dtype=np.float64)
        per_residual = []
        per_view_dists = []
        for residual in stage_residuals:
            scores = np.zeros((k,), dtype=np.float64)
            prior_factor = np.exp(-(mu + residual))
            views = []
            for inner_idx in range(stage_idx + 1):
                inv = inverse_assignments[inner_idx]
                stage_row = probs[inner_idx]
                v = stage_row[inv] * prior_factor[inv]
                tv = float(np.sum(v))
                views.append(v / tv if (np.isfinite(tv) and tv > eps) else np.ones((k,)) / float(k))
                scores += v
            total = float(np.sum(scores))
            if not np.isfinite(total) or total <= eps:
                post = np.ones((k,), dtype=np.float64) / float(k)
            else:
                post = scores / total
            per_residual.append(post)
            per_view_dists.append(views)
        arr = np.asarray(per_residual, dtype=np.float64)
        n_res = arr.shape[0]
        wmode = _EMPIRICAL_RESIDUAL_WEIGHTING
        if wmode == "uniform" or n_res == 1:
            posterior = np.mean(arr, axis=0)
        else:
            if wmode == "confidence":
                w = arr.max(axis=1)
            elif wmode == "agreement":
                logw = np.zeros((n_res,), dtype=np.float64)
                for r_i in range(n_res):
                    for v in per_view_dists[r_i]:
                        logw[r_i] += np.log(float(np.dot(v, arr[r_i])) + eps)
                logw -= float(np.max(logw))
                w = np.exp(logw)
            elif wmode == "proximity":
                obs = np.asarray(probs[0], dtype=np.float64)
                obs = obs / (float(np.sum(obs)) + eps)
                lo = np.log(obs + eps)
                lo = lo - float(np.mean(lo))
                d2 = np.sum((lo[None, :] - (mu[None, :] + residuals)) ** 2, axis=1)
                var = float(np.mean(np.var(residuals, axis=0)))
                w = np.exp(-(d2 - float(np.min(d2))) / (2.0 * max(var, eps)))
            else:
                w = np.ones((n_res,), dtype=np.float64)
            sw = float(np.sum(w))
            if not np.isfinite(sw) or sw <= eps:
                w = np.ones((n_res,), dtype=np.float64) / float(n_res)
            else:
                w = w / sw
            posterior = np.sum(w[:, None] * arr, axis=0)
        posterior = posterior / (float(np.sum(posterior)) + eps)
        posterior_by_stage.append(posterior)
        pred_idx_by_stage.append(int(np.argmax(posterior)))
        conf_by_stage.append(float(np.max(posterior)))

    return posterior_by_stage, pred_idx_by_stage, conf_by_stage


def _run_empirical_pride_policy_from_stage_infos(
    stage_infos: List[Dict[str, Any]],
    labels_idx: List[int],
    k: int,
    percentile: float,
    stage_schedule: str = "sqrt",
    stage_gamma: float = 0.5,
    percentile_mode: str = "online",
) -> Tuple[float, float, List[int], Dict[str, int]]:
    """
    Evaluate the empirical PriDe policy online from precomputed stage confidences.
    Thresholds are running quantiles over previously observed confidences for each stage,
    and a stage history is only updated when a sample actually reaches that stage.
    """
    N = len(stage_infos)
    if N == 0:
        return float("nan"), float("nan"), [], {}

    stage_schedule = str(stage_schedule or "sqrt").strip().lower()
    if stage_schedule not in {"flat", "sqrt"}:
        stage_schedule = "sqrt"
    percentile_mode = str(percentile_mode or "online").strip().lower()
    if percentile_mode not in {"online", "fixed_prefix"}:
        percentile_mode = "online"
    gamma = float(stage_gamma) if np.isfinite(float(stage_gamma)) and float(stage_gamma) > 0.0 else 0.5
    base_percentile = max(0.0, min(100.0, float(percentile)))
    histories: List[List[float]] = [[] for _ in range(int(k))]
    fixed_thresholds: List[float] = [0.0 for _ in range(int(k))]
    if percentile_mode == "fixed_prefix":
        fixed_histories: List[List[float]] = [[] for _ in range(int(k))]
        for info in stage_infos:
            if not bool(info.get("prefix_forced", False)):
                continue
            confs = [float(x) for x in (info.get("conf_by_stage") or [])]
            decision_stages = [int(x) for x in (info.get("decision_stages") or list(range(1, int(k) + 1)))]
            if len(confs) != len(decision_stages):
                continue
            for local_idx, stage_id in enumerate(decision_stages):
                fixed_histories[int(stage_id) - 1].append(float(confs[local_idx]))
        for stage_idx in range(int(k)):
            stage_id = stage_idx + 1
            if stage_schedule == "sqrt":
                stage_percentile = base_percentile / (float(stage_id) ** gamma)
            else:
                stage_percentile = base_percentile
            q = max(0.0, min(1.0, float(stage_percentile) / 100.0))
            hist = fixed_histories[stage_idx]
            fixed_thresholds[stage_idx] = float(np.quantile(np.asarray(hist, dtype=np.float64), q)) if hist else 0.0
    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    stage_counts = {f"n_stage_{stage + 1}": 0 for stage in range(int(k))}

    for sample_idx, info in enumerate(stage_infos):
        confs = [float(x) for x in (info.get("conf_by_stage") or [])]
        pred_by_stage = [int(x) for x in (info.get("pred_by_stage") or [])]
        decision_stages = [int(x) for x in (info.get("decision_stages") or list(range(1, int(k) + 1)))]
        forced_prefix = bool(info.get("prefix_forced", False))
        if len(confs) != len(pred_by_stage) or len(confs) != len(decision_stages):
            raise ValueError(f"Empirical PriDe stage info is inconsistent for sample {sample_idx}: conf={len(confs)}, pred={len(pred_by_stage)}, stages={len(decision_stages)}")

        if forced_prefix:
            stop_stage = int(decision_stages[-1])
        else:
            stop_stage = int(decision_stages[-1])
            for local_idx, stage_id in enumerate(decision_stages):
                if percentile_mode == "fixed_prefix":
                    thr = float(fixed_thresholds[int(stage_id) - 1])
                else:
                    hist = histories[int(stage_id) - 1]
                    if stage_schedule == "sqrt":
                        stage_percentile = base_percentile / (float(stage_id) ** gamma)
                    else:
                        stage_percentile = base_percentile
                    q = max(0.0, min(1.0, float(stage_percentile) / 100.0))
                    thr = float(np.quantile(np.asarray(hist, dtype=np.float64), q)) if hist else 0.0
                if float(confs[local_idx]) >= thr:
                    stop_stage = int(stage_id)
                    break

        stop_local_idx = decision_stages.index(int(stop_stage))
        pred_idx = int(pred_by_stage[stop_local_idx])
        preds.append(pred_idx)
        total_cost += float(stop_stage)
        corrects += 1 if pred_idx == int(labels_idx[sample_idx]) else 0
        stage_counts[f"n_stage_{stop_stage}"] = stage_counts.get(f"n_stage_{stop_stage}", 0) + 1

        if percentile_mode == "online":
            for local_idx, stage_id in enumerate(decision_stages[: stop_local_idx + 1]):
                histories[int(stage_id) - 1].append(float(confs[local_idx]))

    return total_cost / float(N), corrects / float(N), preds, stage_counts


def _run_empirical_pride_policy_from_stage_infos_confidence(
    stage_infos: List[Dict[str, Any]],
    labels_idx: List[int],
    k: int,
    confidence_threshold: float,
    stage_schedule: str = "sqrt",
    stage_gamma: float = 0.5,
) -> Tuple[float, float, List[int], Dict[str, int]]:
    """
    Evaluate the empirical PriDe policy with a fixed confidence threshold shared
    across all stages. Prefix calibration samples are forced to use the full
    Latin-square schedule to keep calibration accounting aligned with PriDe.
    """
    N = len(stage_infos)
    if N == 0:
        return float("nan"), float("nan"), [], {}

    stage_schedule = str(stage_schedule or "sqrt").strip().lower()
    if stage_schedule not in {"flat", "sqrt"}:
        stage_schedule = "sqrt"
    gamma = float(stage_gamma) if np.isfinite(float(stage_gamma)) and float(stage_gamma) > 0.0 else 0.5
    base_tau = float(confidence_threshold)
    total_cost = 0.0
    corrects = 0
    preds: List[int] = []
    stage_counts = {f"n_stage_{stage + 1}": 0 for stage in range(int(k))}

    for sample_idx, info in enumerate(stage_infos):
        confs = [float(x) for x in (info.get("conf_by_stage") or [])]
        pred_by_stage = [int(x) for x in (info.get("pred_by_stage") or [])]
        decision_stages = [int(x) for x in (info.get("decision_stages") or list(range(1, int(k) + 1)))]
        forced_prefix = bool(info.get("prefix_forced", False))
        if len(confs) != len(pred_by_stage) or len(confs) != len(decision_stages):
            raise ValueError(f"Empirical PriDe stage info is inconsistent for sample {sample_idx}: conf={len(confs)}, pred={len(pred_by_stage)}, stages={len(decision_stages)}")

        stop_stage = int(decision_stages[-1])
        if not forced_prefix:
            for local_idx, stage_id in enumerate(decision_stages):
                if stage_schedule == "sqrt":
                    chance = 1.0 / float(k)
                    tau = chance + (base_tau - chance) / (float(stage_id) ** gamma)
                else:
                    tau = base_tau
                tau = max(0.0, min(1.0, float(tau)))
                if float(confs[local_idx]) >= tau:
                    stop_stage = int(stage_id)
                    break

        stop_local_idx = decision_stages.index(int(stop_stage))
        pred_idx = int(pred_by_stage[stop_local_idx])
        preds.append(pred_idx)
        total_cost += float(stop_stage)
        corrects += 1 if pred_idx == int(labels_idx[sample_idx]) else 0
        stage_counts[f"n_stage_{stop_stage}"] = stage_counts.get(f"n_stage_{stop_stage}", 0) + 1

    return total_cost / float(N), corrects / float(N), preds, stage_counts


def _compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correct, dtype=np.float64).ravel()
    mask = np.isfinite(conf) & np.isfinite(corr)
    if np.sum(mask) <= 0:
        return float("nan")
    conf = np.clip(conf[mask], 0.0, 1.0)
    corr = corr[mask]
    edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1, dtype=np.float64)
    ece = 0.0
    n = float(conf.shape[0])
    for bin_idx in range(len(edges) - 1):
        lo = float(edges[bin_idx])
        hi = float(edges[bin_idx + 1])
        if bin_idx == len(edges) - 2:
            bmask = (conf >= lo) & (conf <= hi)
        else:
            bmask = (conf >= lo) & (conf < hi)
        if np.sum(bmask) <= 0:
            continue
        acc_b = float(np.mean(corr[bmask]))
        conf_b = float(np.mean(conf[bmask]))
        ece += (float(np.sum(bmask)) / n) * abs(acc_b - conf_b)
    return float(ece)


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64).ravel()
    mm = np.asarray(mask, dtype=bool).ravel()
    if vals.size == 0 or mm.size == 0 or vals.size != mm.size or np.sum(mm) <= 0:
        return float("nan")
    return float(np.mean(vals[mm]))


def _build_empirical_stage_analysis(
    stage_infos: List[Dict[str, Any]],
    labels_idx: List[int],
    k: int,
    sweep_mode: str,
    sweep_values: List[float],
    heuristic_points: List[Dict[str, Any]],
    ece_bins: int = 10,
    eps: float = 1e-12,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Build a stage-by-stage diagnostic summary plus per-sample trajectories.
    The summary is dataset-level over the evaluated samples for one subject/run/alpha.
    """
    ordered_stages: List[int] = []
    stage_buffers: Dict[int, Dict[str, List[float]]] = {}
    trajectories: List[Dict[str, Any]] = []

    for sample_pos, info in enumerate(stage_infos):
        label_idx = int(labels_idx[sample_pos])
        sample_id = int(info.get("sample_id", sample_pos))
        decision_stages = [int(x) for x in (info.get("decision_stages") or [])]
        pred_by_stage = [int(x) for x in (info.get("pred_by_stage") or [])]
        conf_by_stage = [float(x) for x in (info.get("conf_by_stage") or [])]
        true_prob_by_stage = [float(x) for x in (info.get("true_prob_by_stage") or [])]
        if not (len(decision_stages) == len(pred_by_stage) == len(conf_by_stage) == len(true_prob_by_stage)):
            raise ValueError(
                f"Empirical analysis stage info mismatch for sample {sample_pos}: "
                f"stages={len(decision_stages)}, pred={len(pred_by_stage)}, conf={len(conf_by_stage)}, true_prob={len(true_prob_by_stage)}"
            )
        correct_by_stage = [1 if int(pred) == int(label_idx) else 0 for pred in pred_by_stage]
        trajectories.append({
            "sample_pos": int(sample_pos),
            "sample_id": int(sample_id),
            "label_idx": int(label_idx),
            "prefix_forced": bool(info.get("prefix_forced", False)),
            "decision_stages": [int(x) for x in decision_stages],
            "pred_by_stage": [int(x) for x in pred_by_stage],
            "conf_by_stage": [float(x) for x in conf_by_stage],
            "true_prob_by_stage": [float(x) for x in true_prob_by_stage],
            "correct_by_stage": [int(x) for x in correct_by_stage],
        })
        for local_idx, stage_id in enumerate(decision_stages):
            if int(stage_id) not in stage_buffers:
                ordered_stages.append(int(stage_id))
                stage_buffers[int(stage_id)] = {
                    "pred": [],
                    "conf": [],
                    "true_prob": [],
                    "correct": [],
                }
            buf = stage_buffers[int(stage_id)]
            buf["pred"].append(int(pred_by_stage[local_idx]))
            buf["conf"].append(float(conf_by_stage[local_idx]))
            buf["true_prob"].append(float(true_prob_by_stage[local_idx]))
            buf["correct"].append(int(correct_by_stage[local_idx]))

    stage_metrics: Dict[str, Any] = {}
    for stage_id in ordered_stages:
        buf = stage_buffers[int(stage_id)]
        conf = np.asarray(buf["conf"], dtype=np.float64)
        true_prob = np.asarray(buf["true_prob"], dtype=np.float64)
        corr = np.asarray(buf["correct"], dtype=np.float64)
        correct_mask = corr >= 0.5
        wrong_mask = ~correct_mask
        stage_metrics[str(stage_id)] = {
            "n_samples": int(conf.shape[0]),
            "acc": float(np.mean(corr)) if corr.size > 0 else float("nan"),
            "nll": float(np.mean(-np.log(np.clip(true_prob, eps, 1.0)))) if true_prob.size > 0 else float("nan"),
            "avg_conf": float(np.mean(conf)) if conf.size > 0 else float("nan"),
            "ece": _compute_ece(conf, corr, n_bins=ece_bins),
            "conf_correct": _masked_mean(conf, correct_mask),
            "conf_wrong": _masked_mean(conf, wrong_mask),
        }

    transitions: List[Dict[str, Any]] = []
    sweep_mode_norm = str(sweep_mode or "percentile").strip().lower()
    if sweep_mode_norm not in {"percentile", "confidence"}:
        sweep_mode_norm = "percentile"
    for prev_stage, next_stage in zip(ordered_stages[:-1], ordered_stages[1:]):
        prev_buf = stage_buffers[int(prev_stage)]
        next_buf = stage_buffers[int(next_stage)]
        conf_prev = np.asarray(prev_buf["conf"], dtype=np.float64)
        corr_prev = np.asarray(prev_buf["correct"], dtype=np.float64)
        corr_next = np.asarray(next_buf["correct"], dtype=np.float64)
        w2c_mask = (corr_prev < 0.5) & (corr_next >= 0.5)
        c2w_mask = (corr_prev >= 0.5) & (corr_next < 0.5)
        trans_entry: Dict[str, Any] = {
            "from_stage": int(prev_stage),
            "to_stage": int(next_stage),
            "acc_from": float(np.mean(corr_prev)) if corr_prev.size > 0 else float("nan"),
            "acc_to": float(np.mean(corr_next)) if corr_next.size > 0 else float("nan"),
            "delta_acc": float(np.mean(corr_next - corr_prev)) if corr_prev.size > 0 else float("nan"),
            "w2c": float(np.mean(w2c_mask.astype(np.float64))) if corr_prev.size > 0 else float("nan"),
            "c2w": float(np.mean(c2w_mask.astype(np.float64))) if corr_prev.size > 0 else float("nan"),
            "threshold_analysis": [],
        }
        for sweep_value in sweep_values:
            sweep_f = float(sweep_value)
            if sweep_mode_norm == "percentile":
                tau = float(np.quantile(conf_prev, max(0.0, min(1.0, sweep_f / 100.0)))) if conf_prev.size > 0 else 0.0
                sweep_key = "p"
            else:
                tau = max(0.0, min(1.0, sweep_f))
                sweep_key = "confidence"
            low_mask = conf_prev < tau
            high_mask = ~low_mask
            trans_entry["threshold_analysis"].append({
                sweep_key: sweep_f,
                "threshold": float(tau),
                "low_ratio": float(np.mean(low_mask.astype(np.float64))) if conf_prev.size > 0 else float("nan"),
                "coverage": float(np.mean(high_mask.astype(np.float64))) if conf_prev.size > 0 else float("nan"),
                "accepted_accuracy": _masked_mean(corr_prev, high_mask),
                "acc_low_from": _masked_mean(corr_prev, low_mask),
                "acc_low_to": _masked_mean(corr_next, low_mask),
                "delta_low": _masked_mean(corr_next - corr_prev, low_mask),
                "w2c_low": _masked_mean(w2c_mask.astype(np.float64), low_mask),
                "c2w_low": _masked_mean(c2w_mask.astype(np.float64), low_mask),
                "acc_high_from": _masked_mean(corr_prev, high_mask),
                "acc_high_to": _masked_mean(corr_next, high_mask),
                "delta_high": _masked_mean(corr_next - corr_prev, high_mask),
                "w2c_high": _masked_mean(w2c_mask.astype(np.float64), high_mask),
                "c2w_high": _masked_mean(c2w_mask.astype(np.float64), high_mask),
            })
        transitions.append(trans_entry)

    adaptive_points: List[Dict[str, Any]] = []
    for hp in (heuristic_points or []):
        if not isinstance(hp, dict):
            continue
        out = {
            "cost": float(hp.get("cost", float("nan"))),
            "acc": float(hp.get("acc", float("nan"))),
            "recall_std": float(hp.get("recall_std", float("nan"))),
        }
        if hp.get("th1_p") is not None:
            out["p"] = float(hp.get("th1_p"))
        if hp.get("conf_th") is not None:
            out["confidence"] = float(hp.get("conf_th"))
        for stage_id in ordered_stages:
            key = f"n_stage_{int(stage_id)}"
            if key in hp:
                out[key] = int(hp.get(key, 0))
        adaptive_points.append(out)

    summary = {
        "n_samples": int(len(stage_infos)),
        "k": int(k),
        "decision_stages": [int(x) for x in ordered_stages],
        "ece_bins": int(ece_bins),
        "sweep_mode": sweep_mode_norm,
        "sweep_values": [float(x) for x in sweep_values],
        "stage_metrics": stage_metrics,
        "transitions": transitions,
        "adaptive_points": adaptive_points,
    }
    return summary, trajectories


def _run_prefix_cyclic_postfix_base(
    base_correct: List[bool],
    cyclic_correct: List[bool],
    k: int,
    prefix_ids: set,
) -> Tuple[float, float]:
    """
    Default+PRIDE: prefix=cyclic, postfix=base.
    alpha=2 → 앞 2% cyclic, 뒤 98% 보정된 base로 측정.
    Returns (cost, acc).
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan")
    total_cost, corrects = 0.0, 0
    for i in range(N):
        if int(i) in prefix_ids:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
    return total_cost / float(N), corrects / float(N)


def _run_cyclic_random_fraction(
    base_correct: List[bool],
    cyclic_correct: List[bool],
    k: int,
    fraction_pct: float,
    seed: int,
) -> Tuple[float, float]:
    """
    Randomly select fraction_pct% of samples to run cyclic; rest use base.
    Returns (cost, acc).
    """
    N = len(base_correct)
    if N == 0:
        return float("nan"), float("nan")
    frac = max(0.0, min(1.0, float(fraction_pct) / 100.0))
    m = int(round(frac * N))
    rng = np.random.default_rng(int(seed))
    cyclic_indices = set(rng.choice(np.arange(N, dtype=np.int64), size=min(m, N), replace=False))
    total_cost, corrects = 0.0, 0
    for i in range(N):
        if i in cyclic_indices:
            total_cost += float(k)
            corrects += 1 if cyclic_correct[i] else 0
        else:
            total_cost += 1.0
            corrects += 1 if base_correct[i] else 0
    return total_cost / float(N), corrects / float(N)


def _run_cyclic_random_fraction_with_preds(
    base_pred_idx: List[int],
    cyclic_pred_idx: List[int],
    labels_idx: List[int],
    k: int,
    fraction_pct: float,
    seed: int,
) -> Tuple[float, float, List[int]]:
    """Returns (cost, acc, preds) for recall_std."""
    N = len(base_pred_idx)
    if N == 0:
        return float("nan"), float("nan"), []
    frac = max(0.0, min(1.0, float(fraction_pct) / 100.0))
    m = int(round(frac * N))
    rng = np.random.default_rng(int(seed))
    cyclic_indices = set(rng.choice(np.arange(N, dtype=np.int64), size=min(m, N), replace=False))
    total_cost, corrects = 0.0, 0
    preds: List[int] = []
    for i in range(N):
        if i in cyclic_indices:
            pred_i = int(cyclic_pred_idx[i])
            total_cost += float(k)
        else:
            pred_i = int(base_pred_idx[i])
            total_cost += 1.0
        preds.append(pred_i)
        corrects += 1 if (pred_i == int(labels_idx[i])) else 0
    return total_cost / float(N), corrects / float(N), preds


def logging_cuda_memory_usage():
    if not _NVML_OK:
        logger.info("******** Memory usage ********")
        logger.info("NVML unavailable; skipping GPU memory usage logging.")
        return
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(
            "GPU {}: {:.2f} GB / {:.2f} GB".format(
                i,
                meminfo.used / 1024 ** 3,
                meminfo.total / 1024 ** 3
            )
        )


def _rotations(k: int):
    """cyclic rotations: ABCD, BCDA, CDAB, DABC"""
    return [tuple((i + s) % k for i in range(k)) for s in range(k)]


def _aggregate_probs_over_permutations(probs_seq, permuted_indices, k: int):
    """
    probs_seq: list/array of length = (#permutations used)
      each element: length k (letter-space probs)
    permuted_indices: list of permutations p where p[j] is content-index at letter position j.
    Returns: agg (k,) content-space aggregated probs (mean over permutations)
    """
    agg = np.zeros(k, dtype=np.float64)
    for perm_idx, p in enumerate(permuted_indices):
        letter_probs = np.asarray(probs_seq[perm_idx], dtype=np.float64)
        for j in range(k):
            agg[p[j]] += letter_probs[j]
    if len(permuted_indices) > 0:
        agg /= float(len(permuted_indices))
    return agg


def _probe_shift_cyclic_put_top2_into_top1_slot(base_probs: np.ndarray, k: int) -> Tuple[int, int, int]:
    """
    규칙:
      - base(letter-space)에서 top1=t1, top2=t2를 찾고,
      - cyclic rotations 중 "원래 top1 슬롯(=t1 위치)에 top2(t2)가 오도록" shift s 선택
    shift s = (t2 - t1) mod k
    """
    bp = np.asarray(base_probs, dtype=np.float64)
    sidx = np.argsort(bp)[::-1]
    t1 = int(sidx[0])
    t2 = int(sidx[1]) if len(sidx) > 1 else int(sidx[0])
    s = int((t2 - t1) % int(k))
    if s == 0:
        s = 1 if k > 1 else 0
    return s, t1, t2


def _read_results_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        lines = [e for e in lines if e.get('type') == 'result']
        lines = sorted(lines, key=lambda x: int(x['data']['idx']))
        return lines
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _expected_cached_result_ids(num_eval_samples: int, max_samples: Optional[int] = None) -> set:
    indices = list(range(int(num_eval_samples)))
    random.Random(123).shuffle(indices)
    if max_samples is not None:
        indices = indices[:int(max_samples)]
    return set(indices)


def _validate_cached_results(results, num_eval_samples: int, max_samples: Optional[int] = None) -> Tuple[bool, str]:
    if results is None:
        return False, "file could not be read or parsed"

    expected_ids = _expected_cached_result_ids(num_eval_samples, max_samples=max_samples)
    seen_ids = []
    try:
        for result in results:
            if result.get("type") != "result":
                continue
            seen_ids.append(int(result["data"]["idx"]))
    except Exception:
        return False, "one or more result rows are malformed"

    seen_set = set(seen_ids)
    if len(seen_ids) != len(seen_set):
        return False, "duplicate result idx values found"
    if seen_set != expected_ids:
        missing = len(expected_ids - seen_set)
        extra = len(seen_set - expected_ids)
        return False, f"expected {len(expected_ids)} results, found {len(seen_set)} (missing={missing}, extra={extra})"
    return True, ""


def _empirical_stage_cache_path(args, subject: str, run_idx: int, use_run_suffix: bool, alpha: float) -> str:
    analysis_dir = os.path.join(
        build_results_dir(args, task=args.task, num_few_shot=args.num_few_shot, setting="full"),
        "empirical_analysis",
    )
    os.makedirs(analysis_dir, exist_ok=True)
    alpha_tag = f"{float(alpha):g}"
    run_tag = f"_run{int(run_idx)}" if use_run_suffix else ""
    return os.path.join(analysis_dir, f"{subject}{run_tag}_empirical_alpha{alpha_tag}_stage_cache.jsonl")


def _schedule_signature(stage_schedule) -> List[List[int]]:
    return [[int(x) for x in row] for row in stage_schedule]


def _load_empirical_stage_cache(cache_path: str) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    if not os.path.exists(cache_path):
        return rows
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") != "empirical_stage_cache":
                    continue
                try:
                    sample_pos = int(obj["sample_pos"])
                except Exception:
                    continue
                rows[sample_pos] = obj
    except Exception:
        return {}
    return rows


def _append_empirical_stage_cache(cache_path: str, row: dict) -> None:
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _cached_empirical_stage_probs(cache_row: Optional[dict], sample_id: int, k: int, stage_schedule) -> Optional[np.ndarray]:
    if not cache_row:
        return None
    try:
        if int(cache_row.get("sample_id")) != int(sample_id):
            return None
        if int(cache_row.get("k")) != int(k):
            return None
        if cache_row.get("stage_schedule") != _schedule_signature(stage_schedule):
            return None
        probs = np.asarray(cache_row.get("stage_probs"), dtype=np.float64)
        if probs.ndim != 2 or probs.shape[0] != len(stage_schedule) or probs.shape[1] != int(k):
            return None
        return probs
    except Exception:
        return None


def _quantile(arr: np.ndarray, p01: float) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    p01 = float(max(0.0, min(1.0, p01)))
    return float(np.quantile(arr, p01))


def _plot_confidence_distribution(
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    out_path: str,
    title: str
):
    """
    default_conf (Base Gap)와 mean_conf (Avg Gap)의 분포(Histogram)를 그리고
    주요 Percentile 지점(10, 20, 30)을 표시
    """
    plt.figure(figsize=(10, 6), dpi=160)
    
    # Histogram
    plt.hist(default_conf, bins=50, range=(0, 1), alpha=0.5, label='Base Gap (default_conf)', color='gray', density=True)
    plt.hist(mean_conf, bins=50, range=(0, 1), alpha=0.5, label='Avg Gap (mean_conf)', color='blue', density=True)
    
    # Percentiles
    percs = [10, 20, 30]
    colors = ['red', 'green', 'purple']
    
    # Base Gap Percentiles
    for p, c in zip(percs, colors):
        val = np.percentile(default_conf, p)
        plt.axvline(val, color=c, linestyle='--', alpha=0.7, label=f'Base p{p}: {val:.3f}')
        
    # Avg Gap Percentiles
    for p, c in zip(percs, colors):
        val = np.percentile(mean_conf, p)
        plt.axvline(val, color=c, linestyle=':', alpha=0.9, linewidth=2, label=f'Avg p{p}: {val:.3f}')

    plt.xlabel("Confidence Gap")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_baseline_points_scatter(
    curve_obj: dict,
    out_path: str,
    title: str,
    extra_points: List[dict] = None
):
    """
    Baseline report에 나오는 각 정책들의 Cost vs Accuracy를 Point로 찍어서 비교
    extra_points: [{'cost': float, 'acc': float, 'label': str, 'marker': str, 'color': str}, ...]
    """
    plt.figure(figsize=(8, 6), dpi=160)
    
    # 1. Always Points (Reference)
    always = curve_obj.get("always", {})
    if "default" in always:
        plt.scatter(always["default"]["cost"], always["default"]["acc"], 
                   marker='*', s=300, color='gray', label='Default', zorder=10)
    if "cyclic" in always:
        plt.scatter(always["cyclic"]["cost"], always["cyclic"]["acc"], 
                   marker='d', s=150, color='purple', label='Cyclic', zorder=10)
    # Full removed as per user request
    # if "full" in always:
    #     plt.scatter(always["full"]["cost"], always["full"]["acc"], 
    #                marker='X', s=150, color='black', label='Full', zorder=10)

    # 2. Policy Points (REAL-WORLD online; single point)
    policies = ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]
    markers = ['s', '^', 'v', 'o']
    colors = ['orange', 'brown', 'green', 'blue']
    
    for key, m, c in zip(policies, markers, colors):
        if key in curve_obj:
            # single point (length-1)
            cost = float(curve_obj[key]["costs"][0])
            acc = float(curve_obj[key]["accuracies"][0])
            plt.scatter(cost, acc, marker=m, s=120, color=c, label=key, alpha=0.9)

    # 3. Extra Points (th1/sqrt(k), th1^2, th1^1.5, ...)
    if extra_points:
        for p in extra_points:
            plt.scatter(p['cost'], p['acc'], marker=p['marker'], s=150, color=p['color'], 
                       edgecolors='black', label=p['label'], zorder=15)

    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right')
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_baseline_vs_pride_points_scatter(
    baseline_obj: dict,
    pride_obj: dict,
    out_path: str,
    title: str,
):
    """
    Overlay baseline vs PRIDE+OURS (same p) on one scatter.
    Baseline: filled markers
    PRIDE+OURS: same markers but hollow (edge only)
    """
    plt.figure(figsize=(8, 6), dpi=160)

    def _plot_one(obj: dict, prefix: str, hollow: bool):
        always = obj.get("always", {}) if isinstance(obj, dict) else {}
        if "default" in always:
            plt.scatter(
                float(always["default"]["cost"]),
                float(always["default"]["acc"]),
                marker="*",
                s=260,
                facecolors="none" if hollow else "gray",
                edgecolors="gray",
                linewidths=1.8 if hollow else 1.0,
                label=f"{prefix}Default",
                zorder=10,
            )
        if "cyclic" in always:
            plt.scatter(
                float(always["cyclic"]["cost"]),
                float(always["cyclic"]["acc"]),
                marker="d",
                s=140,
                facecolors="none" if hollow else "purple",
                edgecolors="purple",
                linewidths=1.8 if hollow else 1.0,
                label=f"{prefix}Cyclic",
                zorder=10,
            )

        policies = ["switch_full", "switch_cyclic", "ours_top2flip", "ours_avggap"]
        markers = ['s', '^', 'v', 'o']
        colors = ['orange', 'brown', 'green', 'blue']
        for key, m, c in zip(policies, markers, colors):
            if key not in obj:
                continue
            cost = float(obj[key]["costs"][0])
            acc = float(obj[key]["accuracies"][0])
            plt.scatter(
                cost,
                acc,
                marker=m,
                s=110,
                facecolors="none" if hollow else c,
                edgecolors=c,
                linewidths=1.8 if hollow else 1.0,
                alpha=0.95,
                label=f"{prefix}{key}",
            )

        # heuristic points, if present (use stored marker/color when available)
        for hp in (obj.get("heuristic_points", []) or []):
            if not isinstance(hp, dict):
                continue
            cost = float(hp.get("cost", float("nan")))
            acc = float(hp.get("acc", float("nan")))
            lab = str(hp.get("label", "heuristic"))
            mk = str(hp.get("marker", "o"))
            col = str(hp.get("color", "black"))
            if np.isnan(cost) or np.isnan(acc):
                continue
            plt.scatter(
                cost,
                acc,
                marker=mk,
                s=80,
                facecolors="none" if hollow else col,
                edgecolors=col,
                linewidths=1.8 if hollow else 1.0,
                alpha=0.45,
                label=f"{prefix}{lab}" if prefix else lab,
            )

    _plot_one(baseline_obj, prefix="BASE_", hollow=False)
    _plot_one(pride_obj, prefix="PRIDE_", hollow=True)

    plt.xlabel("Computational Cost (× of default)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='lower right', fontsize=7, ncol=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_delta_cost_bars_by_p(
    delta_cost_by_p: Dict[float, Dict[str, float]],
    out_path: str,
    title: str,
    ylabel: str = "Δ Cost (PRIDE+OURS - BASELINE)",
):
    """
    Grouped bar chart: x-axis = p values, bars = policies/heuristics.
    delta_cost_by_p[p][label] = mean delta cost.
    """
    if not isinstance(delta_cost_by_p, dict) or len(delta_cost_by_p) == 0:
        return

    ps = sorted([float(p) for p in delta_cost_by_p.keys()])
    # collect labels that have at least one finite value
    all_labels = set()
    for p in ps:
        for lab, v in (delta_cost_by_p.get(p, {}) or {}).items():
            all_labels.add(str(lab))
    labels = sorted(list(all_labels))
    if len(labels) == 0:
        return

    # filter labels with any finite
    filt_labels = []
    for lab in labels:
        vs = []
        for p in ps:
            v = float((delta_cost_by_p.get(p, {}) or {}).get(lab, float("nan")))
            vs.append(v)
        if any(np.isfinite(v) for v in vs):
            filt_labels.append(lab)
    labels = filt_labels
    if len(labels) == 0:
        return

    x = np.arange(len(ps), dtype=np.float64)
    width = 0.80 / float(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)

    for i, lab in enumerate(labels):
        vals = []
        for p in ps:
            v = float((delta_cost_by_p.get(p, {}) or {}).get(lab, float("nan")))
            vals.append(0.0 if (not np.isfinite(v)) else v)
        offset = (i - (len(labels) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=str(lab))

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"p{int(round(p))}" for p in ps])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7, ncol=3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_scatter(
    delta_points: List[dict],
    out_path: str,
    title: str,
    xlabel: str = "Δ Cost (PRIDE+OURS - BASELINE)",
    ylabel: str = "Δ Accuracy (PRIDE+OURS - BASELINE)",
):
    """
    delta_points: [{'label': str, 'dcost': float, 'dacc': float, 'marker': str, 'color': str}, ...]
    """
    if not delta_points:
        return
    plt.figure(figsize=(8.4, 6.0), dpi=180)
    for p in delta_points:
        dcost = float(p.get("dcost", float("nan")))
        dacc = float(p.get("dacc", float("nan")))
        if not (np.isfinite(dcost) and np.isfinite(dacc)):
            continue
        lab = str(p.get("label", ""))
        mk = str(p.get("marker", "o"))
        col = str(p.get("color", "black"))
        plt.scatter(dcost, dacc, marker=mk, s=110, color=col, edgecolors="black", alpha=0.85)
        if lab:
            plt.annotate(lab, (dcost, dacc), textcoords="offset points", xytext=(6, 4), fontsize=7, alpha=0.85)
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_th2_tradeoff_curve_compare(
    subject: str,
    curve_save_path: str,
    th1_list: List[float],
    default_conf_base: np.ndarray,
    mean_conf_base: np.ndarray,
    base_correct_base: List[bool],
    cyclic_correct_base: List[bool],
    probe2_correct_base: np.ndarray,
    default_conf_pr: np.ndarray,
    mean_conf_pr: np.ndarray,
    base_correct_pr: List[bool],
    cyclic_correct_pr: List[bool],
    probe2_correct_pr: np.ndarray,
    k: int,
    args: Any,
    wandb_ok: bool = False,
    wandb_run: Any = None,
    fname_tag: str = "PRIDE_COMPARE",
    forced_cyclic_ids_pr: Optional[set] = None,
):
    """
    Compare dense th2-sweep curves (baseline vs PRIDE+OURS debiased stats).
    Lines: baseline solid, PRIDE dashed. Colors encode th1.
    """
    dense_th2_list = list(range(1, 31))
    default_acc_base = float(np.mean(np.asarray(base_correct_base, dtype=np.float64))) if len(base_correct_base) else float("nan")
    # PRIDE default: prefix->cyclic, postfix->base (debias_pride.py와 동일)
    Npr = len(base_correct_pr)
    if forced_cyclic_ids_pr is not None and Npr > 0:
        default_corrects_pr = [
            cyclic_correct_pr[i] if i in forced_cyclic_ids_pr else base_correct_pr[i]
            for i in range(Npr)
        ]
        default_acc_pr = float(np.mean(np.asarray(default_corrects_pr, dtype=np.float64)))
    else:
        default_acc_pr = float(np.mean(np.asarray(base_correct_pr, dtype=np.float64))) if Npr else float("nan")
    # Anchor for PRIDE curves: use BASELINE default as reference (requested)
    anchor_default_acc = float(default_acc_base)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    suffix = f"_{str(fname_tag).strip()}" if str(fname_tag).strip() else ""

    # Plot A: Cost vs th2
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        costs_b = []
        costs_p = []
        for th2p in dense_th2_list:
            cb, _ = _run_online_avggap_policy(
                default_conf=default_conf_base,
                mean_conf=mean_conf_base,
                base_correct=base_correct_base,
                cyclic_correct=cyclic_correct_base,
                probe2_correct=probe2_correct_base,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
            )
            cp, _ = _run_online_avggap_policy(
                default_conf=default_conf_pr,
                mean_conf=mean_conf_pr,
                base_correct=base_correct_pr,
                cyclic_correct=cyclic_correct_pr,
                probe2_correct=probe2_correct_pr,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids_pr,
            )
            costs_b.append(float(cb))
            costs_p.append(float(cp))
        ax1.plot(dense_th2_list, costs_b, color=color, linewidth=1.6, alpha=0.75)
        ax1.plot(dense_th2_list, costs_p, color=color, linewidth=1.6, alpha=0.75, linestyle="--")

        # Heuristic points on Cost-vs-th2 compare
        try:
            def _pt(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_base,
                    mean_conf=mean_conf_base,
                    base_correct=base_correct_base,
                    cyclic_correct=cyclic_correct_base,
                    probe2_correct=probe2_correct_base,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=None,
                )

            def _pt_pr(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_pr,
                    mean_conf=mean_conf_pr,
                    base_correct=base_correct_pr,
                    cyclic_correct=cyclic_correct_pr,
                    probe2_correct=probe2_correct_pr,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=forced_cyclic_ids_pr,
                )

            pts = [
                ("*", lambda x: x / 2.0),
                ("P", lambda x, kk=k: x / math.sqrt(float(kk))),
                ("s", lambda x: x ** 2),
                ("^", lambda x: x ** 1.5),
            ]
            for mk, rf in pts:
                cb, _, p_b = _pt(rf)
                cp2, _, p_p2 = _pt_pr(rf)
                ax1.scatter([float(p_b)], [float(cb)], marker=mk, s=70, color=color, edgecolors="black", zorder=7)
                ax1.scatter([float(p_p2)], [float(cp2)], marker=mk, s=70, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_s, _, p_b_s = _run_online_sqrt_policy(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_s, _, p_p_s = _run_online_sqrt_policy(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax1.scatter([float(p_b_s)], [float(cb_s)], marker="D", s=60, color=color, edgecolors="black", zorder=7)
            ax1.scatter([float(p_p_s)], [float(cp_s)], marker="D", s=60, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_lc, _, p_b_lc = _run_online_sqrt_policy_lowconf_update(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_lc, _, p_p_lc = _run_online_sqrt_policy_lowconf_update(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax1.scatter([float(p_b_lc)], [float(cb_lc)], marker="X", s=60, color=color, edgecolors="black", zorder=7)
            ax1.scatter([float(p_p_lc)], [float(cp_lc)], marker="X", s=60, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)
        except Exception:
            pass
    ax1.set_xlabel("th2 (percentile, avg gap)")
    ax1.set_ylabel("Computational Cost (× of default)")
    ax1.set_title(f"{getattr(args,'task','task')} {subject} — Cost vs th2 (BASELINE solid vs PRIDE dashed)")
    try:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="BASELINE (solid)"),
            Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="PRIDE+OURS (dashed)"),
        ]
        ax1.legend(handles=handles, loc="best", fontsize=9)
    except Exception:
        pass
    ax1.grid(True, linestyle="--", alpha=0.35)
    out_cost = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST_compare{suffix}.png")
    os.makedirs(os.path.dirname(out_cost), exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(out_cost, bbox_inches="tight")
    plt.close(fig1)

    # Plot B: ΔAcc vs th2 (reference = BASELINE default; PRIDE also uses BASELINE default)
    fig2, ax2 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        da_b = []
        da_p = []
        for th2p in dense_th2_list:
            _, ab = _run_online_avggap_policy(
                default_conf=default_conf_base,
                mean_conf=mean_conf_base,
                base_correct=base_correct_base,
                cyclic_correct=cyclic_correct_base,
                probe2_correct=probe2_correct_base,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
            )
            _, ap = _run_online_avggap_policy(
                default_conf=default_conf_pr,
                mean_conf=mean_conf_pr,
                base_correct=base_correct_pr,
                cyclic_correct=cyclic_correct_pr,
                probe2_correct=probe2_correct_pr,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids_pr,
            )
            da_b.append((float(ab) - float(anchor_default_acc)) * 100.0)
            da_p.append((float(ap) - float(anchor_default_acc)) * 100.0)
        ax2.plot(dense_th2_list, da_b, color=color, linewidth=1.6, alpha=0.75)
        ax2.plot(dense_th2_list, da_p, color=color, linewidth=1.6, alpha=0.75, linestyle="--")

        # Heuristic points on ΔAcc-vs-th2 compare
        try:
            def _pt(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_base,
                    mean_conf=mean_conf_base,
                    base_correct=base_correct_base,
                    cyclic_correct=cyclic_correct_base,
                    probe2_correct=probe2_correct_base,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=None,
                )

            def _pt_pr(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_pr,
                    mean_conf=mean_conf_pr,
                    base_correct=base_correct_pr,
                    cyclic_correct=cyclic_correct_pr,
                    probe2_correct=probe2_correct_pr,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=forced_cyclic_ids_pr,
                )

            pts = [
                ("*", lambda x: x / 2.0),
                ("P", lambda x, kk=k: x / math.sqrt(float(kk))),
                ("s", lambda x: x ** 2),
                ("^", lambda x: x ** 1.5),
            ]
            for mk, rf in pts:
                cb, ab, p_b = _pt(rf)
                cp2, ap2, p_p2 = _pt_pr(rf)
                ax2.scatter([float(p_b)], [(float(ab) - float(anchor_default_acc)) * 100.0], marker=mk, s=70, color=color, edgecolors="black", zorder=7)
                ax2.scatter([float(p_p2)], [(float(ap2) - float(anchor_default_acc)) * 100.0], marker=mk, s=70, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_s, ab_s, p_b_s = _run_online_sqrt_policy(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_s, ap_s, p_p_s = _run_online_sqrt_policy(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax2.scatter([float(p_b_s)], [(float(ab_s) - float(anchor_default_acc)) * 100.0], marker="D", s=60, color=color, edgecolors="black", zorder=7)
            ax2.scatter([float(p_p_s)], [(float(ap_s) - float(anchor_default_acc)) * 100.0], marker="D", s=60, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_lc, ab_lc, p_b_lc = _run_online_sqrt_policy_lowconf_update(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_lc, ap_lc, p_p_lc = _run_online_sqrt_policy_lowconf_update(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax2.scatter([float(p_b_lc)], [(float(ab_lc) - float(anchor_default_acc)) * 100.0], marker="X", s=60, color=color, edgecolors="black", zorder=7)
            ax2.scatter([float(p_p_lc)], [(float(ap_lc) - float(anchor_default_acc)) * 100.0], marker="X", s=60, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)
        except Exception:
            pass
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.set_xlabel("th2 (percentile, avg gap)")
    ax2.set_ylabel("Δ Accuracy (%)")
    ax2.set_title(f"{getattr(args,'task','task')} {subject} — ΔAcc vs th2 (ref=BASE default; BASE solid vs PRIDE dashed)")
    try:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="BASELINE (solid)"),
            Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="PRIDE+OURS (dashed)"),
        ]
        ax2.legend(handles=handles, loc="best", fontsize=9)
    except Exception:
        pass
    ax2.grid(True, linestyle="--", alpha=0.35)
    out_da = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_DELTA_ACC_compare{suffix}.png")
    fig2.tight_layout()
    fig2.savefig(out_da, bbox_inches="tight")
    plt.close(fig2)

    # Plot C: Cost vs ΔAcc (trade-off; reference = BASELINE default)
    fig3, ax3 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        xs_b, ys_b = [], []
        xs_p, ys_p = [], []
        for th2p in dense_th2_list:
            cb, ab = _run_online_avggap_policy(
                default_conf=default_conf_base,
                mean_conf=mean_conf_base,
                base_correct=base_correct_base,
                cyclic_correct=cyclic_correct_base,
                probe2_correct=probe2_correct_base,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
            )
            cp, ap = _run_online_avggap_policy(
                default_conf=default_conf_pr,
                mean_conf=mean_conf_pr,
                base_correct=base_correct_pr,
                cyclic_correct=cyclic_correct_pr,
                probe2_correct=probe2_correct_pr,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids_pr,
            )
            xs_b.append(float(cb)); ys_b.append((float(ab) - float(anchor_default_acc)) * 100.0)
            xs_p.append(float(cp)); ys_p.append((float(ap) - float(anchor_default_acc)) * 100.0)
        ax3.plot(xs_b, ys_b, color=color, linewidth=1.4, alpha=0.55)
        ax3.plot(xs_p, ys_p, color=color, linewidth=1.4, alpha=0.55, linestyle="--")

        # Heuristic points on the compare trade-off plot (same marker set as th2_tradeoff)
        try:
            # baseline (filled)
            def _pt(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_base,
                    mean_conf=mean_conf_base,
                    base_correct=base_correct_base,
                    cyclic_correct=cyclic_correct_base,
                    probe2_correct=probe2_correct_base,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=None,
                )

            def _pt_pr(rule_func):
                return _run_online_th1_quantile_th2_from_th1_rule(
                    default_conf=default_conf_pr,
                    mean_conf=mean_conf_pr,
                    base_correct=base_correct_pr,
                    cyclic_correct=cyclic_correct_pr,
                    probe2_correct=probe2_correct_pr,
                    k=k,
                    th1_percent=th1p,
                    th2_rule_from_th1_value=rule_func,
                    forced_cyclic_ids=forced_cyclic_ids_pr,
                )

            pts = [
                ("*", lambda x: x / 2.0),
                ("P", lambda x, kk=k: x / math.sqrt(float(kk))),
                ("s", lambda x: x ** 2),
                ("^", lambda x: x ** 1.5),
            ]

            for mk, rf in pts:
                cb, ab, _ = _pt(rf)
                cp2, ap2, _ = _pt_pr(rf)
                ax3.scatter([float(cb)], [(float(ab) - float(anchor_default_acc)) * 100.0], marker=mk, s=75, color=color, edgecolors="black", zorder=7)
                ax3.scatter([float(cp2)], [(float(ap2) - float(anchor_default_acc)) * 100.0], marker=mk, s=75, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            # Online sqrt points
            cb_s, ab_s, _ = _run_online_sqrt_policy(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_s, ap_s, _ = _run_online_sqrt_policy(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax3.scatter([float(cb_s)], [(float(ab_s) - float(anchor_default_acc)) * 100.0], marker="D", s=65, color=color, edgecolors="black", zorder=7)
            ax3.scatter([float(cp_s)], [(float(ap_s) - float(anchor_default_acc)) * 100.0], marker="D", s=65, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)

            cb_lc, ab_lc, _ = _run_online_sqrt_policy_lowconf_update(
                default_conf_base, mean_conf_base, base_correct_base, cyclic_correct_base, probe2_correct_base, k, th1p, forced_cyclic_ids=None
            )
            cp_lc, ap_lc, _ = _run_online_sqrt_policy_lowconf_update(
                default_conf_pr, mean_conf_pr, base_correct_pr, cyclic_correct_pr, probe2_correct_pr, k, th1p, forced_cyclic_ids=forced_cyclic_ids_pr
            )
            ax3.scatter([float(cb_lc)], [(float(ab_lc) - float(anchor_default_acc)) * 100.0], marker="X", s=65, color=color, edgecolors="black", zorder=7)
            ax3.scatter([float(cp_lc)], [(float(ap_lc) - float(anchor_default_acc)) * 100.0], marker="X", s=65, facecolors="none", edgecolors=color, linewidths=1.8, zorder=7)
        except Exception:
            pass
    ax3.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax3.set_xlabel("Computational Cost (× of default)")
    ax3.set_ylabel("Δ Accuracy (%)")
    ax3.set_title(f"{getattr(args,'task','task')} {subject} — Trade-off (ΔAcc ref=BASE default; BASE solid vs PRIDE dashed)")
    try:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="BASELINE (solid)"),
            Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="PRIDE+OURS (dashed)"),
        ]
        ax3.legend(handles=handles, loc="best", fontsize=9)
    except Exception:
        pass
    ax3.grid(True, linestyle="--", alpha=0.35)
    try:
        ax3.text(
            0.01,
            0.01,
            "Heuristic markers: * th1/2, P th1/sqrt(k), s th1^2, ^ th1^1.5, D Sqrt(All), X Sqrt(LowConf)\n"
            "PRIDE heuristic pts are hollow (edge=color), BASELINE are filled.",
            transform=ax3.transAxes,
            fontsize=7,
            alpha=0.85,
            va="bottom",
        )
    except Exception:
        pass
    out_tr = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST_vs_DELTA_compare{suffix}.png")
    fig3.tight_layout()
    fig3.savefig(out_tr, bbox_inches="tight")
    plt.close(fig3)

    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({
                f"plots/{subject}/th2_tradeoff_COST_compare{suffix}": wandb.Image(out_cost),
                f"plots/{subject}/th2_tradeoff_DELTA_ACC_compare{suffix}": wandb.Image(out_da),
                f"plots/{subject}/th2_tradeoff_COST_vs_DELTA_compare{suffix}": wandb.Image(out_tr),
            })
        except Exception:
            pass


def _compute_and_plot_th2_tradeoff(
    subject: str,
    curve_save_path: str,
    th1_list: List[float],
    th2_list: List[float],
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    base_correct_list: List[bool],
    cyclic_correct_list: List[bool],
    arr_probe2_correct: np.ndarray,
    k: int,
    args: Any,
    wandb_ok: bool = False,
    wandb_run: Any = None,
    plot_tag: str = "BASELINE",
    fname_tag: str = "",
    forced_cyclic_ids: Optional[set] = None,
):
    """
    th1/th2 trade-off plot with heuristic points:
    1. th1/2 (*)  (fixed divide-by-2 baseline)
    2. th1/sqrt(k) (P)  (k-aware scaling: k=4 -> th1/2, k=5 -> th1/sqrt(5), ...)
    2. th1^2 (s)
    3. th1^1.5 (^)
    4. Online Sqrt (All) (D)
    5. Online Sqrt (LowConf-only update) (X)
    """
    # Default: prefix->cyclic, postfix->base (debias_pride.py와 동일)
    N = len(base_correct_list)
    if forced_cyclic_ids is not None:
        default_corrects = [
            cyclic_correct_list[i] if i in forced_cyclic_ids else base_correct_list[i]
            for i in range(N)
        ]
        default_acc = float(np.mean(np.asarray(default_corrects, dtype=np.float64)))
    else:
        default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    
    # Dense curve range (requested: 1..30)
    dense_th2_list = list(range(1, 31))
    
    # Real-world online points: th1 is online-quantile (past-only), th2 derived from th1 value
    def _online_point(th1_p: float, rule_func):
        return _run_online_th1_quantile_th2_from_th1_rule(
            default_conf=default_conf,
            mean_conf=mean_conf,
            base_correct=base_correct_list,
            cyclic_correct=cyclic_correct_list,
            probe2_correct=arr_probe2_correct,
            k=k,
            th1_percent=float(th1_p),
            th2_rule_from_th1_value=rule_func,
            forced_cyclic_ids=forced_cyclic_ids,
        )

    # Plot 1: Cost vs th2
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # 1) Dense Curve
        costs = []
        for th2p in dense_th2_list:
            # REAL-WORLD online curve: sweep th2_percent while th1_percent fixed
            c, _ = _run_online_avggap_policy(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids,
            )
            costs.append(c)
        ax1.plot(dense_th2_list, costs, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.6)

        # 2) Heuristics Points
        # (A) th1 / 2  (fixed baseline)
        c_half, _, p_half = _online_point(th1p, lambda x: x / 2.0)
        ax1.scatter([p_half], [c_half], marker='*', s=120, color=color, edgecolors='black', zorder=6, label='th1/2' if idx==0 else "")

        # (A2) th1 / sqrt(k)  (k-aware)
        c_sqrtk, _, p_sqrtk = _online_point(th1p, lambda x, kk=k: x / math.sqrt(float(kk)))
        ax1.scatter([p_sqrtk], [c_sqrtk], marker='P', s=110, color=color, edgecolors='black', zorder=6, label='th1/sqrt(k)' if idx==0 else "")
        
        # (B) th1 ^ 2
        c_sq, _, p_sq = _online_point(th1p, lambda x: x ** 2)
        ax1.scatter([p_sq], [c_sq], marker='s', s=80, color=color, edgecolors='black', zorder=6, label='th1^2' if idx==0 else "")

        # (C) th1 ^ 1.5
        c_pow, _, p_pow = _online_point(th1p, lambda x: x ** 1.5)
        ax1.scatter([p_pow], [c_pow], marker='^', s=90, color=color, edgecolors='black', zorder=6, label='th1^1.5' if idx==0 else "")

        # (D) Online Sqrt
        c_sqrt, _, p_sqrt = _run_online_sqrt_policy(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p, forced_cyclic_ids=forced_cyclic_ids
        )
        ax1.scatter([p_sqrt], [c_sqrt], marker='D', s=80, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (All)' if idx==0 else "")

        # (E) Online Sqrt (LowConf-only update)
        c_sqrt_lc, _, p_sqrt_lc = _run_online_sqrt_policy_lowconf_update(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p, forced_cyclic_ids=forced_cyclic_ids
        )
        ax1.scatter([p_sqrt_lc], [c_sqrt_lc], marker='X', s=70, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (LowConf-only)' if idx==0 else "")

    ax1.set_xlabel("th2 (percentile, avg gap)", fontsize=11)
    ax1.set_ylabel("Computational Cost (× of default)", fontsize=11)
    ax1.set_title(f"{getattr(args, 'task', 'task')} {subject} — Cost vs th2 ({plot_tag})", fontsize=12)
    ax1.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax1.set_xticklabels([str(t) for t in [1, 5, 10, 15, 20, 25, 30]])
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.4)
    suffix = f"_{str(fname_tag).strip()}" if str(fname_tag).strip() else ""
    out_cost = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST{suffix}.png")
    os.makedirs(os.path.dirname(out_cost), exist_ok=True)
    fig1.tight_layout()
    fig1.savefig(out_cost, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: Δ Accuracy (%) vs Cost   (trade-off axis)
    fig2, ax2 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # 1) Dense Curve (sweep th2 -> (cost, Δacc))
        costs_line = []
        delta_accs_line = []
        for th2p in dense_th2_list:
            c, a = _run_online_avggap_policy(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids,
            )
            costs_line.append(float(c))
            delta_accs_line.append((float(a) - float(default_acc)) * 100.0)
        ax2.plot(costs_line, delta_accs_line, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.5)

        # 2) Heuristics (points on cost axis)
        # (A) th1 / 2
        c_half, a_half, _ = _online_point(th1p, lambda x: x / 2.0)
        ax2.scatter([c_half], [(a_half-default_acc)*100], marker='*', s=120, color=color, edgecolors='black', zorder=6)

        # (A2) th1 / sqrt(k)
        c_sqrtk, a_sqrtk, _ = _online_point(th1p, lambda x, kk=k: x / math.sqrt(float(kk)))
        ax2.scatter([c_sqrtk], [(a_sqrtk-default_acc)*100], marker='P', s=110, color=color, edgecolors='black', zorder=6)
        
        # (B) th1 ^ 2
        c_sq, a_sq, _ = _online_point(th1p, lambda x: x ** 2)
        ax2.scatter([c_sq], [(a_sq-default_acc)*100], marker='s', s=80, color=color, edgecolors='black', zorder=6)

        # (C) th1 ^ 1.5
        c_pow, a_pow, _ = _online_point(th1p, lambda x: x ** 1.5)
        ax2.scatter([c_pow], [(a_pow-default_acc)*100], marker='^', s=90, color=color, edgecolors='black', zorder=6)

        # (D) Online Sqrt
        c_sqrt, a_sqrt, _ = _run_online_sqrt_policy(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        ax2.scatter([c_sqrt], [(a_sqrt-default_acc)*100], marker='D', s=80, color=color, edgecolors='black', zorder=6)

        # (E) Online Sqrt (LowConf-only update)
        c_sqrt_lc, a_sqrt_lc, _ = _run_online_sqrt_policy_lowconf_update(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        ax2.scatter([c_sqrt_lc], [(a_sqrt_lc-default_acc)*100], marker='X', s=70, color=color, edgecolors='black', zorder=6)

    ax2.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("Computational Cost (× of default)", fontsize=11)
    ax2.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax2.set_title(f"{getattr(args, 'task', 'task')} {subject} — Δ Accuracy vs Cost ({plot_tag})", fontsize=12)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.4)
    out_delta = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_DELTA_ACC{suffix}.png")
    fig2.tight_layout()
    fig2.savefig(out_delta, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: Cost vs Δ Accuracy
    fig3, ax3 = plt.subplots(figsize=(9.0, 6.0), dpi=160)
    
    for idx, th1p in enumerate(th1_list):
        th1p = float(th1p)
        color = colors[idx % len(colors)]
        
        # Curve
        costs_line = []
        delta_accs_line = []
        for th2p in dense_th2_list:
            c, a = _run_online_avggap_policy(
                default_conf=default_conf,
                mean_conf=mean_conf,
                base_correct=base_correct_list,
                cyclic_correct=cyclic_correct_list,
                probe2_correct=arr_probe2_correct,
                k=k,
                th1_percent=th1p,
                th2_percent=float(th2p),
                offline_prefix_n=0,
                forced_cyclic_ids=forced_cyclic_ids,
            )
            costs_line.append(c)
            delta_accs_line.append((a - default_acc) * 100.0)
        ax3.plot(costs_line, delta_accs_line, label=f'th1={int(th1p)}', color=color, linewidth=1.5, alpha=0.3)
        
        # Points
        # (A) th1 / 2
        c_h, a_h, p_h = _online_point(th1p, lambda x: x / 2.0)
        d_h = (a_h - default_acc) * 100.0
        ax3.scatter([c_h], [d_h], marker='*', s=120, color=color, edgecolors='black', zorder=6, label='th1/2' if idx==0 else "")

        # (A2) th1 / sqrt(k)
        c_hk, a_hk, p_hk = _online_point(th1p, lambda x, kk=k: x / math.sqrt(float(kk)))
        d_hk = (a_hk - default_acc) * 100.0
        ax3.scatter([c_hk], [d_hk], marker='P', s=110, color=color, edgecolors='black', zorder=6, label='th1/sqrt(k)' if idx==0 else "")

        # (B) th1 ^ 2
        c_s, a_s, p_s = _online_point(th1p, lambda x: x ** 2)
        d_s = (a_s - default_acc) * 100.0
        ax3.scatter([c_s], [d_s], marker='s', s=80, color=color, edgecolors='black', zorder=6, label='th1^2' if idx==0 else "")

        # (C) th1 ^ 1.5
        c_p, a_p, p_p = _online_point(th1p, lambda x: x ** 1.5)
        d_p = (a_p - default_acc) * 100.0
        ax3.scatter([c_p], [d_p], marker='^', s=90, color=color, edgecolors='black', zorder=6, label='th1^1.5' if idx==0 else "")

        # (D) Online Sqrt
        c_sqt, a_sqt, p_sqt = _run_online_sqrt_policy(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        d_sqt = (a_sqt - default_acc) * 100.0
        ax3.scatter([c_sqt], [d_sqt], marker='D', s=80, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (All)' if idx==0 else "")

        # (E) Online Sqrt (LowConf-only update)
        c_sqt_lc, a_sqt_lc, p_sqt_lc = _run_online_sqrt_policy_lowconf_update(
            default_conf, mean_conf, base_correct_list, cyclic_correct_list, arr_probe2_correct, k, th1p
        )
        d_sqt_lc = (a_sqt_lc - default_acc) * 100.0
        ax3.scatter([c_sqt_lc], [d_sqt_lc], marker='X', s=70, color=color, edgecolors='black', zorder=6,
                    label='Online Sqrt (LowConf-only)' if idx==0 else "")

        # Log (verbose only; this block is very long)
        if bool(getattr(args, "verbose", False)):
            logger.info(_purple(f"==== TH2 online-point report (th1={int(th1p)}) ===="))
            logger.info(f"default              : cost=1.000, acc={default_acc:.4f}")
            logger.info(f"th1/2                : cost={c_h:.3f}, acc={a_h:.4f}, th2≈p{p_h:.1f}")
            logger.info(f"th1/sqrt(k)          : cost={c_hk:.3f}, acc={a_hk:.4f}, th2≈p{p_hk:.1f}")
            logger.info(f"th1^2                : cost={c_s:.3f}, acc={a_s:.4f}, th2≈p{p_s:.1f}")
            logger.info(f"th1^1.5              : cost={c_p:.3f}, acc={a_p:.4f}, th2≈p{p_p:.1f}")
            logger.info(f"Online Sqrt (All)    : cost={c_sqt:.3f}, acc={a_sqt:.4f}, th2≈p{p_sqt:.1f}")
            logger.info(f"Online Sqrt (LowConf): cost={c_sqt_lc:.3f}, acc={a_sqt_lc:.4f}, th2≈p{p_sqt_lc:.1f}")

    ax3.scatter([1.0], [0.0], marker='*', s=200, label='default', color='gray', zorder=5)
    ax3.set_xlabel("Computational Cost (× of default)", fontsize=11)
    ax3.set_ylabel("Δ Accuracy (%)", fontsize=11)
    ax3.set_title(f"{getattr(args, 'task', 'task')} {subject} — Trade-off (All Heuristics, {plot_tag})", fontsize=12)
    ax3.legend(loc='best', fontsize=9, ncol=2)
    ax3.grid(True, linestyle='--', alpha=0.4)
    out_trade = os.path.join(curve_save_path, f"{subject}_th2_tradeoff_COST_vs_DELTA{suffix}.png")
    fig3.tight_layout()
    fig3.savefig(out_trade, bbox_inches="tight")
    plt.close(fig3)

    logger.info(_purple(f"th2 trade-off plots saved ({plot_tag}): {subject}"))
    if wandb_ok and wandb_run is not None:
        try:
            import wandb
            wandb_run.log({
                f"plots/{subject}/th2_tradeoff_COST{suffix}": wandb.Image(out_cost),
                f"plots/{subject}/th2_tradeoff_DELTA_ACC{suffix}": wandb.Image(out_delta),
                f"plots/{subject}/th2_tradeoff_COST_vs_DELTA{suffix}": wandb.Image(out_trade),
            })
        except Exception:
            pass


def _parse_percent_value_list(v) -> List[float]:
    if v is None:
        return [30.0]
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                pass
        return out if len(out) > 0 else [30.0]
    if isinstance(v, str):
        s = v.strip()
        if "," in s:
            out = []
            for t in s.split(","):
                t = t.strip()
                if t == "":
                    continue
                try:
                    out.append(float(t))
                except Exception:
                    pass
            return out if len(out) > 0 else [30.0]
        try:
            return [float(s)]
        except Exception:
            return [30.0]
    return [30.0]


def _parse_float_value_list(v, default: Optional[List[float]] = None) -> List[float]:
    fallback = list(default) if default is not None else [0.5]
    if v is None:
        return fallback
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                pass
        return out if out else fallback
    if isinstance(v, str):
        s = v.strip()
        if "," in s:
            out = []
            for tok in s.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    out.append(float(tok))
                except Exception:
                    pass
            return out if out else fallback
        try:
            return [float(s)]
        except Exception:
            return fallback
    return fallback


# =========================================================
# Curves: baseline (all methods)
# =========================================================
def _merge_curve_objs_over_runs(cobjs: List[dict]) -> Optional[dict]:
    """
    Average cost, acc, recall_std over multiple curve_objs (from n_runs).
    Returns one merged cobj with averaged numeric values.
    """
    if not cobjs or len(cobjs) == 0:
        return None
    if len(cobjs) == 1:
        return copy.deepcopy(cobjs[0])
    ref = cobjs[0]
    out = copy.deepcopy(ref)
    n = float(len(cobjs))
    # always
    for key in ["default", "cyclic", "full"]:
        if key in (out.get("always") or {}):
            costs = [float(c.get("always", {}).get(key, {}).get("cost", float("nan"))) for c in cobjs]
            accs = [float(c.get("always", {}).get(key, {}).get("acc", float("nan"))) for c in cobjs]
            costs = [x for x in costs if np.isfinite(x)]
            accs = [x for x in accs if np.isfinite(x)]
            if costs and accs:
                out["always"][key]["cost"] = float(np.mean(costs))
                out["always"][key]["acc"] = float(np.mean(accs))
    # cyclic_random_{fp}
    for k, v in list((out or {}).items()):
        if isinstance(k, str) and k.startswith("cyclic_random_") and not k.endswith("_recall_std"):
            if isinstance(v, dict) and "costs" in v and "accuracies" in v:
                costs = []
                accs = []
                for c in cobjs:
                    if k in c and isinstance(c[k], dict):
                        costs.append(float(c[k].get("costs", [float("nan")])[0]))
                        accs.append(float(c[k].get("accuracies", [float("nan")])[0]))
                costs = [x for x in costs if np.isfinite(x)]
                accs = [x for x in accs if np.isfinite(x)]
                if costs and accs:
                    out[k]["costs"] = [float(np.mean(costs))]
                    out[k]["accuracies"] = [float(np.mean(accs))]
    # *_recall_std
    for k in list((out or {}).keys()):
        if isinstance(k, str) and k.endswith("_recall_std") and isinstance(out.get(k), (int, float)):
            vals = [float(c.get(k, float("nan"))) for c in cobjs if k in c]
            vals = [x for x in vals if np.isfinite(x)]
            if vals:
                out[k] = float(np.mean(vals))
    # heuristic_points (group by (label, th1_p) when th1_p present, else by label only)
    hp_ref = ref.get("heuristic_points") or []
    if hp_ref:
        def _hp_key(h):
            lab = str(h.get("label")) if h.get("label") else ""
            for sweep_key in ("th1_p", "conf_th"):
                sweep_val = h.get(sweep_key)
                if sweep_val is not None:
                    return (lab, sweep_key, float(sweep_val))
            return (lab, None, None)

        keys_seen = set()
        for h in hp_ref:
            if isinstance(h, dict) and h.get("label"):
                keys_seen.add(_hp_key(h))
        merged_hp = []
        for (lab, sweep_key, sweep_val) in sorted(
            keys_seen,
            key=lambda k: (k[0], k[1] or "", (k[2] if k[2] is not None else -1.0)),
        ):
            costs, accs, rstds, nbs, np2s, ncs = [], [], [], [], [], []
            marker_ref, color_ref = "o", "black"
            for c in cobjs:
                for h in (c.get("heuristic_points") or []):
                    if not isinstance(h, dict):
                        continue
                    if str(h.get("label")) != lab:
                        continue
                    if sweep_key is not None and h.get(sweep_key) != sweep_val:
                        continue
                    if sweep_key is None and any(h.get(sk) is not None for sk in ("th1_p", "conf_th")):
                        continue
                    costs.append(float(h.get("cost", float("nan"))))
                    accs.append(float(h.get("acc", float("nan"))))
                    if "recall_std" in h:
                        rstds.append(float(h["recall_std"]))
                    if "n_base" in h:
                        nbs.append(int(h.get("n_base", 0)))
                    if "n_probe2" in h:
                        np2s.append(int(h.get("n_probe2", 0)))
                    if "n_cyclic" in h:
                        ncs.append(int(h.get("n_cyclic", 0)))
                    marker_ref = str(h.get("marker", "o"))
                    color_ref = str(h.get("color", "black"))
                    break
            costs = [x for x in costs if np.isfinite(x)]
            accs = [x for x in accs if np.isfinite(x)]
            if costs and accs:
                entry = {"label": lab, "cost": float(np.mean(costs)), "acc": float(np.mean(accs)), "marker": marker_ref, "color": color_ref}
                if sweep_key is not None:
                    entry[sweep_key] = sweep_val
                if rstds:
                    entry["recall_std"] = float(np.mean(rstds))
                if nbs:
                    entry["n_base"] = int(np.mean(nbs))
                if np2s:
                    entry["n_probe2"] = int(np.mean(np2s))
                if ncs:
                    entry["n_cyclic"] = int(np.mean(ncs))
                merged_hp.append(entry)
        out["heuristic_points"] = merged_hp
    # transition (Ours baseline): sum counts over runs
    trans_list = [c.get("transition") for c in cobjs if isinstance(c.get("transition"), dict)]
    if trans_list:
        out["transition"] = {
            "t_to_f_count": sum(int(t.get("t_to_f_count", 0)) for t in trans_list),
            "f_to_t_count": sum(int(t.get("f_to_t_count", 0)) for t in trans_list),
            "base_t_count": sum(int(t.get("base_t_count", 0)) for t in trans_list),
            "base_f_count": sum(int(t.get("base_f_count", 0)) for t in trans_list),
        }
    return out


def _compute_curves_for_one_percentile(
    subject: str,
    tag: str,
    k: int,
    perm_list: List[Tuple[int, ...]],
    base_correct_list: List[bool],
    cyclic_correct_list: List[bool],
    full_correct_list: List[bool],
    default_conf: np.ndarray,
    mean_conf: np.ndarray,
    flip_trigger: np.ndarray,
    probe2_correct: np.ndarray,
    perc_value: float,
    full_enabled: bool = True,
    forced_cyclic_ids: Optional[set] = None,
    labels_idx: Optional[List[int]] = None,
    base_pred_idx: Optional[List[int]] = None,
    cyclic_pred_idx: Optional[List[int]] = None,
    probe2_pred_idx: Optional[List[int]] = None,
    full_pred_idx: Optional[List[int]] = None,
    cyclic_fractions: Optional[List[float]] = None,
    run_seed_offset: int = 0,
) -> dict:
    """
    REAL-WORLD online evaluation (no beta / no offline prefix).
    [Modified] If forced_cyclic_ids (Prefix) is active, force Cost=k and Acc=Cyclic.
    """
    N = len(base_correct_list)
    if N == 0:
        return {}

    perc01 = float(max(0.0, min(100.0, perc_value))) / 100.0

    C_cyc = float(k)
    C_full = float(len(perm_list)) if full_enabled else float("nan")

    # Default ensemble: prefix -> cyclic, postfix -> base (debias_pride.py와 동일)
    if forced_cyclic_ids is not None:
        default_corrects = [
            cyclic_correct_list[i] if i in forced_cyclic_ids else base_correct_list[i]
            for i in range(N)
        ]
        default_acc = float(np.mean(np.asarray(default_corrects, dtype=np.float64)))
    else:
        default_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64)))
    cyclic_acc_always = float(np.mean(np.asarray(cyclic_correct_list, dtype=np.float64)))
    full_acc_always = float(np.mean(np.asarray(full_correct_list, dtype=np.float64))) if full_enabled and len(full_correct_list) == N else float("nan")

    # 1) switch policies (REAL-WORLD online)
    total_cost_sc = 0.0
    corrects_sc = 0
    total_cost_sf = 0.0
    corrects_sf = 0
    past_gaps: List[float] = []

    for i in range(N):
        # 1. Calculate Threshold based on PAST data
        if len(past_gaps) > 0:
            thresh = float(np.quantile(np.asarray(past_gaps, dtype=np.float64), perc01))
        else:
            thresh = float("-inf")
        
        # 2. Check if current sample is in Prefix (Investment Phase)
        is_forced = (forced_cyclic_ids is not None and int(i) in forced_cyclic_ids)
        
        # 3. Policy Decision (Ambiguous?)
        # Even if forced, we calculate this to simulate what the policy *would* have thought
        amb = (float(default_conf[i]) < thresh)

        # -----------------------------------------------------
        # Logic: switch_cyclic
        # -----------------------------------------------------
        if is_forced:
            # [Prefix] 무조건 Cyclic 수행 (Cost=k, Acc=Cyclic)
            c_step_sc = C_cyc
            corrects_sc += 1 if cyclic_correct_list[i] else 0
        else:
            # [Postfix] Policy 판단에 따름
            if amb:
                c_step_sc = C_cyc
                corrects_sc += 1 if cyclic_correct_list[i] else 0
            else:
                c_step_sc = 1.0
                corrects_sc += 1 if base_correct_list[i] else 0

        # -----------------------------------------------------
        # Logic: switch_full (if enabled)
        # -----------------------------------------------------
        if full_enabled and len(full_correct_list) == N:
            if is_forced:
                # [Prefix] Full 수행 (Cost=Full/Cyclic, Acc=Full)
                c_step_sf = C_full 
                corrects_sf += 1 if full_correct_list[i] else 0
            else:
                # [Postfix]
                if amb:
                    c_step_sf = C_full
                    corrects_sf += 1 if full_correct_list[i] else 0
                else:
                    c_step_sf = 1.0
                    corrects_sf += 1 if base_correct_list[i] else 0
        else:
            c_step_sf = 0.0

        # Update History (Observe the gap regardless of decision)
        total_cost_sc += float(c_step_sc)
        if full_enabled and len(full_correct_list) == N:
            total_cost_sf += float(c_step_sf)
            
        past_gaps.append(float(default_conf[i]))

    switch_cyclic_cost = total_cost_sc / float(N)
    switch_cyclic_acc = corrects_sc / float(N)
    switch_full_cost = (total_cost_sf / float(N)) if (full_enabled and len(full_correct_list) == N) else float("nan")
    switch_full_acc = (corrects_sf / float(N)) if (full_enabled and len(full_correct_list) == N) else float("nan")

    # 2) ours_top2flip / ours_avggap (REAL-WORLD online) + stats
    _, _, top2_stats = _run_online_top2flip_policy_with_stats(
        default_conf=default_conf,
        flip_trigger=flip_trigger,
        base_correct=base_correct_list,
        cyclic_correct=cyclic_correct_list,
        probe2_correct=probe2_correct,
        k=k,
        th1_percent=perc_value,
        offline_prefix_n=0,
        forced_cyclic_ids=forced_cyclic_ids,
    )
    c_top2, a_top2 = _run_online_top2flip_policy(
        default_conf=default_conf,
        flip_trigger=flip_trigger,
        base_correct=base_correct_list,
        cyclic_correct=cyclic_correct_list,
        probe2_correct=probe2_correct,
        k=k,
        th1_percent=perc_value,
        offline_prefix_n=0,
        forced_cyclic_ids=forced_cyclic_ids,
    )
    _, _, sc_stats = _run_online_switch_cyclic_with_stats(
        default_conf=default_conf,
        base_correct=base_correct_list,
        cyclic_correct=cyclic_correct_list,
        k=k,
        th1_percent=perc_value,
        offline_prefix_n=0,
        forced_cyclic_ids=forced_cyclic_ids,
    )
    c_avg, a_avg, avg_stats = _run_online_avggap_policy_with_stats(
        default_conf=default_conf,
        mean_conf=mean_conf,
        base_correct=base_correct_list,
        cyclic_correct=cyclic_correct_list,
        probe2_correct=probe2_correct,
        k=k,
        th1_percent=perc_value,
        th2_percent=perc_value,
        offline_prefix_n=0,
        forced_cyclic_ids=forced_cyclic_ids,
    )

    # Cyclic random fraction (for three-curves plot)
    # Default+PRIDE: alpha<100 → prefix cyclic, postfix base(보정). alpha>=100 → Cyclic과 동일(원본).
    cyclic_fractions = cyclic_fractions or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seed_base = _stable_u32_seed(str(subject), int(run_seed_offset))
    cyclic_random_costs: Dict[str, float] = {}
    cyclic_random_accs: Dict[str, float] = {}
    prefix_ids_set = set(int(x) for x in forced_cyclic_ids) if forced_cyclic_ids else set()
    for fp in cyclic_fractions:
        fp_f = float(fp)
        if prefix_ids_set and float(fp_f) == float(perc_value):
            c_r, a_r = _run_prefix_cyclic_postfix_base(
                base_correct_list, cyclic_correct_list, k, prefix_ids_set
            )
        else:
            fp_seed_off = int(fp_f) if float(fp_f).is_integer() else int(round(fp_f * 1000.0))
            c_r, a_r = _run_cyclic_random_fraction(
                base_correct_list, cyclic_correct_list, k, fp_f, seed_base + fp_seed_off
            )
        cyclic_random_costs[f"cyclic_random_{fp}"] = float(c_r)
        cyclic_random_accs[f"cyclic_random_{fp}"] = float(a_r)

    # Prefix overhead accounting for "always" ensembles
    default_cost_always = 1.0
    if forced_cyclic_ids is not None and N > 0:
        m = int(len(forced_cyclic_ids))
        if m > 0:
            # Default ensemble also paid 'k' for the prefix samples to estimate prior
            default_cost_always = 1.0 + (float(m) * (float(k) - 1.0)) / float(N)

    curve_obj = {
        "subject": subject,
        "tag": str(tag),
        "k": int(k),
        "percentile": float(perc_value),
        "n_samples": int(N),
        "default_accuracy": float(default_acc),

        "always": {
            "default": {"cost": float(default_cost_always), "acc": float(default_acc)},
            "cyclic": {"cost": float(C_cyc), "acc": float(cyclic_acc_always)},
        },

        "cyclic": {"costs": [float(C_cyc)], "accuracies": [float(cyclic_acc_always)]},
        **{key: {"costs": [cyclic_random_costs[key]], "accuracies": [cyclic_random_accs[key]]} for key in cyclic_random_costs},
        "switch_cyclic": {"costs": [float(switch_cyclic_cost)], "accuracies": [float(switch_cyclic_acc)], "stats": dict(sc_stats)},
        "ours_top2flip": {"costs": [float(c_top2)], "accuracies": [float(a_top2)], "stats": dict(top2_stats)},
        "ours_avggap": {"costs": [float(c_avg)], "accuracies": [float(a_avg)]},
        "ours_avggap_stats": dict(avg_stats),
    }
    # Optional: add recall_std when labels_idx and preds available
    if labels_idx is not None and base_pred_idx is not None and cyclic_pred_idx is not None and probe2_pred_idx is not None:
        try:
            _, _, preds_sc = _run_online_switch_cyclic_with_preds(
                default_conf, base_pred_idx, cyclic_pred_idx, labels_idx, k, perc_value, 0, forced_cyclic_ids
            )
            _, _, preds_top2 = _run_online_top2flip_policy_with_preds(
                default_conf, flip_trigger, base_pred_idx, cyclic_pred_idx, probe2_pred_idx, labels_idx, k, perc_value, 0, forced_cyclic_ids
            )
            _, _, preds_avg = _run_online_avggap_policy_with_preds(
                default_conf, mean_conf, base_pred_idx, cyclic_pred_idx, probe2_pred_idx, labels_idx, k, perc_value, perc_value, 0, forced_cyclic_ids
            )
            curve_obj["switch_cyclic_recall_std"] = float(_recall_std(labels_idx, preds_sc, k))
            curve_obj["ours_top2flip_recall_std"] = float(_recall_std(labels_idx, preds_top2, k))
            curve_obj["ours_avggap_recall_std"] = float(_recall_std(labels_idx, preds_avg, k))
            # Default: prefix->cyclic, postfix->base (debias_pride.py와 동일)
            default_pred_idx = (
                [cyclic_pred_idx[i] if i in forced_cyclic_ids else base_pred_idx[i] for i in range(N)]
                if forced_cyclic_ids is not None
                else base_pred_idx
            )
            curve_obj["default_recall_std"] = float(_recall_std(labels_idx, default_pred_idx, k))
            curve_obj["cyclic_recall_std"] = float(_recall_std(labels_idx, cyclic_pred_idx, k))
            for fp in cyclic_fractions:
                fp_f = float(fp)
                if forced_cyclic_ids is not None and float(fp_f) == float(perc_value):
                    # Default+PRIDE: prefix=cyclic, postfix=base (same as default_pred_idx)
                    preds_r = default_pred_idx
                else:
                    fp_seed_off = int(fp_f) if float(fp_f).is_integer() else int(round(fp_f * 1000.0))
                    _, _, preds_r = _run_cyclic_random_fraction_with_preds(
                        base_pred_idx, cyclic_pred_idx, labels_idx, k, fp_f, seed_base + fp_seed_off
                    )  # seed_base already includes run_seed_offset
                curve_obj[f"cyclic_random_{fp}_recall_std"] = float(_recall_std(labels_idx, preds_r, k))
            if full_enabled and full_pred_idx is not None and len(full_pred_idx) == len(labels_idx):
                curve_obj["full_recall_std"] = float(_recall_std(labels_idx, full_pred_idx, k))
        except Exception:
            pass
    if full_enabled:
        curve_obj["always"]["full"] = {"cost": float(C_full), "acc": float(full_acc_always)}
        curve_obj["full"] = {"costs": [float(C_full)], "accuracies": [float(full_acc_always)]}
        if full_enabled and len(full_correct_list) == N:
            curve_obj["switch_full"] = {"costs": [float(switch_full_cost)], "accuracies": [float(switch_full_acc)]}
    return curve_obj


def _api_call_cost(call):
    try:
        return float((call or {}).get("cost_usd", 0.0) or 0.0)
    except Exception:
        return 0.0


def _summarize_api_call_records(calls):
    usage_keys = (
        "input_tokens", "cached_input_tokens", "output_tokens",
        "reasoning_tokens", "total_tokens",
    )
    usage = {key: 0 for key in usage_keys}
    returned_models = {}
    total_cost = 0.0
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        call_usage = call.get("usage", {}) or {}
        for key in usage_keys:
            usage[key] += int(call_usage.get(key, 0) or 0)
        total_cost += _api_call_cost(call)
        returned = str(call.get("returned_model") or call.get("requested_model") or "")
        if returned:
            returned_models[returned] = returned_models.get(returned, 0) + 1
    requests = len(calls or [])
    return {
        "requests": int(requests),
        "cache_hits": int(sum(bool(c.get("cache_hit")) for c in (calls or []) if isinstance(c, dict))),
        "usage": usage,
        "cost_usd": float(total_cost),
        "returned_model": next(iter(returned_models)) if len(returned_models) == 1 else None,
        "returned_models": dict(sorted(returned_models.items())),
    }


def _collect_counterfactual_api_costs(points_payload, avg_call_cost, n_samples):
    """Extract every plotted E[T] series and attach deployment-cost equivalents."""
    out = {}

    def visit(node, path):
        if not isinstance(node, dict):
            return
        costs = node.get("cost")
        if isinstance(costs, list):
            axis_key = next(
                (key for key, value in node.items()
                 if key != "cost" and isinstance(value, list) and len(value) == len(costs)),
                None,
            )
            rows = []
            for idx, mean_t in enumerate(costs):
                try:
                    mean_t = float(mean_t)
                except Exception:
                    continue
                if not np.isfinite(mean_t):
                    continue
                rows.append({
                    "operating_value": node[axis_key][idx] if axis_key else idx,
                    "mean_permutations": mean_t,
                    "usd_per_sample": mean_t * float(avg_call_cost),
                    "total_usd": mean_t * float(avg_call_cost) * int(n_samples),
                })
            if rows:
                out["/".join(path) or "root"] = {
                    "axis": axis_key or "index",
                    "points": rows,
                }
        for key, value in node.items():
            if isinstance(value, dict):
                visit(value, path + [str(key)])

    visit(points_payload or {}, [])
    return out


def _api_scoring_metadata(args):
    mode = str(getattr(args, "api_scoring_mode", "topk_strict"))
    return {
        "scoring_mode": mode,
        "equal_label_bias": (
            float(getattr(args, "api_equal_label_bias", 100.0))
            if mode == "equal_label_bias" else None
        ),
    }


def _api_scoring_context(args):
    """Keep legacy strict request hashes stable; only constrained runs add context fields."""
    metadata = _api_scoring_metadata(args)
    return metadata if metadata["scoring_mode"] != "topk_strict" else {}


def _run_api_adaptive(args, model, wandb_ok=False, wandb_run=None):
    """Physically stop commercial API calls at the selected empirical policy stage."""
    alphas = [float(x) for x in _parse_percent_value_list(
        getattr(args, "plot_empirical_prefix_fractions", None)
    ) if 0.0 < float(x) <= 100.0]
    if len(alphas) != 1:
        raise ValueError("adaptive API runs require exactly one --plot_empirical_prefix_fractions value")
    alpha = float(alphas[0])
    percentile = float(args.api_adaptive_percentile)
    n_runs = 1 if bool(args.api_probe_only) else max(1, int(args.n_runs))
    task_payloads = {}

    for eval_name in args.eval_names:
        subjects, prep_few, prep_samples, prep_fn = prepare_eval(args, eval_name)
        if getattr(args, "api_cache_dir", None) is None:
            model.set_cache_dir(os.path.join(args.save_path, "api_cache", str(args.api_provider)))
        if args.setting not in {"full", "cyclic", "perm"}:
            raise ValueError("adaptive API requires a permutation setting; use --eval_names task,shots,full")
        subjects = select_api_probe_subjects(args, subjects)
        option_ids = list(args.option_id_set or ("ABCDE" if args.task == "csqa" else "ABCD"))
        k = len(option_ids)
        cyclic_schedule = _rotations(k)
        run_rows, trajectories = [], []

        for subject in subjects:
            for run_idx in range(n_runs):
                few = prep_few(subject, few_shot_seed=run_idx if n_runs > 1 else None)
                samples = prep_samples(subject)
                if n_runs > 1:
                    random.Random(run_idx + 42).shuffle(samples)
                if args.api_probe_only:
                    samples = samples[: int(args.api_probe_samples)]
                if not samples:
                    continue
                N = len(samples)
                seed = _stable_u32_seed(subject, int(args.pride_seed) + run_idx)
                prefix_n = max(1, int(round(N * alpha / 100.0)))
                prefix_ids = set(int(x) for x in np.random.default_rng(seed).choice(
                    np.arange(N, dtype=np.int64), size=prefix_n, replace=False
                ).tolist())
                model.set_context(
                    task=args.task,
                    subject=subject,
                    run_idx=run_idx,
                    eval_name=eval_name,
                    execution_mode="adaptive",
                    prompt_mode=str(args.api_prompt_mode),
                    **_api_scoring_context(args),
                )
                eval_fn = prep_fn(model, None, few)

                meta = []
                for pos, (inputs, options, ideal) in enumerate(samples):
                    pair = inputs[0] if isinstance(inputs, list) and inputs and isinstance(inputs[0], list) else inputs
                    sys_msg, user_prompt = pair
                    meta.append({
                        "sample_id": pos, "sys_msg": str(sys_msg),
                        "question": _extract_question_from_user_prompt(str(user_prompt)),
                        "options": list(options), "ideal": str(ideal),
                    })

                # Full cyclic Latin evidence is acquired only for calibration samples.
                probs_bank = [np.ones((k, k), dtype=np.float64) / k for _ in range(N)]
                prefix_calls = {}
                for pos in sorted(prefix_ids):
                    m = meta[pos]
                    prompts = []
                    for perm in cyclic_schedule:
                        opts = [m["options"][int(i)] for i in perm]
                        prompts.append([m["sys_msg"], _build_option_user_prompt(m["question"], opts, option_ids)])
                    result = eval_fn((pos, (prompts, m["options"], m["ideal"])), random.Random(0))
                    arr = np.asarray(result["data"]["probs"], dtype=np.float64)
                    if arr.shape != (k, k):
                        raise ValueError(f"calibration response shape={arr.shape}, expected={(k, k)}")
                    probs_bank[pos] = arr
                    prefix_calls[pos] = list(result["data"].get("api_calls") or [])

                if args.empirical_residual_model == "logistic_normal":
                    _, mu_hat, residual_bank, _, prior_meta = _estimate_logistic_normal_pride_bank(
                        probs_bank, list(range(k)), k, alpha / 100.0, seed,
                        args.empirical_logit_delta, args.empirical_mc_samples,
                        args.empirical_cov_shrinkage,
                    )
                else:
                    _, mu_hat, residual_bank, prior_meta = _estimate_empirical_pride_bank(
                        probs_bank, list(range(k)), k, alpha / 100.0, seed,
                        args.empirical_logit_delta,
                    )
                if args.empirical_residual_model in ("zero", "identify"):
                    residual_bank = np.zeros((1, k), dtype=np.float64)
                if set(int(x) for x in prior_meta.get("prefix_ids", [])) != prefix_ids:
                    raise RuntimeError("adaptive prefix selection diverged from the PriDe estimator")

                router = OnlinePercentileRouter(
                    k=k,
                    percentile=percentile,
                    schedule=args.empirical_stage_schedule,
                    gamma=args.empirical_stage_gamma,
                )
                stage_counts = {f"n_stage_{s}": 0 for s in range(1, k + 1)}
                corrects = total_stages = 0
                total_usd = 0.0
                confidences, correct_flags, true_probs = [], [], []

                for pos, m in enumerate(meta):
                    label_idx = option_ids.index(m["ideal"])
                    forced = pos in prefix_ids
                    if forced:
                        schedule = list(cyclic_schedule)
                        stage_probs = list(np.asarray(probs_bank[pos], dtype=np.float64))
                        calls = list(prefix_calls[pos])
                    else:
                        identity = [[m["sys_msg"], _build_option_user_prompt(
                            m["question"], m["options"], option_ids
                        )]]
                        result = eval_fn((pos, (identity, m["options"], m["ideal"])), random.Random(0))
                        stage_probs = [np.asarray(result["data"]["probs"][0], dtype=np.float64)]
                        calls = list(result["data"].get("api_calls") or [])
                        base_post, base_pred, _ = _compute_empirical_stage_posteriors(
                            np.asarray(stage_probs), [tuple(range(k))], mu_hat, residual_bank
                        )
                        order = np.argsort(np.asarray(base_post[0]))[::-1]
                        top1 = int(base_pred[0])
                        runner = int(order[1]) if len(order) > 1 else top1
                        schedule = _build_targeted_latin_schedule(k, top1, runner)

                        while len(stage_probs) < k:
                            _, _, conf_so_far = _compute_empirical_stage_posteriors(
                                np.asarray(stage_probs), schedule[:len(stage_probs)], mu_hat, residual_bank
                            )
                            stage_id = len(stage_probs)
                            if router.should_stop(stage_id, float(conf_so_far[-1])):
                                break
                            perm = schedule[len(stage_probs)]
                            opts = [m["options"][int(i)] for i in perm]
                            prompt = [[m["sys_msg"], _build_option_user_prompt(m["question"], opts, option_ids)]]
                            next_result = eval_fn((pos, (prompt, m["options"], m["ideal"])), random.Random(0))
                            stage_probs.append(np.asarray(next_result["data"]["probs"][0], dtype=np.float64))
                            calls.extend(list(next_result["data"].get("api_calls") or []))

                    posts, preds, confs = _compute_empirical_stage_posteriors(
                        np.asarray(stage_probs), schedule[:len(stage_probs)], mu_hat, residual_bank
                    )
                    stop_stage = len(stage_probs)
                    post = np.asarray(posts[-1], dtype=np.float64)
                    is_correct = int(preds[-1]) == label_idx
                    sample_usd = sum(_api_call_cost(c) for c in calls)
                    corrects += int(is_correct)
                    total_stages += stop_stage
                    total_usd += sample_usd
                    stage_counts[f"n_stage_{stop_stage}"] += 1
                    confidences.append(float(confs[-1]))
                    correct_flags.append(bool(is_correct))
                    true_probs.append(float(post[label_idx]))
                    router.observe(confs[:stop_stage])
                    trajectories.append({
                        "type": "api_adaptive_trajectory", "task": args.task,
                        "subject": subject, "run_idx": run_idx, "sample_pos": pos,
                        "prompt_mode": str(args.api_prompt_mode),
                        **_api_scoring_metadata(args),
                        "prefix_forced": forced, "stop_stage": stop_stage,
                        "pred": int(preds[-1]), "label": label_idx,
                        "correct": bool(is_correct), "confidence": float(confs[-1]),
                        "true_prob": float(post[label_idx]), "policy_cost_usd": sample_usd,
                        "request_hashes": [str(c.get("request_hash", "")) for c in calls],
                    })

                conf_arr = np.asarray(confidences, dtype=np.float64)
                corr_arr = np.asarray(correct_flags, dtype=bool)
                tp_arr = np.asarray(true_probs, dtype=np.float64)
                row = {
                    "subject": subject, "run_idx": run_idx, "n_samples": N,
                    "alpha": alpha, "percentile": percentile,
                    "prompt_mode": str(args.api_prompt_mode),
                    **_api_scoring_metadata(args),
                    "accuracy": corrects / N,
                    "nll": float(np.mean(-np.log(np.clip(tp_arr, 1e-12, 1.0)))),
                    "ece": float(_compute_ece(conf_arr, corr_arr, 10)),
                    "mean_permutations": total_stages / N,
                    "policy_cost_usd": total_usd,
                    "policy_cost_usd_per_sample": total_usd / N,
                    "stage_counts": stage_counts, "prefix_samples": len(prefix_ids),
                }
                run_rows.append(row)
                logger.info(_purple(
                    f"API adaptive {args.task}/{subject} run={run_idx}: acc={row['accuracy']:.4f}, "
                    f"nll={row['nll']:.4f}, ece={row['ece']:.4f}, "
                    f"E[T]={row['mean_permutations']:.4f}, policy_usd=${row['policy_cost_usd']:.6f}"
                ))

        total_n = sum(int(r["n_samples"]) for r in run_rows)
        if total_n <= 0:
            raise RuntimeError(f"adaptive evaluation produced no samples for {args.task}")
        weighted = lambda key: sum(float(r[key]) * int(r["n_samples"]) for r in run_rows) / total_n
        counts = {f"n_stage_{s}": sum(int(r["stage_counts"][f"n_stage_{s}"]) for r in run_rows)
                  for s in range(1, k + 1)}
        payload = {
            "version": 1, "task": args.task, "execution_mode": "adaptive",
            "prompt_mode": str(args.api_prompt_mode),
            **_api_scoring_metadata(args),
            "provider": args.api_provider, "requested_model": args.pretrained_model_path,
            "alpha": alpha, "percentile": percentile, "n_samples": total_n,
            "n_runs": n_runs, "accuracy": weighted("accuracy"),
            "nll": weighted("nll"), "ece": weighted("ece"),
            "mean_permutations": weighted("mean_permutations"),
            "policy_cost_usd": sum(float(r["policy_cost_usd"]) for r in run_rows),
            "policy_cost_usd_per_sample": weighted("policy_cost_usd_per_sample"),
            "stage_counts": counts, "runs": run_rows,
        }
        out_dir = build_results_dir(args, args.task, args.num_few_shot, "full")
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, f"{args.task}_api_adaptive_summary.json")
        trajectory_path = os.path.join(out_dir, f"{args.task}_api_adaptive_trajectories.jsonl")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(trajectory_path, "w", encoding="utf-8") as f:
            for row in trajectories:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload["summary_path"] = summary_path
        payload["trajectories_path"] = trajectory_path
        task_payloads[args.task] = payload

    api_summary = model.summary()
    for payload in task_payloads.values():
        payload["returned_model"] = api_summary.get("returned_model")
        payload["returned_models"] = dict(api_summary.get("returned_models") or {})
        payload["physical"] = dict(api_summary.get("physical") or {})
        payload["logical"] = dict(api_summary.get("logical") or {})
        payload["pricing"] = dict(api_summary.get("pricing") or {})
        payload["physical_scope"] = "current process (all requested tasks)"
        with open(payload["summary_path"], "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    if wandb_ok and wandb_run is not None:
        wandb_run.summary["api_evaluation_v1"] = task_payloads
        try:
            import wandb
            art = wandb.Artifact(f"api-adaptive-{args.model_name}-{wandb_run.id}", type="api_adaptive")
            for payload in task_payloads.values():
                art.add_file(payload["summary_path"])
                art.add_file(payload["trajectories_path"])
            for cache_idx, cache_dir in enumerate(api_summary.get("cache_dirs") or [str(model.cache_dir)]):
                calls_path = os.path.join(str(cache_dir), "calls.jsonl")
                diagnostics_path = os.path.join(str(cache_dir), "diagnostics.jsonl")
                if os.path.exists(calls_path):
                    art.add_file(calls_path, name=f"cache_{cache_idx}/calls.jsonl")
                if os.path.exists(diagnostics_path):
                    art.add_file(diagnostics_path, name=f"cache_{cache_idx}/diagnostics.jsonl")
            wandb_run.log_artifact(art)
        except Exception as e:
            logger.warning(f"W&B adaptive artifact logging failed: {e}")
    return task_payloads


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    hf_logging.set_verbosity_error()

    args = parse_arguments()
    global _EMPIRICAL_RESIDUAL_WEIGHTING, _EMPIRICAL_RESIDUAL_IDENT, _EMPIRICAL_RESIDUAL_IDENT_SHRINK
    global _EMPIRICAL_RESIDUAL_IDENT_PROJECT
    _EMPIRICAL_RESIDUAL_WEIGHTING = str(
        getattr(args, "empirical_residual_weighting", "uniform")).strip().lower()
    _EMPIRICAL_RESIDUAL_IDENT = (
        str(getattr(args, "empirical_residual_model", "")).strip().lower() == "identify"
    )
    _EMPIRICAL_RESIDUAL_IDENT_SHRINK = float(getattr(args, "empirical_ident_shrink", 1.0))
    _EMPIRICAL_RESIDUAL_IDENT_PROJECT = bool(getattr(args, "empirical_ident_project", False))
    if len(getattr(args, "eval_names", [])) == 0:
        return

    # -------- W&B init (optional) --------
    wandb_run = None
    wandb_ok = False
    if bool(getattr(args, "wandb", False)):
        try:
            import wandb
            wandb_ok = True
            project = getattr(args, "wandb_project", None) or "eval_clm"
            run_name = getattr(args, "wandb_run_name", None) or f"{getattr(args,'model_name','model')}-{args.eval_names[0]}"
            entity = getattr(args, "wandb_entity", None) or "capde"
            cfg = {
                "pretrained_model_path": getattr(args, "pretrained_model_path", None),
                "model_name": getattr(args, "model_name", None),
                "eval_names": getattr(args, "eval_names", None),
                "option_id_set": getattr(args, "option_id_set", None),
                "ours_low_conf_percent": getattr(args, "ours_low_conf_percent", None),
                "inference_backend": getattr(args, "inference_backend", "local"),
                "api_provider": getattr(args, "api_provider", None),
                "api_execution_mode": getattr(args, "api_execution_mode", None),
                "api_prompt_mode": getattr(args, "api_prompt_mode", "baseline"),
                "api_scoring_mode": getattr(args, "api_scoring_mode", "topk_strict"),
                "api_equal_label_bias": (
                    getattr(args, "api_equal_label_bias", None)
                    if getattr(args, "api_scoring_mode", "topk_strict") == "equal_label_bias"
                    else None
                ),
                "api_adaptive_percentile": getattr(args, "api_adaptive_percentile", None),
            }
            wandb_run = wandb.init(project=project, entity=entity, name=run_name, config=cfg)
            logger.info(_blue(f"W&B init ok: project={project}, entity={entity}, name={run_name}"))
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            wandb_run = None
            wandb_ok = False

    # -------- DEVELOP MODE: generate dummy curve points only --------
    if bool(getattr(args, "develop", False)):
        try:
            # Use the same fractions as real plotting
            cyclic_fracs = [int(x) for x in _parse_percent_value_list(getattr(args, "plot_cyclic_fractions", "0,10,20,30,40,50,60,70,80,90,100")) if 0 <= int(x) <= 100]
            pride_fracs = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_ours_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0]
        except Exception:
            cyclic_fracs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            pride_fracs = [0.5, 1.0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        logger.info(_orange("[develop] Skipping model/data eval. Writing dummy three-curves plots/points."))

        # Always-random seed per process execution (time/pid/wandb_run.id mixed)
        try:
            wid = (wandb_run.id if wandb_run is not None else "no_wandb")
            wid_u32 = int(zlib.adler32(str(wid).encode("utf-8"))) & 0xFFFFFFFF
        except Exception:
            wid_u32 = 0
        base_seed = (int(time.time_ns()) ^ (int(os.getpid()) << 16) ^ int(wid_u32)) & 0xFFFFFFFF

        for eval_name in (getattr(args, "eval_names", None) or []):
            try:
                parts = str(eval_name).split(",")
                task = str(parts[0]).strip()
                num_few_shot = int(parts[1]) if len(parts) > 1 and str(parts[1]).strip() else 0
                args.task = task
                args.num_few_shot = num_few_shot

                out_dir = build_results_dir(args, task=args.task, num_few_shot=args.num_few_shot, setting="full")
                os.makedirs(out_dir, exist_ok=True)

                # Randomness per eval_name, mixed with base_seed
                seed = (int(base_seed) ^ (int(zlib.adler32(str(eval_name).encode("utf-8"))) & 0xFFFFFFFF)) & 0xFFFFFFFF
                rng = np.random.default_rng(seed)

                # Build a single "subject" curve object that contains cyclic_random_{fp} keys.
                # (three-curves plotting expects these keys inside one list element)
                cobj_base = {}
                k = 4.0  # pretend 4-choice
                for fp in cyclic_fracs:
                    frac = float(fp) / 100.0
                    cost = 1.0 + frac * (k - 1.0) + float(rng.normal(0.0, 0.02))
                    acc = 0.45 + 0.25 * frac + float(rng.normal(0.0, 0.01))
                    acc = float(np.clip(acc, 0.0, 1.0))
                    rstd = 0.20 - 0.10 * frac + float(rng.normal(0.0, 0.005))
                    rstd = float(np.clip(rstd, 0.0, 1.0))
                    cobj_base[f"cyclic_random_{fp}"] = {"costs": [float(cost)], "accuracies": [float(acc)]}
                    cobj_base[f"cyclic_random_{fp}_recall_std"] = float(rstd)

                derived_records_by_p = {}
                for p in pride_fracs:
                    frac = float(p) / 100.0
                    # Heuristic point for OURS at each p
                    ours_cost = 1.0 + frac * 1.5 + float(rng.normal(0.0, 0.02))
                    ours_acc = 0.50 + 0.20 * frac + float(rng.normal(0.0, 0.01))
                    ours_acc = float(np.clip(ours_acc, 0.0, 1.0))
                    ours_rstd = 0.18 - 0.08 * frac + float(rng.normal(0.0, 0.006))
                    ours_rstd = float(np.clip(ours_rstd, 0.0, 1.0))

                    cobj = dict(cobj_base)
                    cobj["heuristic_points"] = [{
                        "label": LEGACY_OURS_LABEL,
                        "cost": float(ours_cost),
                        "acc": float(ours_acc),
                        "recall_std": float(ours_rstd),
                        "n_base": int(800 + rng.integers(0, 50)),
                        "n_probe2": int(150 + rng.integers(0, 50)),
                        "n_cyclic": int(50 + rng.integers(0, 50)),
                    }, {
                        "label": PRIMARY_OURS_LABEL,
                        "cost": float(ours_cost + 0.05),
                        "acc": float(np.clip(ours_acc + 0.01, 0.0, 1.0)),
                        "recall_std": float(np.clip(ours_rstd - 0.01, 0.0, 1.0)),
                        "n_base": int(780 + rng.integers(0, 50)),
                        "n_probe2": int(170 + rng.integers(0, 50)),
                        "n_cyclic": int(70 + rng.integers(0, 50)),
                    }]
                    derived_records_by_p[float(p)] = [cobj]  # 1 "subject"

                pride_prefix = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_prefix_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0] or [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
                empirical_prefix = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_empirical_prefix_fractions", None)) if 0.0 <= float(x) <= 100.0] or list(pride_prefix)
                derived_records_pride_by_p = {}
                derived_records_pride_by_alpha = {}
                for alpha in pride_prefix:
                    cobj_pr = {}
                    for p in pride_fracs:
                        frac = float(p) / 100.0
                        pride_cost = 1.0 + frac * 2.2 + float(rng.normal(0.0, 0.02))
                        pride_acc = 0.52 + 0.18 * frac + float(rng.normal(0.0, 0.01))
                        pride_acc = float(np.clip(pride_acc, 0.0, 1.0))
                        pride_rstd = 0.16 - 0.07 * frac + float(rng.normal(0.0, 0.006))
                        pride_rstd = float(np.clip(pride_rstd, 0.0, 1.0))
                        key = f"cyclic_random_{p}"
                        cobj_pr[key] = {"costs": [float(pride_cost)], "accuracies": [float(pride_acc)]}
                        cobj_pr[f"{key}_recall_std"] = float(pride_rstd)
                        cobj_pr.setdefault("heuristic_points", []).append({
                            "label": LEGACY_OURS_LABEL, "th1_p": p, "cost": float(pride_cost),
                            "acc": float(pride_acc), "recall_std": float(pride_rstd),
                            "n_base": 800, "n_probe2": 150, "n_cyclic": 50,
                        })
                        cobj_pr["heuristic_points"].append({
                            "label": PRIMARY_OURS_LABEL, "th1_p": p, "cost": float(pride_cost + 0.05),
                            "acc": float(np.clip(pride_acc + 0.01, 0.0, 1.0)), "recall_std": float(np.clip(pride_rstd - 0.01, 0.0, 1.0)),
                            "n_base": 780, "n_probe2": 170, "n_cyclic": 70,
                        })
                        sqrt_cost = 1.0 + frac * 1.9 + float(rng.normal(0.0, 0.02))
                        sqrt_acc = 0.53 + 0.19 * frac + float(rng.normal(0.0, 0.01))
                        sqrt_rstd = 0.15 - 0.06 * frac + float(rng.normal(0.0, 0.006))
                        cobj_pr["heuristic_points"].append({
                            "label": "online_sqrt_all", "th1_p": p, "cost": float(sqrt_cost),
                            "acc": float(np.clip(sqrt_acc, 0.0, 1.0)), "recall_std": float(np.clip(sqrt_rstd, 0.0, 1.0)),
                            "n_base": 800, "n_probe2": 150, "n_cyclic": 50,
                        })
                    derived_records_pride_by_alpha[alpha] = [cobj_pr]

                _plot_three_curves_acc_recall_std(
                    derived_records_by_p,
                    derived_records_pride_by_p,
                    derived_records_pride_by_alpha,
                    {},
                    out_dir,
                    args.task,
                    cyclic_fractions=cyclic_fracs,
                    pride_ours_fractions=pride_fracs,
                    pride_prefix_list=pride_prefix,
                    empirical_prefix_list=empirical_prefix,
                    wandb_ok=wandb_ok,
                    wandb_run=wandb_run,
                )
            except Exception as e:
                logger.warning(f"[develop] Failed to write dummy plots for eval_name='{eval_name}': {e}")

        # Finish W&B early (since we skip the rest of main)
        if wandb_ok and wandb_run is not None:
            try:
                import wandb
                logger.info(_blue("W&B: syncing and finishing run (develop)..."))
                wandb.finish()
                time.sleep(2)
            except Exception:
                pass
        return

    # -------- Inference backend --------
    api_backend = str(getattr(args, "inference_backend", "local")) == "api"
    if api_backend:
        first_eval = str(args.eval_names[0]).split(",") if args.eval_names else ["api", "0", "full"]
        first_result_dir = build_results_dir(
            args,
            first_eval[0],
            int(first_eval[1]),
            first_eval[2] if len(first_eval) > 2 else None,
        )
        initial_cache = getattr(args, "api_cache_dir", None) or os.path.join(
            first_result_dir, "api_cache", str(args.api_provider)
        )
        model = CommercialAPIClient(
            provider=args.api_provider,
            model=args.pretrained_model_path,
            cache_dir=initial_cache,
            timeout_seconds=args.api_timeout_seconds,
            max_retries=args.api_max_retries,
            max_cost_usd=args.api_max_cost_usd,
            max_requests=args.api_max_requests,
            force_requests=args.api_force_requests,
            base_url=args.api_base_url,
            input_usd_per_mtok=args.api_input_usd_per_mtok,
            cached_input_usd_per_mtok=args.api_cached_input_usd_per_mtok,
            output_usd_per_mtok=args.api_output_usd_per_mtok,
            scoring_mode=args.api_scoring_mode,
            equal_label_bias=args.api_equal_label_bias,
        )
        toker = None
        logger.info(
            _blue(
                f"API backend ready: provider={args.api_provider}, model={args.pretrained_model_path}, "
                f"mode={args.api_execution_mode}, "
                f"prompt_mode={args.api_prompt_mode}, "
                f"scoring_mode={args.api_scoring_mode}, "
                f"pricing_snapshot={model.pricing['snapshot_date']}, "
                f"max_cost={args.api_max_cost_usd}, max_requests={args.api_max_requests}"
            )
        )
    else:
        # 일부 LLaMA/Mistral 계열은 slow tokenizer 파일이 없으므로 fast 기본값을 쓴다.
        toker = AutoTokenizer.from_pretrained(
            args.pretrained_model_path,
            add_bos_token=False,
            add_eos_token=False,
            cache_dir=getattr(args, "cache_dir", None),
        )
        use_bf16 = bool(torch.cuda.is_available()) and bool(torch.cuda.is_bf16_supported())
        config = AutoConfig.from_pretrained(
            args.pretrained_model_path,
            cache_dir=getattr(args, "cache_dir", None),
        )
        model_type = getattr(config, "model_type", "").lower()
        model_cls = AutoModelForSeq2SeqLM if model_type in ("t5", "mt5", "umt5") else AutoModelForCausalLM
        model = model_cls.from_pretrained(
            args.pretrained_model_path,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            cache_dir=getattr(args, "cache_dir", None),
        )
        logging_cuda_memory_usage()

    if api_backend and str(args.api_execution_mode) == "adaptive":
        _run_api_adaptive(args, model, wandb_ok=wandb_ok, wandb_run=wandb_run)
        if wandb_ok and wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B adaptive finish failed: {e}")
        return

    offline_api_records_by_task = {}
    for eval_name in args.eval_names[::1]:
        (subjects, prepare_few_shot_samples,
         prepare_eval_samples, prepare_eval_fn) = prepare_eval(args, eval_name)
        if api_backend:
            if getattr(args, "api_cache_dir", None) is None:
                model.set_cache_dir(os.path.join(args.save_path, "api_cache", str(args.api_provider)))
            subjects = select_api_probe_subjects(args, subjects)

        # =========================
        # Aggregate (MMLU-style) summary over subjects
        # =========================
        eval_acc_records: List[dict] = []  # [{'subject':str,'corrects':int,'total':int,'acc':float}]
        derived_records_by_p: Dict[float, List[dict]] = {}  # p -> list of curve_obj (per subject)
        derived_records_pride_by_p: Dict[float, List[dict]] = {}  # legacy: p->list (Default+PRIDE 단일 alpha용)
        derived_records_pride_by_alpha: Dict[float, List[dict]] = {}  # alpha(0.5,1.0,2,5,...) -> list of PRIDE curve_obj
        derived_records_empirical_by_alpha: Dict[float, List[dict]] = {}  # alpha -> list of empirical PriDe curve_obj
        empirical_analysis_records: List[dict] = []
        pride_recall_std_records: List[dict] = []  # [{'subject':str,'rstd':float,'m':int,'N':int}]
        recall_std_vs_p_records: List[dict] = []  # [{'subject':str,'p':float,'method':str,'kind':str,'rstd':float}]
        n_runs = (
            1 if api_backend and bool(getattr(args, "api_probe_only", False))
            else max(1, int(getattr(args, "n_runs", 1)))
        )
        skip_per_subject_plots = (args.task == "mmlu" and len(subjects) > 1)

        # 논문 테이블용 Base T/F 기준 트랜지션 기록 (Cyclic & Full)
        transition_records_cyclic: List[dict] = []
        transition_records_full: List[dict] = []
        # Default+PRIDE, Ours+PRIDE, Ours (per perc 2~100) - 논문 Experiments/Analysis용
        transition_records_default_pride_by_p: Dict[float, List[dict]] = {}
        transition_records_ours_pride_by_p: Dict[float, List[dict]] = {}
        transition_records_ours_by_p: Dict[float, List[dict]] = {}
        sigma_analysis_baseline_records: List[dict] = []
        sigma_analysis_pride_by_alpha: Dict[float, List[dict]] = {}

        def _make_transition_record_from_preds(base_correct, pred_idx, labels_idx, conf_arr, subject):
            """base_correct: List[bool], pred_idx: List[int], labels_idx: List[int], conf_arr: np.ndarray"""
            N = len(base_correct)
            pred_correct = [int(pred_idx[i]) == int(labels_idx[i]) for i in range(N)]
            base_t_gaps, base_f_gaps = [], []
            t_to_f, f_to_t = 0, 0
            conf_flat = np.asarray(conf_arr, dtype=np.float64).ravel()
            for i in range(N):
                c = float(conf_flat[i]) if i < len(conf_flat) else 0.0
                if base_correct[i]:
                    base_t_gaps.append(c)
                    if not pred_correct[i]:
                        t_to_f += 1
                else:
                    base_f_gaps.append(c)
                    if pred_correct[i]:
                        f_to_t += 1
            return {"subject": str(subject), "base_t_gaps": base_t_gaps, "base_f_gaps": base_f_gaps,
                    "t_to_f_count": t_to_f, "f_to_t_count": f_to_t}

        for subject in subjects[::1]:
            # n_runs > 1이면 run별 다른 seed로 평가 (few-shot, run_seed 고정 해제)
            run_indices = list(range(n_runs))
            for run_idx in run_indices:
                use_run_suffix = (n_runs > 1)
                cached_path = (f'{args.save_path}/{subject}_run{run_idx}.jsonl' if use_run_suffix
                               else f'{args.save_path}/{subject}.jsonl')
                recompute_results = bool(getattr(args, "force", False)) or (
                    api_backend and bool(getattr(args, "api_force_requests", False))
                )
                maybe_use_cached = (not recompute_results) and os.path.exists(cached_path)

                few_shot_seed = run_idx if use_run_suffix else None
                run_seed = run_idx if use_run_suffix else None

                logger.info(_blue(f"Preparing: {subject}" + (f" [run {run_idx+1}/{n_runs}]" if use_run_suffix else "")))
                few_shot_samples = prepare_few_shot_samples(subject, few_shot_seed=few_shot_seed)
                eval_samples = prepare_eval_samples(subject)
                # n_runs > 1: 매 run마다 데이터 순서를 다르게 shuffle (재현 가능한 시드)
                if n_runs > 1:
                    shuffler = random.Random(int(run_idx) + 42)
                    shuffler.shuffle(eval_samples)
                eval_fn = prepare_eval_fn(model, toker, few_shot_samples)
                if api_backend and bool(getattr(args, "api_probe_only", False)):
                    max_samples = int(getattr(args, "api_probe_samples", 10))
                else:
                    max_samples = 100 if bool(getattr(args, 'test', False)) else None

                if api_backend:
                    model.set_context(
                        task=str(args.task), subject=str(subject), run_idx=int(run_idx),
                        eval_name=str(eval_name), execution_mode=str(args.api_execution_mode),
                        prompt_mode=str(args.api_prompt_mode),
                        **_api_scoring_context(args),
                    )

                use_cached = False
                results = []
                if maybe_use_cached:
                    cached_results = _read_results_file(cached_path)
                    cache_ok, cache_reason = _validate_cached_results(
                        cached_results,
                        num_eval_samples=len(eval_samples),
                        max_samples=max_samples,
                    )
                    if cache_ok:
                        logger.info(_blue(f"Using cached results: {cached_path}"))
                        results = cached_results
                        use_cached = True
                    else:
                        logger.warning(_orange(f"Ignoring incomplete/invalid cache: {cached_path} ({cache_reason})"))

                if not use_cached:
                    logger.info(_blue(f"Run started: {subject}" + (f" [run {run_idx+1}/{n_runs}]" if use_run_suffix else "")))
                    if api_backend:
                        n_threads = 1 if args.api_execution_mode == "adaptive" else max(1, int(args.api_concurrency))
                    else:
                        n_threads = torch.cuda.device_count()
                        n_threads = max(1, int(n_threads)) if 'falcon' not in args.pretrained_model_path else 1
                    results = eval_all_samples(
                        eval_fn, eval_samples,
                        name=f'{args.task},{args.num_few_shot},{args.setting},{subject}' + (f',run{run_idx}' if use_run_suffix else ''),
                        threads=n_threads,
                        max_num_samples=max_samples,
                        run_seed=run_seed,
                    )
                    gc.collect()
                    if not api_backend:
                        torch.cuda.empty_cache()

                if api_backend:
                    task_records = offline_api_records_by_task.setdefault(
                        str(args.task), {"n_samples": 0, "calls": []}
                    )
                    task_records["n_samples"] += len(results)
                    for result in results:
                        data = result.get("data", {}) if isinstance(result, dict) else {}
                        task_records["calls"].extend(
                            call for call in (data.get("api_calls") or []) if isinstance(call, dict)
                        )

                metrics = None
                if len(results) > 0:
                    if args.setting in ['perm', 'full', 'cyclic']:
                        if getattr(args, 'option_id_set', None):
                            option_ids = list(args.option_id_set)
                        else:
                            k_guess = len(results[0]['data']['options'])
                            option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                        k = len(option_ids)

                        # If results contain only k rotations (e.g., k>=5 full-permutation disabled),
                        # aggregate with rotations instead of factorial permutations.
                        probs_len0 = None
                        try:
                            probs_len0 = len(results[0]['data'].get('probs', []))
                        except Exception:
                            probs_len0 = None

                        if args.setting in ['perm', 'full']:
                            if probs_len0 == k:
                                logger.info(_orange(f"[Auto] Full permutation disabled or not provided (k={k}). Using cyclic rotations for aggregation."))
                                perm_list = _rotations(k)
                            else:
                                from itertools import permutations
                                perm_list = list(sorted(permutations(range(k))))
                        else:
                            perm_list = _rotations(k)

                        total = 0
                        corrects = 0
                        for r in results:
                            if r.get('type') != 'result':
                                continue
                            data = r['data']
                            probs_seq = data.get('probs', None)
                            if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                                continue
                            agg = _aggregate_probs_over_permutations(probs_seq, perm_list, k)
                            pred_letter = option_ids[int(np.argmax(agg))]
                            if pred_letter == data['ideal']:
                                corrects += 1
                            total += 1
                        acc = (corrects / total) if total > 0 else float('nan')
                        metrics = {'type': 'metric', 'data': {'accuracy': acc}}
                        logger.info(_purple(f"==== Ensemble report ({args.setting}) ===="))
                        logger.info(f"accuracy: {acc:.4f}")

                        # aggregate: (micro uses correct/total; macro uses per-subject acc mean)
                        if total > 0:
                            eval_acc_records.append({
                                "subject": str(subject),
                                "corrects": int(corrects),
                                "total": int(total),
                                "acc": float(acc),
                            })
                    else:
                        metrics = {'type': 'metric', 'data': {}}
                        metrics['data']['accuracy'] = get_accuracy(results)
                        metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)

                        # aggregate (base/noid/shuffle etc)
                        total_b = 0
                        corrects_b = 0
                        for r in results:
                            if r.get("type") != "result":
                                continue
                            data = r.get("data", {}) or {}
                            corr = data.get("correct", None)
                            if corr is None:
                                if ("sampled" in data) and ("ideal" in data):
                                    corr = (data.get("sampled") == data.get("ideal"))
                                else:
                                    continue
                            total_b += 1
                            corrects_b += 1 if bool(corr) else 0
                        if total_b > 0:
                            eval_acc_records.append({
                                "subject": str(subject),
                                "corrects": int(corrects_b),
                                "total": int(total_b),
                                "acc": float(corrects_b) / float(total_b),
                            })

                logger.info(_orange(f"Run completed: {subject}" + (f" [run {run_idx+1}/{n_runs}]" if use_run_suffix else "")))

                if not use_cached:
                    save_results(cached_path, results, metrics)
                    logger.info(f"Results saved: {subject}" + (f" [run {run_idx}]" if use_run_suffix else ""))

                # =========================================================
                # Derived policies & PRIDE_FREE (full or cyclic for MMLU aggregate plots)
                # =========================================================
                if args.setting in ('full', 'cyclic') and len(results) > 0:
                    try:
                        if getattr(args, 'option_id_set', None):
                            option_ids = list(args.option_id_set)
                        else:
                            k_guess = len(results[0]['data']['options'])
                            option_ids = list('ABCDE' if k_guess == 5 else 'ABCD')
                        k = len(option_ids)

                        # Determine whether full permutations exist in cached results.
                        probs_len0 = None
                        try:
                            probs_len0 = len(results[0]['data'].get('probs', []))
                        except Exception:
                            probs_len0 = None

                        full_enabled = (probs_len0 is not None and probs_len0 == math.factorial(k))
                        if full_enabled:
                            from itertools import permutations
                            perm_list = list(sorted(permutations(range(k))))
                        else:
                            # fallback to cyclic rotations only
                            logger.info(_orange(f"[Auto] k={k} full permutations not available. Running derived policies with cyclic rotations only."))
                            perm_list = _rotations(k)
                        identity_idx = perm_list.index(tuple(range(k)))

                        cyclic_indices = [
                            perm_list.index(tuple((i + s) % k for i in range(k)))
                            for s in range(k)
                        ]
                        cyc_perms = [tuple((i + s) % k for i in range(k)) for s in range(k)]
                        sample_prompt_meta: Dict[int, Dict[str, Any]] = {}
                        for sample_idx, sample_entry in enumerate(eval_samples):
                            probing_inputs_raw, raw_options, raw_ideal = sample_entry
                            if not probing_inputs_raw:
                                continue
                            first_prompt = probing_inputs_raw[0]
                            if not isinstance(first_prompt, list) or len(first_prompt) < 2:
                                continue
                            sys_msg_i = str(first_prompt[0])
                            user_prompt_i = str(first_prompt[1])
                            sample_prompt_meta[int(sample_idx)] = {
                                "idx": int(sample_idx),
                                "sys_msg": sys_msg_i,
                                "question": _extract_question_from_user_prompt(user_prompt_i),
                                "options": list(raw_options),
                                "ideal": str(raw_ideal),
                            }

                        # ---------- collect per-sample raw probs ----------
                        per_sample_probs = []
                        ideals = []
                        empirical_prompt_meta = []

                        # ---------- derived correctness lists (baseline) ----------
                        base_correct_list = []
                        cyclic_correct_list = []
                        full_correct_list = []
                        full_pred_idx_list = []  # argmax(agg_full) for recall_std when full_enabled

                        base_probs_list = []  # identity row (letter-space)
                        base_pred_idx_list = []     # argmax(base_probs) as index
                        cyclic_pred_idx_list = []   # argmax(agg_cyc) as index
                        probe2_pred_idx_list = []   # argmax(mean_probs) as index

                        cyclic_results = []
                        base_results = []

                        full_total = 0
                        full_corrects = 0
                        cyclic_total = 0
                        cyclic_corrects = 0

                        for r in results:
                            if r.get('type') != 'result':
                                continue
                            data = r['data']
                            probs_seq = data.get('probs')
                            if not isinstance(probs_seq, list) or len(probs_seq) != len(perm_list):
                                continue

                            probs_seq_np = np.asarray(probs_seq, dtype=np.float64)
                            per_sample_probs.append(probs_seq_np)

                            ideals.append(data['ideal'])
                            empirical_prompt_meta.append(
                                sample_prompt_meta.get(
                                    int(data.get("idx", len(empirical_prompt_meta))),
                                    {
                                        "idx": int(data.get("idx", len(empirical_prompt_meta))),
                                        "sys_msg": "",
                                        "question": "",
                                        "options": list(data.get("options", [])),
                                        "ideal": str(data.get("ideal", "")),
                                    },
                                )
                            )

                            # cyclic (k rotations)
                            cyc_probs = [probs_seq_np[idx] for idx in cyclic_indices]
                            cyclic_results.append({
                                'type': 'result',
                                'data': {
                                    'idx': data['idx'],
                                    'prompt': data.get('prompt'),
                                    'options': data['options'],
                                    'probs': [cp.tolist() for cp in cyc_probs],
                                    'ideal': data['ideal'],
                                },
                            })
                            agg_cyc = _aggregate_probs_over_permutations([cp.tolist() for cp in cyc_probs], cyc_perms, k)
                            pred_cyc = option_ids[int(np.argmax(agg_cyc))]
                            cyclic_pred_idx_list.append(int(np.argmax(agg_cyc)))
                            corr_cyc = (pred_cyc == data['ideal'])
                            cyclic_correct_list.append(corr_cyc)
                            cyclic_corrects += 1 if corr_cyc else 0
                            cyclic_total += 1

                            # base (identity only)
                            base_probs = np.asarray(probs_seq_np[identity_idx], dtype=np.float64)
                            base_probs_list.append(base_probs)
                            pred_base = option_ids[int(np.argmax(base_probs))]
                            base_pred_idx_list.append(int(np.argmax(base_probs)))
                            corr_base = (pred_base == data['ideal'])
                            base_correct_list.append(corr_base)
                            base_results.append({
                                'type': 'result',
                                'data': {
                                    'idx': data['idx'],
                                    'prompt': data.get('prompt'),
                                    'options': data['options'],
                                    'probs': base_probs.tolist(),
                                    'sampled': pred_base,
                                    'ideal': data['ideal'],
                                    'correct': corr_base,
                                },
                            })

                            # full (all perms) - only if available
                            if full_enabled:
                                agg_full = _aggregate_probs_over_permutations(probs_seq_np, perm_list, k)
                                pred_full_idx = int(np.argmax(agg_full))
                                pred_full = option_ids[pred_full_idx]
                                full_pred_idx_list.append(pred_full_idx)
                                corr_full = (pred_full == data['ideal'])
                                full_correct_list.append(corr_full)
                                full_corrects += 1 if corr_full else 0
                                full_total += 1

                        # ---------- confidence stats & probe triggers (baseline) ----------
                        labels_idx_for_curves = [option_ids.index(str(x)) for x in ideals]
                        default_conf = []          # base gap (letter-space)
                        mean_gap_list = []         # gap(mean(base,probe)) (content-space)
                        flip_trigger_mask = []     # pred_base != pred_probe (content-space)
                        probe2_correct_list = []   # correctness of argmax(mean_probs)
                        cyclic_gap_mean_list = []  # per-sample mean gap over cyclic rotations
                        cyclic_gap_std_list = []   # per-sample sigma over cyclic rotations

                        for i, bp in enumerate(base_probs_list):
                            bp = np.asarray(bp, dtype=np.float64)
                            vals = np.sort(bp)[::-1]
                            top1 = float(vals[0]) if vals.shape[0] > 0 else 0.0
                            top2 = float(vals[1]) if vals.shape[0] > 1 else 0.0
                            default_conf.append(top1 - top2)

                            shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(bp, k)
                            probe_perm_idx = cyclic_indices[shift]

                            probs_base_raw = per_sample_probs[i][identity_idx]
                            agg_base = _aggregate_probs_over_permutations([probs_base_raw.tolist()], [tuple(range(k))], k)

                            cyc_gaps_i = []
                            for cyc_local_idx, perm_idx in enumerate(cyclic_indices):
                                agg_cyc_single = _aggregate_probs_over_permutations(
                                    [per_sample_probs[i][perm_idx].tolist()],
                                    [cyc_perms[cyc_local_idx]],
                                    k,
                                )
                                cyc_gaps_i.append(_gap_of_distribution(np.asarray(agg_cyc_single, dtype=np.float64)))
                            cyc_gaps_arr = np.asarray(cyc_gaps_i, dtype=np.float64)
                            cyclic_gap_mean_list.append(float(np.mean(cyc_gaps_arr)) if cyc_gaps_arr.size > 0 else 0.0)
                            cyclic_gap_std_list.append(float(np.std(cyc_gaps_arr)) if cyc_gaps_arr.size > 0 else 0.0)

                            probs_probe_raw = per_sample_probs[i][probe_perm_idx]
                            agg_probe = _aggregate_probs_over_permutations([probs_probe_raw.tolist()], [cyc_perms[shift]], k)

                            mean_probs = (agg_base + agg_probe) / 2.0
                            vals_mean = np.sort(mean_probs)[::-1]
                            mean_gap = float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0
                            mean_gap_list.append(mean_gap)

                            pred_base_cs = option_ids[int(np.argmax(agg_base))]
                            pred_probe_cs = option_ids[int(np.argmax(agg_probe))]
                            flip_trigger_mask.append(pred_base_cs != pred_probe_cs)

                            pred2 = option_ids[int(np.argmax(mean_probs))]
                            probe2_pred_idx_list.append(int(np.argmax(mean_probs)))
                            probe2_correct_list.append(pred2 == ideals[i])

                        default_conf = np.asarray(default_conf, dtype=np.float64)
                        mean_conf = np.asarray(mean_gap_list, dtype=np.float64)
                        arr_flip_trigger = np.asarray(flip_trigger_mask, dtype=bool)
                        arr_probe2_correct = np.asarray(probe2_correct_list, dtype=bool)
                        cyclic_gap_mean = np.asarray(cyclic_gap_mean_list, dtype=np.float64)
                        cyclic_gap_std = np.asarray(cyclic_gap_std_list, dtype=np.float64)
                        sigma_analysis_baseline_records.append(
                            _build_sigma_analysis_record(
                                subject=subject,
                                default_conf=default_conf,
                                mean_conf=mean_conf,
                                cyclic_gap_mean=cyclic_gap_mean,
                                cyclic_gap_std=cyclic_gap_std,
                                flip_mask=arr_flip_trigger,
                            )
                        )

                        # Base T/F 그룹별 Gap 및 트랜지션 카운트 수집 (Cyclic & Full)
                        base_t_gaps_cyc, base_f_gaps_cyc = [], []
                        t_to_f_count_cyc, f_to_t_count_cyc = 0, 0
                        for bc, cc, conf in zip(base_correct_list, cyclic_correct_list, default_conf):
                            if bc:
                                base_t_gaps_cyc.append(float(conf))
                                if not cc:
                                    t_to_f_count_cyc += 1
                            else:
                                base_f_gaps_cyc.append(float(conf))
                                if cc:
                                    f_to_t_count_cyc += 1
                        transition_records_cyclic.append({
                            "subject": str(subject),
                            "base_t_gaps": base_t_gaps_cyc,
                            "base_f_gaps": base_f_gaps_cyc,
                            "t_to_f_count": t_to_f_count_cyc,
                            "f_to_t_count": f_to_t_count_cyc,
                        })

                        if full_enabled and len(full_correct_list) == len(base_correct_list):
                            base_t_gaps_full, base_f_gaps_full = [], []
                            t_to_f_count_full, f_to_t_count_full = 0, 0
                            for bc, fc, conf in zip(base_correct_list, full_correct_list, default_conf):
                                if bc:
                                    base_t_gaps_full.append(float(conf))
                                    if not fc:
                                        t_to_f_count_full += 1
                                else:
                                    base_f_gaps_full.append(float(conf))
                                    if fc:
                                        f_to_t_count_full += 1
                            transition_records_full.append({
                                "subject": str(subject),
                                "base_t_gaps": base_t_gaps_full,
                                "base_f_gaps": base_f_gaps_full,
                                "t_to_f_count": t_to_f_count_full,
                                "f_to_t_count": f_to_t_count_full,
                            })

                        # ---------- optional: PRIDE debiasing + n_runs averaging (like debiase_pride.py) ----------
                        empirical_enabled = bool(getattr(args, "empirical_pride", False))
                        pride_enabled = bool(getattr(args, "pride_mix", False) or empirical_enabled)
                        empirical_sweep_mode = str(getattr(args, "empirical_sweep_mode", "percentile")).strip().lower()
                        if empirical_sweep_mode not in {"percentile", "confidence"}:
                            empirical_sweep_mode = "percentile"
                        empirical_percentile_mode = str(getattr(args, "empirical_percentile_mode", "online")).strip().lower()
                        if empirical_percentile_mode not in {"online", "fixed_prefix"}:
                            empirical_percentile_mode = "online"
                        empirical_residual_model = str(getattr(args, "empirical_residual_model", "logistic_normal")).strip().lower()
                        if empirical_residual_model not in {"logistic_normal", "empirical", "zero", "identify"}:
                            empirical_residual_model = "logistic_normal"
                        empirical_stage_schedule = str(getattr(args, "empirical_stage_schedule", "sqrt")).strip().lower()
                        if empirical_stage_schedule not in {"flat", "sqrt"}:
                            empirical_stage_schedule = "sqrt"
                        empirical_stage_gamma = float(getattr(args, "empirical_stage_gamma", 0.5))
                        if not np.isfinite(empirical_stage_gamma) or empirical_stage_gamma <= 0.0:
                            empirical_stage_gamma = 0.5
                        empirical_mc_samples = max(1, int(getattr(args, "empirical_mc_samples", 64)))
                        empirical_cov_shrinkage = min(max(float(getattr(args, "empirical_cov_shrinkage", 0.1)), 0.0), 1.0)
                        empirical_transition_mode = str(getattr(args, "empirical_transition_mode", "latin")).strip().lower()
                        if empirical_transition_mode not in {"latin", "probe_cyclic", "cyclic_random", "cyclic_targeted", "cyclic_learned"}:
                            empirical_transition_mode = "latin"
                        empirical_skip_residual_on_cyclic = bool(getattr(args, "empirical_skip_residual_on_cyclic", False))
                        empirical_conf_thresholds = [
                            float(x) for x in _parse_float_value_list(
                                getattr(args, "empirical_conf_thresholds", "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"),
                                default=[0.5],
                            )
                            if 0.0 <= float(x) <= 1.0
                        ] or [0.5]
                        by_perc_baseline: Dict[float, List[dict]] = defaultdict(list)
                        by_pride_alpha: Dict[float, List[dict]] = defaultdict(list)  # alpha(0.5,1.0,2,5,...) -> [cobj]
                        by_empirical_alpha: Dict[float, List[dict]] = defaultdict(list)  # alpha -> [cobj]
                        curve_objs_baseline = []
                        curve_objs_pride = []
                        curve_objs_empirical = []

                        pride_prefix_list = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_prefix_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0]
                        if not pride_prefix_list:
                            pride_prefix_list = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
                        empirical_prefix_list = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_empirical_prefix_fractions", None)) if 0.0 <= float(x) <= 100.0]
                        if not empirical_prefix_list:
                            empirical_prefix_list = list(pride_prefix_list)
                        ours_th1_list = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_ours_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0]
                        if not ours_th1_list:
                            ours_th1_list = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

                        # n_runs>1이면 외부 run_idx당 1회만 (이미 다른 결과). n_runs==1이면 n_runs번 curve variation
                        inner_run_indices = [run_idx] if use_run_suffix else list(range(n_runs))
                        for run_idx_inner in inner_run_indices:
                            cyclic_fracs_run = [int(x) for x in _parse_percent_value_list(getattr(args, "plot_cyclic_fractions", "0,10,20,30,40,50,60,70,80,90,100")) if 0 <= x <= 100]

                            for perc in ours_th1_list:
                                perc = float(perc)

                                # --- 1) BASELINE Curves 계산 ---
                                cobj = _compute_curves_for_one_percentile(
                                    subject=subject, tag="baseline", k=k, perm_list=perm_list,
                                    base_correct_list=base_correct_list, cyclic_correct_list=cyclic_correct_list,
                                    full_correct_list=full_correct_list if full_enabled else [],
                                    default_conf=default_conf, mean_conf=mean_conf, flip_trigger=arr_flip_trigger,
                                    probe2_correct=arr_probe2_correct, perc_value=perc, full_enabled=bool(full_enabled),
                                    labels_idx=labels_idx_for_curves, base_pred_idx=base_pred_idx_list,
                                    cyclic_pred_idx=cyclic_pred_idx_list, probe2_pred_idx=probe2_pred_idx_list,
                                    full_pred_idx=full_pred_idx_list if full_enabled and len(full_pred_idx_list) == len(ideals) else None,
                                    cyclic_fractions=cyclic_fracs_run, run_seed_offset=run_idx_inner,
                                )

                                if cobj:
                                    by_perc_baseline[perc].append(cobj)
                                    def _get_static_pt(th1_p, rule_func, label_key, marker_key):
                                        c, a, th2p, st = _run_online_th1_quantile_th2_from_th1_rule_with_stats(
                                            default_conf, mean_conf, base_correct_list, cyclic_correct_list,
                                            arr_probe2_correct, k, th1_p, rule_func, None)
                                        out = {'cost': c, 'acc': a, 'label': label_key, 'marker': marker_key, 'color': 'gray'}
                                        try:
                                            _, _, _, preds = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                                default_conf, mean_conf, base_pred_idx_list, cyclic_pred_idx_list, probe2_pred_idx_list,
                                                labels_idx_for_curves, k, th1_p, rule_func, None)
                                            out['recall_std'] = float(_recall_std(labels_idx_for_curves, preds, k))
                                            out['n_base'], out['n_probe2'], out['n_cyclic'] = st['n_base'], st['n_probe2'], st['n_cyclic']
                                        except Exception:
                                            pass
                                        return out
                                    if "heuristic_points" not in cobj:
                                        cobj["heuristic_points"] = [
                                            _get_static_pt(perc, _rule_th1_half, LEGACY_OURS_LABEL, "*"),
                                            _get_static_pt(perc, _rule_th1_sqrt2, PRIMARY_OURS_LABEL, "v"),
                                        ]

                                    # Ours (baseline) transition 기록
                                    try:
                                        _, _, _, preds_ours = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                            default_conf, mean_conf, base_pred_idx_list, cyclic_pred_idx_list,
                                            probe2_pred_idx_list, labels_idx_for_curves, k, perc, _rule_th1_sqrt2, None)
                                        rec = _make_transition_record_from_preds(
                                            base_correct_list, preds_ours, labels_idx_for_curves, default_conf, subject)
                                        transition_records_ours_by_p.setdefault(perc, []).append(rec)
                                        # curve에 transition counts만 저장 (routing 리포트용)
                                        cobj["transition"] = {
                                            "t_to_f_count": rec["t_to_f_count"],
                                            "f_to_t_count": rec["f_to_t_count"],
                                            "base_t_count": len(rec["base_t_gaps"]),
                                            "base_f_count": len(rec["base_f_gaps"]),
                                        }
                                    except Exception:
                                        pass

                        # --- 2) PRIDE Curves: pride_alpha(2,5,10,20) x ours_th1(2..100) 분리 ---
                        if pride_enabled:
                            for pride_alpha in pride_prefix_list:
                                seed = _stable_u32_seed(str(subject), int(getattr(args, "pride_seed", 0)) + run_idx_inner)
                                pride_prior, pride_meta = _estimate_pride_prior_random_prefix_mean(
                                    per_sample_probs=per_sample_probs,
                                    cyclic_indices=cyclic_indices,
                                    k=k,
                                    prefix_ratio=float(pride_alpha) / 100.0,
                                    seed=seed,
                                )
                                prefix_ids_set = set(int(x) for x in (pride_meta.get("prefix_ids") or []))

                                base_correct_list_pr = []
                                cyclic_correct_list_pr = []
                                full_correct_list_pr = []
                                full_pred_idx_list_pr = []
                                default_conf_pr = []
                                mean_gap_list_pr = []
                                flip_trigger_mask_pr = []
                                probe2_correct_list_pr = []
                                base_pred_idx_list_pr = []
                                cyclic_pred_idx_list_pr = []
                                probe2_pred_idx_list_pr = []
                                cyclic_gap_mean_list_pr = []
                                cyclic_gap_std_list_pr = []

                                for i in range(len(per_sample_probs)):
                                    ps = np.asarray(per_sample_probs[i], dtype=np.float64)
                                    ps_corr = np.asarray([_pride_correct_row(ps[j], pride_prior) for j in range(ps.shape[0])], dtype=np.float64)

                                    # Cyclic
                                    cyc_probs_corr = [ps_corr[idx] for idx in cyclic_indices]
                                    agg_cyc_corr = _aggregate_probs_over_permutations([cp.tolist() for cp in cyc_probs_corr], cyc_perms, k)
                                    pred_cyc_corr = option_ids[int(np.argmax(agg_cyc_corr))]
                                    cyclic_pred_idx_list_pr.append(int(np.argmax(agg_cyc_corr)))
                                    cyclic_correct_list_pr.append(pred_cyc_corr == ideals[i])

                                    # Base
                                    base_row_corr = np.asarray(ps_corr[identity_idx], dtype=np.float64)
                                    pred_base_corr = option_ids[int(np.argmax(base_row_corr))]
                                    base_pred_idx_list_pr.append(int(np.argmax(base_row_corr)))
                                    base_correct_list_pr.append(pred_base_corr == ideals[i])

                                    # Full
                                    if full_enabled:
                                        agg_full_corr = _aggregate_probs_over_permutations(ps_corr, perm_list, k)
                                        pred_full_idx_pr = int(np.argmax(agg_full_corr))
                                        full_pred_idx_list_pr.append(pred_full_idx_pr)
                                        full_correct_list_pr.append(option_ids[pred_full_idx_pr] == ideals[i])

                                    # Gaps
                                    vals = np.sort(base_row_corr)[::-1]
                                    default_conf_pr.append((float(vals[0]) if len(vals) > 0 else 0.0) - (float(vals[1]) if len(vals) > 1 else 0.0))

                                    shift, _, _ = _probe_shift_cyclic_put_top2_into_top1_slot(base_row_corr, k)
                                    probe_perm_idx = cyclic_indices[shift]
                                    agg_base = _aggregate_probs_over_permutations([base_row_corr.tolist()], [tuple(range(k))], k)

                                    cyc_gaps_i_pr = []
                                    for cyc_local_idx, perm_idx in enumerate(cyclic_indices):
                                        agg_cyc_single_pr = _aggregate_probs_over_permutations(
                                            [ps_corr[perm_idx].tolist()],
                                            [cyc_perms[cyc_local_idx]],
                                            k,
                                        )
                                        cyc_gaps_i_pr.append(_gap_of_distribution(np.asarray(agg_cyc_single_pr, dtype=np.float64)))
                                    cyc_gaps_arr_pr = np.asarray(cyc_gaps_i_pr, dtype=np.float64)
                                    cyclic_gap_mean_list_pr.append(float(np.mean(cyc_gaps_arr_pr)) if cyc_gaps_arr_pr.size > 0 else 0.0)
                                    cyclic_gap_std_list_pr.append(float(np.std(cyc_gaps_arr_pr)) if cyc_gaps_arr_pr.size > 0 else 0.0)

                                    probe_row_corr = np.asarray(ps_corr[probe_perm_idx], dtype=np.float64)
                                    agg_probe = _aggregate_probs_over_permutations([probe_row_corr.tolist()], [cyc_perms[shift]], k)

                                    mean_probs = (np.asarray(agg_base, dtype=np.float64) + np.asarray(agg_probe, dtype=np.float64)) / 2.0
                                    vals_mean = np.sort(mean_probs)[::-1]
                                    mean_gap_list_pr.append(float(vals_mean[0] - vals_mean[1]) if len(vals_mean) > 1 else 0.0)

                                    pred_base_cs = option_ids[int(np.argmax(agg_base))]
                                    pred_probe_cs = option_ids[int(np.argmax(agg_probe))]
                                    flip_trigger_mask_pr.append(pred_base_cs != pred_probe_cs)

                                    pred2 = option_ids[int(np.argmax(mean_probs))]
                                    probe2_pred_idx_list_pr.append(int(np.argmax(mean_probs)))
                                    probe2_correct_list_pr.append(pred2 == ideals[i])

                                default_conf_pr = np.asarray(default_conf_pr, dtype=np.float64)
                                mean_conf_pr = np.asarray(mean_gap_list_pr, dtype=np.float64)
                                arr_flip_trigger_pr = np.asarray(flip_trigger_mask_pr, dtype=bool)
                                arr_probe2_correct_pr = np.asarray(probe2_correct_list_pr, dtype=bool)
                                cyclic_gap_mean_pr = np.asarray(cyclic_gap_mean_list_pr, dtype=np.float64)
                                cyclic_gap_std_pr = np.asarray(cyclic_gap_std_list_pr, dtype=np.float64)
                                sigma_analysis_pride_by_alpha.setdefault(float(pride_alpha), []).append(
                                    _build_sigma_analysis_record(
                                        subject=subject,
                                        default_conf=default_conf_pr,
                                        mean_conf=mean_conf_pr,
                                        cyclic_gap_mean=cyclic_gap_mean_pr,
                                        cyclic_gap_std=cyclic_gap_std_pr,
                                        flip_mask=arr_flip_trigger_pr,
                                    )
                                )

                                # alpha>=100: prefix=전체 → 보정 불가. Cyclic permutation과 동일 (원본 사용)
                                base_for_dp = base_correct_list if pride_alpha >= 100 else base_correct_list_pr
                                cyclic_for_dp = cyclic_correct_list if pride_alpha >= 100 else cyclic_correct_list_pr
                                base_pred_dp = base_pred_idx_list if pride_alpha >= 100 else base_pred_idx_list_pr
                                cyclic_pred_dp = cyclic_pred_idx_list if pride_alpha >= 100 else cyclic_pred_idx_list_pr

                                cobj_pr = _compute_curves_for_one_percentile(
                                    subject=subject, tag="pride_mix", k=k, perm_list=perm_list,
                                    base_correct_list=base_for_dp, cyclic_correct_list=cyclic_for_dp,
                                    full_correct_list=full_correct_list_pr if full_enabled else [],
                                    default_conf=default_conf_pr, mean_conf=mean_conf_pr,
                                    flip_trigger=arr_flip_trigger_pr, probe2_correct=arr_probe2_correct_pr,
                                    perc_value=float(pride_alpha), full_enabled=bool(full_enabled),
                                    forced_cyclic_ids=prefix_ids_set, labels_idx=labels_idx_for_curves,
                                    base_pred_idx=base_pred_dp, cyclic_pred_idx=cyclic_pred_dp,
                                    probe2_pred_idx=probe2_pred_idx_list_pr,
                                    full_pred_idx=full_pred_idx_list_pr if full_enabled and len(full_pred_idx_list_pr) == len(ideals) else None,
                                    cyclic_fractions=[pride_alpha],
                                    run_seed_offset=run_idx_inner,
                                )
                                if cobj_pr:
                                    # alpha>=100: cost/acc/recall_std 모두 Cyclic과 동일하게 원본 사용
                                    bc_use, cc_use = (base_for_dp, cyclic_for_dp) if pride_alpha >= 100 else (base_correct_list_pr, cyclic_correct_list_pr)
                                    bp_use, cp_use = (base_pred_dp, cyclic_pred_dp) if pride_alpha >= 100 else (base_pred_idx_list_pr, cyclic_pred_idx_list_pr)
                                    def _get_static_pt_pride(th1_p, rule_func, label_key, marker_key):
                                        c, a, th2p, st = _run_online_th1_quantile_th2_from_th1_rule_with_stats(
                                            default_conf_pr, mean_conf_pr, bc_use, cc_use,
                                            arr_probe2_correct_pr, k, th1_p, rule_func, prefix_ids_set)
                                        out = {'cost': c, 'acc': a, 'label': label_key, 'th1_p': th1_p, 'marker': marker_key, 'color': 'gray'}
                                        try:
                                            _, _, _, preds = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                                default_conf_pr, mean_conf_pr, bp_use, cp_use,
                                                probe2_pred_idx_list_pr, labels_idx_for_curves, k, th1_p, rule_func, prefix_ids_set)
                                            out['recall_std'] = float(_recall_std(labels_idx_for_curves, preds, k))
                                            out['n_base'], out['n_probe2'], out['n_cyclic'] = st['n_base'], st['n_probe2'], st['n_cyclic']
                                        except Exception:
                                            pass
                                        return out
                                    def _get_static_pt_online_sqrt(th1_p):
                                        c, a, th2p, st = _run_online_sqrt_policy_with_stats(
                                            default_conf_pr, mean_conf_pr, bc_use, cc_use,
                                            arr_probe2_correct_pr, k, th1_p, prefix_ids_set)
                                        out = {'cost': c, 'acc': a, 'label': 'online_sqrt_all', 'th1_p': th1_p, 'marker': 'D', 'color': 'gray'}
                                        try:
                                            _, _, preds = _run_online_sqrt_policy_with_preds(
                                                default_conf_pr, mean_conf_pr, bp_use, cp_use,
                                                probe2_pred_idx_list_pr, labels_idx_for_curves, k, th1_p, prefix_ids_set)
                                            out['recall_std'] = float(_recall_std(labels_idx_for_curves, preds, k))
                                            out['n_base'], out['n_probe2'], out['n_cyclic'] = st['n_base'], st['n_probe2'], st['n_cyclic']
                                        except Exception:
                                            pass
                                        return out
                                    pts_th12 = [_get_static_pt_pride(float(th1), _rule_th1_half, LEGACY_OURS_LABEL, "*") for th1 in ours_th1_list]
                                    pts_var = [_get_static_pt_pride(float(th1), _rule_th1_sqrt2, PRIMARY_OURS_LABEL, "v") for th1 in ours_th1_list]
                                    pts_sqrt = [_get_static_pt_online_sqrt(float(th1)) for th1 in ours_th1_list]
                                    cobj_pr["heuristic_points"] = pts_th12 + pts_var + pts_sqrt
                                    by_pride_alpha[pride_alpha].append(cobj_pr)

                                for ours_th1 in ours_th1_list:
                                    try:
                                        th1_f = float(ours_th1)
                                        th1_seed_off = int(th1_f) if th1_f.is_integer() else int(round(th1_f * 1000.0))
                                        seed_cyc = _stable_u32_seed(str(subject), int(run_idx_inner)) + int(th1_seed_off)
                                        _, _, preds_dp = _run_cyclic_random_fraction_with_preds(
                                            base_pred_idx_list_pr, cyclic_pred_idx_list_pr,
                                            labels_idx_for_curves, k, th1_f, seed_cyc)
                                        rec_dp = _make_transition_record_from_preds(
                                            base_correct_list_pr, preds_dp, labels_idx_for_curves, default_conf_pr, subject)
                                        transition_records_default_pride_by_p.setdefault(float(th1_f), []).append(rec_dp)
                                    except Exception:
                                        pass
                                    try:
                                        _, _, _, preds_op = _run_online_th1_quantile_th2_from_th1_rule_with_preds(
                                            default_conf_pr, mean_conf_pr, base_pred_idx_list_pr, cyclic_pred_idx_list_pr,
                                            probe2_pred_idx_list_pr, labels_idx_for_curves, k, ours_th1, _rule_th1_sqrt2, prefix_ids_set)
                                        rec_op = _make_transition_record_from_preds(
                                            base_correct_list_pr, preds_op, labels_idx_for_curves, default_conf_pr, subject)
                                        if float(pride_alpha) == 2.0:
                                            transition_records_ours_pride_by_p.setdefault(float(th1_f), []).append(rec_op)
                                    except Exception:
                                        pass

                        if empirical_enabled:
                            empirical_logit_delta = float(getattr(args, "empirical_logit_delta", 1e-12))
                            empirical_base_seed = int(getattr(args, "pride_seed", 0))
                            for pride_alpha in empirical_prefix_list:
                                empirical_seed = _stable_u32_seed(str(subject), empirical_base_seed + run_idx_inner)
                                if empirical_residual_model == "logistic_normal":
                                    _, empirical_mu_hat, empirical_residual_bank, empirical_covariance, empirical_meta = _estimate_logistic_normal_pride_bank(
                                        per_sample_probs=per_sample_probs,
                                        cyclic_indices=cyclic_indices,
                                        k=k,
                                        prefix_ratio=float(pride_alpha) / 100.0,
                                        seed=empirical_seed,
                                        logit_delta=empirical_logit_delta,
                                        mc_samples=empirical_mc_samples,
                                        shrinkage_lambda=empirical_cov_shrinkage,
                                    )
                                else:
                                    _, empirical_mu_hat, empirical_residual_bank, empirical_meta = _estimate_empirical_pride_bank(
                                        per_sample_probs=per_sample_probs,
                                        cyclic_indices=cyclic_indices,
                                        k=k,
                                        prefix_ratio=float(pride_alpha) / 100.0,
                                        seed=empirical_seed,
                                        logit_delta=empirical_logit_delta,
                                    )
                                    empirical_covariance = np.zeros((k, k), dtype=np.float64)
                                if empirical_residual_model in ("zero", "identify"):
                                    # zero: eps=0 ablation. identify: bank unused; the
                                    # per-question residual is estimated per stage inside
                                    # _compute_empirical_stage_posteriors (mu-only before 3 views).
                                    empirical_residual_bank = np.zeros((1, k), dtype=np.float64)
                                empirical_prefix_ids = set(int(x) for x in (empirical_meta.get("prefix_ids") or []))
                                empirical_stage_cache_path = _empirical_stage_cache_path(
                                    args,
                                    subject=str(subject),
                                    run_idx=int(run_idx),
                                    use_run_suffix=bool(use_run_suffix),
                                    alpha=float(pride_alpha),
                                )
                                empirical_stage_cache = _load_empirical_stage_cache(empirical_stage_cache_path)
                                empirical_stage_cache_hits = 0
                                empirical_stage_cache_misses = 0
                                if empirical_stage_cache:
                                    logger.info(
                                        _blue(
                                            f"Loaded empirical stage cache: {empirical_stage_cache_path} "
                                            f"({len(empirical_stage_cache)} samples)"
                                        )
                                    )
                                learned_selection_info = None
                                if empirical_transition_mode == "cyclic_learned":
                                    if len(cyclic_indices) != int(k):
                                        raise ValueError(
                                            f"cyclic_learned requires exactly {k} cyclic permutations, got {len(cyclic_indices)} "
                                            f"(subject={subject}, alpha={pride_alpha:g})"
                                        )
                                    learned_selection_info = _select_best_relative_cyclic_sequence(
                                        sample_indices=sorted(int(x) for x in empirical_prefix_ids),
                                        per_sample_probs=per_sample_probs,
                                        cyclic_indices=cyclic_indices,
                                        cyc_perms=cyc_perms,
                                        mu_hat=empirical_mu_hat,
                                        residual_bank=empirical_residual_bank,
                                        labels_idx=labels_idx_for_curves,
                                    )
                                    logger.info(
                                        _blue(
                                            f"Empirical learned cyclic selection: subject={subject}, alpha={float(pride_alpha):g}, "
                                            f"run={int(run_idx_inner)}, seq={learned_selection_info.get('selected_sequence_name')}, "
                                            f"actions={','.join(learned_selection_info.get('selected_action_sequence') or [])}, "
                                            f"n_val={int(learned_selection_info.get('n_validation', 0))}"
                                        )
                                    )
                                empirical_stage_infos = []

                                for sample_pos, prompt_meta in enumerate(empirical_prompt_meta):
                                    if not prompt_meta.get("question"):
                                        raise ValueError(f"Empirical PriDe prompt metadata missing question for sample position {sample_pos}")

                                    label_idx_emp = int(labels_idx_for_curves[sample_pos])
                                    base_row = np.asarray(per_sample_probs[sample_pos][identity_idx], dtype=np.float64)
                                    base_posterior, base_pred_stage, base_conf_stage = _compute_empirical_stage_posteriors(
                                        stage_probs=base_row.reshape(1, -1),
                                        slot_to_content_schedule=[tuple(range(k))],
                                        mu_hat=empirical_mu_hat,
                                        residual_bank=empirical_residual_bank,
                                    )
                                    corrected_stage1 = np.asarray(base_posterior[0], dtype=np.float64)
                                    sorted_idx = np.argsort(corrected_stage1)[::-1]
                                    top1_idx = int(base_pred_stage[0])
                                    runner_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else int(top1_idx)
                                    if empirical_transition_mode == "latin" or empirical_transition_mode == "probe_cyclic":
                                        stage_schedule = _build_targeted_latin_schedule(k, top1_idx, runner_idx)
                                        stage_shifts = None
                                    elif empirical_transition_mode == "cyclic_learned":
                                        selected_actions = tuple()
                                        if learned_selection_info is not None:
                                            selected_actions = tuple(
                                                tuple(int(tok) for tok in action_str.replace("A_", "").split("to"))
                                                for action_str in (learned_selection_info.get("selected_action_sequence") or [])
                                                if isinstance(action_str, str) and action_str.startswith("A_") and "to" in action_str
                                            )
                                        initial_rank = [int(x) for x in sorted_idx.tolist()]
                                        stage_shifts, stage_schedule = _build_relative_action_cyclic_schedule(
                                            k=int(k),
                                            initial_rank=initial_rank,
                                            actions=selected_actions,
                                        )
                                    else:
                                        transition_seed = _stable_u32_seed(
                                            f"{subject}:{sample_pos}:{top1_idx}:{runner_idx}:{empirical_transition_mode}",
                                            empirical_seed,
                                        )
                                        stage_schedule = _build_incremental_cyclic_schedule(
                                            k=k,
                                            top1_idx=top1_idx,
                                            runner_idx=runner_idx,
                                            mode=empirical_transition_mode,
                                            seed=transition_seed,
                                        )
                                        stage_shifts = None
                                    raw_options = list(prompt_meta["options"])
                                    empirical_api_calls = []
                                    cached_stage_row = empirical_stage_cache.get(int(sample_pos))
                                    if empirical_transition_mode == "probe_cyclic":
                                        probe_slot_to_content = stage_schedule[1]
                                        all_stage_probs = _cached_empirical_stage_probs(
                                            cached_stage_row,
                                            sample_id=int(prompt_meta["idx"]),
                                            k=int(k),
                                            stage_schedule=stage_schedule,
                                        )
                                        if (
                                            api_backend and all_stage_probs is not None
                                            and (bool(args.api_force_requests) or not cached_stage_row.get("api_calls"))
                                        ):
                                            all_stage_probs = None
                                        if all_stage_probs is None:
                                            probing_inputs_emp = []
                                            for slot_to_content in stage_schedule[1:]:
                                                permuted_options = [raw_options[int(content_idx)] for content_idx in slot_to_content]
                                                probing_inputs_emp.append([
                                                    str(prompt_meta["sys_msg"]),
                                                    _build_option_user_prompt(str(prompt_meta["question"]), permuted_options, option_ids),
                                                ])
                                            empirical_sample = (
                                                int(prompt_meta["idx"]),
                                                (probing_inputs_emp, raw_options, str(prompt_meta["ideal"])),
                                            )
                                            empirical_result = eval_fn(empirical_sample, random.Random(0))
                                            empirical_api_calls = list(empirical_result["data"].get("api_calls") or [])
                                            extra_stage_probs = np.asarray(empirical_result["data"]["probs"], dtype=np.float64)
                                            all_stage_probs = np.vstack([base_row.reshape(1, -1), extra_stage_probs])
                                            cache_row = {
                                                "type": "empirical_stage_cache",
                                                "version": 1,
                                                "subject": str(subject),
                                                "run_idx": int(run_idx),
                                                "alpha": float(pride_alpha),
                                                "transition_mode": str(empirical_transition_mode),
                                                "sample_pos": int(sample_pos),
                                                "sample_id": int(prompt_meta["idx"]),
                                                "k": int(k),
                                                "stage_schedule": _schedule_signature(stage_schedule),
                                                "stage_probs": all_stage_probs.tolist(),
                                                "api_calls": empirical_api_calls,
                                            }
                                            _append_empirical_stage_cache(empirical_stage_cache_path, cache_row)
                                            empirical_stage_cache[int(sample_pos)] = cache_row
                                            empirical_stage_cache_misses += 1
                                        else:
                                            empirical_stage_cache_hits += 1
                                            empirical_api_calls = list(cached_stage_row.get("api_calls") or [])
                                        post_by_stage_full, pred_by_stage_full, conf_by_stage_full = _compute_empirical_stage_posteriors(
                                            stage_probs=all_stage_probs,
                                            slot_to_content_schedule=stage_schedule,
                                            mu_hat=empirical_mu_hat,
                                            residual_bank=empirical_residual_bank,
                                        )
                                        fallback_residual_bank = (
                                            np.zeros((1, k), dtype=np.float64)
                                            if empirical_skip_residual_on_cyclic
                                            else empirical_residual_bank
                                        )
                                        post_fallback_stage, pred_fallback_stage, conf_fallback_stage = _compute_empirical_stage_posteriors(
                                            stage_probs=all_stage_probs,
                                            slot_to_content_schedule=stage_schedule,
                                            mu_hat=empirical_mu_hat,
                                            residual_bank=fallback_residual_bank,
                                        )
                                        empirical_stage_infos.append({
                                            "sample_id": int(prompt_meta["idx"]),
                                            "pred_by_stage": [int(base_pred_stage[0]), int(pred_by_stage_full[1]), int(pred_fallback_stage[-1])],
                                            "conf_by_stage": [float(base_conf_stage[0]), float(conf_by_stage_full[1]), float(conf_fallback_stage[-1])],
                                            "true_prob_by_stage": [
                                                float(base_posterior[0][label_idx_emp]),
                                                float(post_by_stage_full[1][label_idx_emp]),
                                                float(post_fallback_stage[-1][label_idx_emp]),
                                            ],
                                            "decision_stages": [1, 2, int(k)],
                                            "prefix_forced": bool(sample_pos in empirical_prefix_ids),
                                        })
                                    elif empirical_transition_mode == "cyclic_learned":
                                        stage_probs = np.asarray(
                                            [per_sample_probs[sample_pos][cyclic_indices[int(shift)]] for shift in (stage_shifts or [0])],
                                            dtype=np.float64,
                                        )
                                        post_by_stage, pred_by_stage, conf_by_stage = _compute_empirical_stage_posteriors(
                                            stage_probs=stage_probs,
                                            slot_to_content_schedule=stage_schedule,
                                            mu_hat=empirical_mu_hat,
                                            residual_bank=empirical_residual_bank,
                                        )
                                        empirical_stage_infos.append({
                                            "sample_id": int(prompt_meta["idx"]),
                                            "pred_by_stage": [int(x) for x in pred_by_stage],
                                            "conf_by_stage": [float(x) for x in conf_by_stage],
                                            "true_prob_by_stage": [float(post[label_idx_emp]) for post in post_by_stage],
                                            "decision_stages": list(range(1, int(k) + 1)),
                                            "prefix_forced": bool(sample_pos in empirical_prefix_ids),
                                            "shift_order": [int(x) for x in (stage_shifts or [0])],
                                            "selected_sequence_name": None if learned_selection_info is None else str(learned_selection_info.get("selected_sequence_name", "")),
                                            "selected_action_sequence": [] if learned_selection_info is None else list(learned_selection_info.get("selected_action_sequence") or []),
                                        })
                                    else:
                                        empirical_stage_probs = _cached_empirical_stage_probs(
                                            cached_stage_row,
                                            sample_id=int(prompt_meta["idx"]),
                                            k=int(k),
                                            stage_schedule=stage_schedule,
                                        )
                                        if (
                                            api_backend and empirical_stage_probs is not None
                                            and (bool(args.api_force_requests) or not cached_stage_row.get("api_calls"))
                                        ):
                                            empirical_stage_probs = None
                                        if empirical_stage_probs is None:
                                            uses_cached_identity = (
                                                len(stage_schedule) > 0
                                                and tuple(int(x) for x in stage_schedule[0]) == tuple(range(k))
                                            )
                                            schedules_to_eval = stage_schedule[1:] if uses_cached_identity else stage_schedule
                                            probing_inputs_emp = []
                                            for slot_to_content in schedules_to_eval:
                                                permuted_options = [raw_options[int(content_idx)] for content_idx in slot_to_content]
                                                probing_inputs_emp.append([
                                                    str(prompt_meta["sys_msg"]),
                                                    _build_option_user_prompt(str(prompt_meta["question"]), permuted_options, option_ids),
                                                ])

                                            empirical_sample = (
                                                int(prompt_meta["idx"]),
                                                (probing_inputs_emp, raw_options, str(prompt_meta["ideal"])),
                                            )
                                            empirical_result = eval_fn(empirical_sample, random.Random(0))
                                            empirical_api_calls = list(empirical_result["data"].get("api_calls") or [])
                                            extra_stage_probs = np.asarray(empirical_result["data"]["probs"], dtype=np.float64)
                                            if uses_cached_identity:
                                                empirical_stage_probs = np.vstack([base_row.reshape(1, -1), extra_stage_probs])
                                            else:
                                                empirical_stage_probs = extra_stage_probs
                                            cache_row = {
                                                "type": "empirical_stage_cache",
                                                "version": 1,
                                                "subject": str(subject),
                                                "run_idx": int(run_idx),
                                                "alpha": float(pride_alpha),
                                                "transition_mode": str(empirical_transition_mode),
                                                "sample_pos": int(sample_pos),
                                                "sample_id": int(prompt_meta["idx"]),
                                                "k": int(k),
                                                "stage_schedule": _schedule_signature(stage_schedule),
                                                "stage_probs": empirical_stage_probs.tolist(),
                                                "api_calls": empirical_api_calls,
                                            }
                                            _append_empirical_stage_cache(empirical_stage_cache_path, cache_row)
                                            empirical_stage_cache[int(sample_pos)] = cache_row
                                            empirical_stage_cache_misses += 1
                                        else:
                                            empirical_stage_cache_hits += 1
                                            empirical_api_calls = list(cached_stage_row.get("api_calls") or [])
                                        if (sample_pos + 1) % 10 == 0 or (sample_pos + 1) == len(empirical_prompt_meta):
                                            logger.info(
                                                _blue(
                                                    f"Empirical stage progress: subject={subject}, alpha={float(pride_alpha):g}, "
                                                    f"{sample_pos + 1}/{len(empirical_prompt_meta)} "
                                                    f"(cache_hit={empirical_stage_cache_hits}, new={empirical_stage_cache_misses})"
                                                )
                                            )
                                        post_by_stage, pred_by_stage, conf_by_stage = _compute_empirical_stage_posteriors(
                                            stage_probs=empirical_stage_probs,
                                            slot_to_content_schedule=stage_schedule,
                                            mu_hat=empirical_mu_hat,
                                            residual_bank=empirical_residual_bank,
                                        )
                                        empirical_stage_infos.append({
                                            "sample_id": int(prompt_meta["idx"]),
                                            "pred_by_stage": [int(x) for x in pred_by_stage],
                                            "conf_by_stage": [float(x) for x in conf_by_stage],
                                            "true_prob_by_stage": [float(post[label_idx_emp]) for post in post_by_stage],
                                            "decision_stages": list(range(1, int(k) + 1)),
                                            "prefix_forced": bool(sample_pos in empirical_prefix_ids),
                                        })

                                    if api_backend and empirical_api_calls:
                                        offline_api_records_by_task.setdefault(
                                            str(args.task), {"n_samples": 0, "calls": []}
                                        )["calls"].extend(
                                            call for call in empirical_api_calls if isinstance(call, dict)
                                        )

                                if empirical_stage_cache_hits or empirical_stage_cache_misses:
                                    logger.info(
                                        _blue(
                                            f"Empirical stage cache summary: subject={subject}, alpha={float(pride_alpha):g}, "
                                            f"hit={empirical_stage_cache_hits}, new={empirical_stage_cache_misses}, "
                                            f"path={empirical_stage_cache_path}"
                                        )
                                    )

                                cobj_emp = {
                                    "subject": subject,
                                    "tag": "empirical_pride",
                                    "k": int(k),
                                    "percentile": float(pride_alpha),
                                    "sweep_mode": empirical_sweep_mode,
                                    "percentile_mode": empirical_percentile_mode,
                                    "residual_model": empirical_residual_model,
                                    "residual_weighting": _EMPIRICAL_RESIDUAL_WEIGHTING,
                                    "mc_samples": int(empirical_mc_samples if empirical_residual_model == "logistic_normal" else empirical_residual_bank.shape[0]),
                                    "cov_shrinkage": float(empirical_cov_shrinkage),
                                    "transition_mode": empirical_transition_mode,
                                    "skip_residual_on_cyclic": bool(empirical_skip_residual_on_cyclic),
                                    "threshold_schedule": empirical_stage_schedule,
                                    "threshold_gamma": empirical_stage_gamma,
                                    "selection_policy": None if learned_selection_info is None else str(learned_selection_info.get("selection_policy", "")),
                                    "selected_sequence_name": None if learned_selection_info is None else str(learned_selection_info.get("selected_sequence_name", "")),
                                    "selected_action_sequence": [] if learned_selection_info is None else list(learned_selection_info.get("selected_action_sequence") or []),
                                    "candidate_sequence_scores": [] if learned_selection_info is None else list(learned_selection_info.get("candidate_scores") or []),
                                    "selection_n_validation": 0 if learned_selection_info is None else int(learned_selection_info.get("n_validation", 0)),
                                    "n_samples": int(len(empirical_stage_infos)),
                                    "heuristic_points": [],
                                }
                                if empirical_sweep_mode == "confidence":
                                    for conf_th in empirical_conf_thresholds:
                                        conf_th_f = float(conf_th)
                                        c_emp, a_emp, preds_emp, counts_emp = _run_empirical_pride_policy_from_stage_infos_confidence(
                                            stage_infos=empirical_stage_infos,
                                            labels_idx=labels_idx_for_curves,
                                            k=k,
                                            confidence_threshold=conf_th_f,
                                            stage_schedule=empirical_stage_schedule,
                                            stage_gamma=empirical_stage_gamma,
                                        )
                                        hp_emp = {
                                            "label": EMPIRICAL_PRIDE_LABEL,
                                            "conf_th": conf_th_f,
                                            "cost": float(c_emp),
                                            "acc": float(a_emp),
                                            "marker": "X",
                                            "color": "gray",
                                            "recall_std": float(_recall_std(labels_idx_for_curves, preds_emp, k)),
                                        }
                                        hp_emp.update({key: int(val) for key, val in counts_emp.items()})
                                        cobj_emp["heuristic_points"].append(hp_emp)
                                else:
                                    for perc in ours_th1_list:
                                        perc_f = float(perc)
                                        c_emp, a_emp, preds_emp, counts_emp = _run_empirical_pride_policy_from_stage_infos(
                                            stage_infos=empirical_stage_infos,
                                            labels_idx=labels_idx_for_curves,
                                            k=k,
                                            percentile=perc_f,
                                            stage_schedule=empirical_stage_schedule,
                                            stage_gamma=empirical_stage_gamma,
                                            percentile_mode=empirical_percentile_mode,
                                        )
                                        hp_emp = {
                                            "label": EMPIRICAL_PRIDE_LABEL,
                                            "th1_p": perc_f,
                                            "cost": float(c_emp),
                                            "acc": float(a_emp),
                                            "marker": "X",
                                            "color": "gray",
                                            "recall_std": float(_recall_std(labels_idx_for_curves, preds_emp, k)),
                                        }
                                        hp_emp.update({key: int(val) for key, val in counts_emp.items()})
                                        cobj_emp["heuristic_points"].append(hp_emp)
                                try:
                                    empirical_sweep_values = empirical_conf_thresholds if empirical_sweep_mode == "confidence" else ours_th1_list
                                    analysis_summary, analysis_trajectories = _build_empirical_stage_analysis(
                                        stage_infos=empirical_stage_infos,
                                        labels_idx=labels_idx_for_curves,
                                        k=k,
                                        sweep_mode=empirical_sweep_mode,
                                        sweep_values=[float(x) for x in empirical_sweep_values],
                                        heuristic_points=cobj_emp["heuristic_points"],
                                    )
                                    analysis_record = {
                                        "task": str(args.task),
                                        "subject": str(subject),
                                        "run_idx": int(run_idx),
                                        "alpha": float(pride_alpha),
                                        "residual_model": empirical_residual_model,
                                        "residual_weighting": _EMPIRICAL_RESIDUAL_WEIGHTING,
                                        "mc_samples": int(empirical_mc_samples if empirical_residual_model == "logistic_normal" else empirical_residual_bank.shape[0]),
                                        "cov_shrinkage": float(empirical_cov_shrinkage),
                                        "transition_mode": empirical_transition_mode,
                                        "skip_residual_on_cyclic": bool(empirical_skip_residual_on_cyclic),
                                        "threshold_schedule": empirical_stage_schedule,
                                        "threshold_gamma": float(empirical_stage_gamma),
                                        "percentile_mode": empirical_percentile_mode,
                                        "selection_policy": None if learned_selection_info is None else str(learned_selection_info.get("selection_policy", "")),
                                        "selected_sequence_name": None if learned_selection_info is None else str(learned_selection_info.get("selected_sequence_name", "")),
                                        "selected_action_sequence": [] if learned_selection_info is None else list(learned_selection_info.get("selected_action_sequence") or []),
                                        "candidate_sequence_scores": [] if learned_selection_info is None else list(learned_selection_info.get("candidate_scores") or []),
                                        "selection_n_validation": 0 if learned_selection_info is None else int(learned_selection_info.get("n_validation", 0)),
                                        "summary": analysis_summary,
                                    }
                                    empirical_analysis_records.append(analysis_record)

                                    analysis_dir = os.path.join(
                                        build_results_dir(args, task=args.task, num_few_shot=args.num_few_shot, setting="full"),
                                        "empirical_analysis",
                                    )
                                    os.makedirs(analysis_dir, exist_ok=True)
                                    alpha_tag = f"{float(pride_alpha):g}"
                                    run_tag = f"_run{int(run_idx)}" if use_run_suffix else ""
                                    analysis_json_path = os.path.join(
                                        analysis_dir,
                                        f"{subject}{run_tag}_empirical_alpha{alpha_tag}_summary.json",
                                    )
                                    with open(analysis_json_path, "w", encoding="utf-8") as f:
                                        json.dump(analysis_record, f, ensure_ascii=False, indent=2)
                                    traj_jsonl_path = os.path.join(
                                        analysis_dir,
                                        f"{subject}{run_tag}_empirical_alpha{alpha_tag}_trajectories.jsonl",
                                    )
                                    with open(traj_jsonl_path, "w", encoding="utf-8") as f:
                                        for row in analysis_trajectories:
                                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                    logger.info(_purple(f"Saved empirical analysis: {analysis_json_path}"))
                                    logger.info(_purple(f"Saved empirical trajectories: {traj_jsonl_path}"))
                                except Exception as e:
                                    logger.warning(f"Failed to build/save empirical analysis for {subject} alpha={pride_alpha:g}: {e}")
                                by_empirical_alpha[pride_alpha].append(cobj_emp)

                        # Merge over runs and append to derived_records
                        for perc in ours_th1_list:
                            perc = float(perc)
                            cobjs_b = by_perc_baseline.get(perc, [])
                            merged_b = _merge_curve_objs_over_runs(cobjs_b) if cobjs_b else None
                            if merged_b:
                                derived_records_by_p.setdefault(perc, []).append(merged_b)
                                curve_objs_baseline.append(merged_b)

                        if pride_enabled:
                            for pride_alpha in pride_prefix_list:
                                cobjs_p = by_pride_alpha.get(pride_alpha, [])
                                merged_p = _merge_curve_objs_over_runs(cobjs_p) if cobjs_p else None
                                if merged_p:
                                    derived_records_pride_by_alpha.setdefault(pride_alpha, []).append(merged_p)
                                    curve_objs_pride.append(merged_p)

                        if empirical_enabled:
                            for pride_alpha in empirical_prefix_list:
                                cobjs_emp = by_empirical_alpha.get(pride_alpha, [])
                                merged_emp = _merge_curve_objs_over_runs(cobjs_emp) if cobjs_emp else None
                                if merged_emp:
                                    derived_records_empirical_by_alpha.setdefault(pride_alpha, []).append(merged_emp)
                                    curve_objs_empirical.append(merged_emp)

                        # ---------- save cyclic/base derived results ----------
                        cyclic_save_path = build_results_dir(args, task=args.task, num_few_shot=args.num_few_shot, setting="cyclic")
                        os.makedirs(cyclic_save_path, exist_ok=True)

                        cyclic_acc = (cyclic_corrects / cyclic_total) if cyclic_total > 0 else float('nan')
                        save_results(f'{cyclic_save_path}/{subject}.jsonl', cyclic_results,
                                 metrics={'type': 'metric', 'data': {'accuracy': cyclic_acc}})

                        base_save_path = build_results_dir(args, task=args.task, num_few_shot=args.num_few_shot, setting=None)
                        os.makedirs(base_save_path, exist_ok=True)

                        base_acc = float(np.mean(np.asarray(base_correct_list, dtype=np.float64))) if len(base_correct_list) else float('nan')
                        save_results(f'{base_save_path}/{subject}.jsonl', base_results,
                                 metrics={'type': 'metric', 'data': {'accuracy': base_acc}})

                        full_acc = (full_corrects / full_total) if full_total > 0 else float('nan')

                        # ---------- curve save path (for per-subject plots when not MMLU) ----------
                        curve_save_path = build_results_dir(args, task=args.task, num_few_shot=args.num_few_shot, setting="full")
                        os.makedirs(curve_save_path, exist_ok=True)

                        # (per-subject report removed — FINAL CONDENSED REPORT only)
                        save_results(f'{curve_save_path}/{subject}_curve.jsonl', curve_objs_baseline, metrics=None)
                        if pride_enabled and len(curve_objs_pride) > 0:
                            save_results(f'{curve_save_path}/{subject}_pride_curve.jsonl', curve_objs_pride, metrics=None)

                        # (th2 tradeoff plot removed — only macro three-curves acc/recall_std at end)

                    except Exception as e:
                        logger.warning(f"Failed to derive curves for subject '{subject}': {e}")
                        import traceback
                        traceback.print_exc()

            if not api_backend:
                logging_cuda_memory_usage()

        # 논문 작성용 T->F/F->T Empirical Analysis (이미지 업로드 전에 먼저 출력)
        def _print_transition_analysis(records, name):
            if not records:
                return
            all_base_t_gaps, all_base_f_gaps = [], []
            tot_t_to_f, tot_f_to_t = 0, 0
            t_to_f_ratios_per_subj, f_to_t_ratios_per_subj = [], []
            for rec in records:
                base_t = len(rec["base_t_gaps"])
                base_f = len(rec["base_f_gaps"])
                t_to_f = rec["t_to_f_count"]
                f_to_t = rec["f_to_t_count"]
                all_base_t_gaps.extend(rec["base_t_gaps"])
                all_base_f_gaps.extend(rec["base_f_gaps"])
                tot_t_to_f += t_to_f
                tot_f_to_t += f_to_t
                if base_t > 0:
                    t_to_f_ratios_per_subj.append(t_to_f / base_t * 100.0)
                if base_f > 0:
                    f_to_t_ratios_per_subj.append(f_to_t / base_f * 100.0)
            tot_base_t = len(all_base_t_gaps)
            tot_base_f = len(all_base_f_gaps)
            n_records = len(records)
            avg_gap_t = float(np.mean(all_base_t_gaps)) if tot_base_t > 0 else 0.0
            avg_gap_f = float(np.mean(all_base_f_gaps)) if tot_base_f > 0 else 0.0
            if n_records > 1 or len(t_to_f_ratios_per_subj) > 1 or len(f_to_t_ratios_per_subj) > 1:
                t_to_f_ratio = float(np.mean(t_to_f_ratios_per_subj)) if t_to_f_ratios_per_subj else 0.0
                f_to_t_ratio = float(np.mean(f_to_t_ratios_per_subj)) if f_to_t_ratios_per_subj else 0.0
                t_to_f_std = float(np.std(t_to_f_ratios_per_subj)) if len(t_to_f_ratios_per_subj) > 1 else 0.0
                f_to_t_std = float(np.std(f_to_t_ratios_per_subj)) if len(f_to_t_ratios_per_subj) > 1 else 0.0
                ratio_note = f" (macro avg over {n_records} records)"
                ratio_std = f" ± {t_to_f_std:.2f}" if t_to_f_std > 0 else ""
                ratio_std_ft = f" ± {f_to_t_std:.2f}" if f_to_t_std > 0 else ""
            else:
                t_to_f_ratio = (tot_t_to_f / tot_base_t * 100.0) if tot_base_t > 0 else 0.0
                f_to_t_ratio = (tot_f_to_t / tot_base_f * 100.0) if tot_base_f > 0 else 0.0
                ratio_note = ""
                ratio_std = ""
                ratio_std_ft = ""
            logger.info(_purple(f"\n==== EMPIRICAL ANALYSIS: {name} Permutation ===="))
            logger.info(f"[Initial Prediction: TRUE (원본 정답 그룹)]")
            logger.info(f" - Total Samples : {tot_base_t}")
            logger.info(f" - Avg Confidence: {avg_gap_t:.4f} (High Confidence)")
            logger.info(f" - Effect : T -> F (훼손) = {tot_t_to_f} / {tot_base_t} ({t_to_f_ratio:.2f}%{ratio_std}{ratio_note})")
            logger.info(f"\n[Initial Prediction: FALSE (원본 오답 그룹)]")
            logger.info(f" - Total Samples : {tot_base_f}")
            logger.info(f" - Avg Confidence: {avg_gap_f:.4f} (Low Confidence)")
            logger.info(f" - Effect : F -> T (교정) = {tot_f_to_t} / {tot_base_f} ({f_to_t_ratio:.2f}%{ratio_std_ft}{ratio_note})")
            logger.info("======================================================\n")

        _print_transition_analysis(transition_records_cyclic, "Cyclic")
        _print_transition_analysis(transition_records_full, "Full")

        # 논문 Experiments/Analysis: Default+PRIDE, Ours+PRIDE, Ours (per perc 2~100) — 정답/오답 avg conf, T→F, F→T
        def _print_transition_analysis_by_perc(records_by_p: Dict[float, List[dict]], method_name: str):
            if not records_by_p:
                return
            pride_fracs_sorted = sorted([p for p in records_by_p.keys() if isinstance(p, (int, float))])
            suffix = " (α=2% 고정)" if method_name == "Ours+PRIDE" else ""
            logger.info(_purple(f"\n==== EMPIRICAL ANALYSIS: {method_name} (per perc){suffix} [{args.task}] ===="))
            logger.info("perc | avg_conf_T(정답) | avg_conf_F(오답) | T→F(훼손) | F→T(교정) | T→F% | F→T%")
            for p in pride_fracs_sorted:
                recs = records_by_p.get(float(p), [])
                if not recs:
                    continue
                all_t, all_f, tot_t_to_f, tot_f_to_t = [], [], 0, 0
                for r in recs:
                    all_t.extend(r.get("base_t_gaps", []))
                    all_f.extend(r.get("base_f_gaps", []))
                    tot_t_to_f += r.get("t_to_f_count", 0)
                    tot_f_to_t += r.get("f_to_t_count", 0)
                nt, nf = len(all_t), len(all_f)
                avg_t = float(np.mean(all_t)) if nt > 0 else float("nan")
                avg_f = float(np.mean(all_f)) if nf > 0 else float("nan")
                t_to_f_pct = (tot_t_to_f / nt * 100.0) if nt > 0 else 0.0
                f_to_t_pct = (tot_f_to_t / nf * 100.0) if nf > 0 else 0.0
                p_str = f"{float(p):g}%"
                logger.info(f"{p_str:>6} | {avg_t:.4f} | {avg_f:.4f} | {tot_t_to_f} | {tot_f_to_t} | {t_to_f_pct:.2f}% | {f_to_t_pct:.2f}%")
            logger.info("======================================================\n")

        if transition_records_default_pride_by_p:
            _print_transition_analysis_by_perc(transition_records_default_pride_by_p, "Default+PRIDE")
        if transition_records_ours_pride_by_p:
            _print_transition_analysis_by_perc(transition_records_ours_pride_by_p, "Ours+PRIDE")
        if transition_records_ours_by_p:
            _print_transition_analysis_by_perc(transition_records_ours_by_p, "Ours")

        def _sigma_summary_slug(name: str) -> str:
            s = str(name).strip().lower()
            out = []
            for ch in s:
                if ch.isalnum():
                    out.append(ch)
                else:
                    out.append("_")
            slug = "".join(out)
            while "__" in slug:
                slug = slug.replace("__", "_")
            return slug.strip("_") or "sigma"

        def _print_sigma_analysis(records: List[dict], name: str):
            if not records:
                return None
            keys = [
                "sigma_mean",
                "sigma_std",
                "sigma_single",
                "sigma_two_view",
                "sigma_ratio",
                "corr_default_gap_sigma",
                "corr_flip_sigma",
                "sigma_low_conf_mean",
                "sigma_high_conf_mean",
                "flip_low_conf",
                "flip_high_conf",
                "flip_low_sigma",
                "flip_high_sigma",
            ]
            agg = {}
            for key in keys:
                vals = [float(r.get(key, float("nan"))) for r in records if np.isfinite(float(r.get(key, float("nan"))))]
                agg[key] = float(np.mean(vals)) if vals else float("nan")
            logger.info(_purple(f"\n==== SIGMA ANALYSIS: {name} ===="))
            logger.info(
                f"records={len(records)} | "
                f"sigma(mean)={agg['sigma_mean']:.4f}, sigma(std)={agg['sigma_std']:.4f}, "
                f"single_resid_sigma={agg['sigma_single']:.4f}, two_view_resid_sigma={agg['sigma_two_view']:.4f}, "
                f"ratio={agg['sigma_ratio']:.4f} (target={1.0 / math.sqrt(2.0):.4f})"
            )
            logger.info(
                f"corr(default_gap,sigma)={agg['corr_default_gap_sigma']:.4f}, "
                f"corr(flip,sigma)={agg['corr_flip_sigma']:.4f}"
            )
            logger.info(
                f"low_conf_sigma={agg['sigma_low_conf_mean']:.4f}, high_conf_sigma={agg['sigma_high_conf_mean']:.4f}, "
                f"flip_low_conf={agg['flip_low_conf']:.4f}, flip_high_conf={agg['flip_high_conf']:.4f}"
            )
            logger.info(
                f"flip_low_sigma={agg['flip_low_sigma']:.4f}, flip_high_sigma={agg['flip_high_sigma']:.4f}"
            )
            logger.info("========================================\n")
            agg["records"] = int(len(records))
            agg["sigma_ratio_target"] = float(1.0 / math.sqrt(2.0))
            return agg

        sigma_summary_payload = {}
        baseline_sigma_summary = _print_sigma_analysis(sigma_analysis_baseline_records, "Baseline")
        if baseline_sigma_summary is not None:
            sigma_summary_payload[_sigma_summary_slug("Baseline")] = baseline_sigma_summary
        for alpha in sorted(sigma_analysis_pride_by_alpha.keys()):
            name = f"PriDe(alpha={float(alpha):g}%)"
            pride_sigma_summary = _print_sigma_analysis(sigma_analysis_pride_by_alpha[alpha], name)
            if pride_sigma_summary is not None:
                sigma_summary_payload[_sigma_summary_slug(name)] = pride_sigma_summary
        if wandb_ok and wandb_run is not None and sigma_summary_payload:
            try:
                existing_sigma = wandb_run.summary.get("sigma_analysis_v1", {})
                if not isinstance(existing_sigma, dict):
                    existing_sigma = {}
                existing_sigma = dict(existing_sigma)
                task_key = f"{str(args.task)}_{int(args.num_few_shot)}shot"
                existing_sigma[task_key] = sigma_summary_payload
                wandb_run.summary["sigma_analysis_v1"] = existing_sigma
            except Exception as e:
                logger.warning(f"W&B sigma summary update failed: {e}")

        # Three-curves: Cost vs Acc, Cost vs Recall_std (Cyclic / Default+PRIDE / OURS th1/sqrt2)
        if len(derived_records_by_p) > 0:
            try:
                out_dir = build_results_dir(args, task=args.task, num_few_shot=args.num_few_shot, setting="full")
                os.makedirs(out_dir, exist_ok=True)
                cyclic_fracs = [int(x) for x in _parse_percent_value_list(getattr(args, "plot_cyclic_fractions", "0,10,20,30,40,50,60,70,80,90,100")) if 0 <= x <= 100]
                pride_fracs = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_ours_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0]
                pride_prefix = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_pride_prefix_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")) if 0.0 <= float(x) <= 100.0] or [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
                empirical_prefix = [float(x) for x in _parse_percent_value_list(getattr(args, "plot_empirical_prefix_fractions", None)) if 0.0 <= float(x) <= 100.0] or list(pride_prefix)
                if empirical_analysis_records:
                    seq_summary: Dict[str, int] = {}
                    for rec in empirical_analysis_records:
                        seq_name = str(rec.get("selected_sequence_name", "")).strip()
                        if not seq_name:
                            continue
                        seq_summary[seq_name] = int(seq_summary.get(seq_name, 0)) + 1
                    task_analysis_path = os.path.join(out_dir, f"{args.task}_empirical_stage_analysis.json")
                    task_analysis_payload = {
                        "task": str(args.task),
                        "model_name": str(args.model_name),
                        "eval_name": str(eval_name),
                        "n_runs": int(n_runs),
                        "selected_sequence_counts": seq_summary,
                        "records": empirical_analysis_records,
                    }
                    with open(task_analysis_path, "w", encoding="utf-8") as f:
                        json.dump(task_analysis_payload, f, ensure_ascii=False, indent=2)
                    logger.info(_purple(f"Saved task-level empirical analysis: {task_analysis_path}"))
                _plot_three_curves_acc_recall_std(
                    derived_records_by_p,
                    derived_records_pride_by_p if len(derived_records_pride_by_p) > 0 else {},
                    derived_records_pride_by_alpha if len(derived_records_pride_by_alpha) > 0 else {},
                    derived_records_empirical_by_alpha if len(derived_records_empirical_by_alpha) > 0 else {},
                    out_dir,
                    args.task,
                    cyclic_fractions=cyclic_fracs or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    pride_ours_fractions=pride_fracs or [2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                    pride_prefix_list=pride_prefix,
                    empirical_prefix_list=empirical_prefix,
                    wandb_ok=wandb_ok,
                    wandb_run=wandb_run,
                )
            except Exception as ex:
                logger.warning(f"Three-curves plot failed: {ex}")

        # =========================================================
        # 커스텀 최종 요약 리포트 (사용자 맞춤형 포맷)
        # =========================================================
        if len(derived_records_by_p) > 0:
            logger.info(_purple("==== FINAL CONDENSED REPORT ===="))
            n_subjects = len(subjects)

            def _macro_mean_std_over_runs(vals_list, n_subj, n_run):
                """57개 과목 macro 평균 → 5 run에 대해 mean ± std (MMLU 스타일)"""
                if n_subj <= 1 or n_run <= 1 or len(vals_list) != n_subj * n_run:
                    m = float(np.mean(vals_list)) if vals_list else float("nan")
                    s = float(np.std(vals_list)) if len(vals_list) > 1 else float("nan")
                    return m, s
                # cobjs 순서: s0r0,s0r1,...,s0r(n_run-1), s1r0,..., s(n_subj-1)r(n_run-1)
                run_means = []
                for r in range(n_run):
                    run_vals = [vals_list[r + i * n_run] for i in range(n_subj)]
                    run_vals = [x for x in run_vals if np.isfinite(x)]
                    run_means.append(float(np.mean(run_vals)) if run_vals else float("nan"))
                run_means = [x for x in run_means if np.isfinite(x)]
                mean = float(np.mean(run_means)) if run_means else float("nan")
                std = float(np.std(run_means)) if len(run_means) > 1 else float("nan")
                return mean, std

            def get_cyclic_stats(cobjs, p):
                pf = float(p)
                key_candidates = [f"cyclic_random_{pf}", f"cyclic_random_{pf:g}"]
                # pick first key that exists in at least one cobj
                key = None
                for kk in key_candidates:
                    if any((kk in c) for c in (cobjs or [])):
                        key = kk
                        break
                if key is None:
                    key = key_candidates[0]
                costs, accs, rstds = [], [], []
                for c in cobjs:
                    if key in c and "costs" in c[key] and "accuracies" in c[key]:
                        costs.append(c[key]["costs"][0])
                        accs.append(c[key]["accuracies"][0])
                    rk = f"{key}_recall_std"
                    if rk in c:
                        rstds.append(c[rk])
                if not accs:
                    return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
                mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                return mean_c, mean_a, mean_r, std_c, std_a, std_r

            def get_heur_stats(cobjs, label=PRIMARY_OURS_LABEL):
                costs, accs, rstds, nb, np2, nc = [], [], [], [], [], []
                for c in cobjs:
                    hps = {str(h.get("label")): h for h in (c.get("heuristic_points") or []) if isinstance(h, dict)}
                    if label in hps:
                        h = hps[label]
                        if "cost" in h:
                            costs.append(h["cost"])
                        accs.append(h.get("acc", float("nan")))
                        if "recall_std" in h:
                            rstds.append(h["recall_std"])
                        if "n_base" in h:
                            nb.append(h["n_base"])
                        if "n_probe2" in h:
                            np2.append(h["n_probe2"])
                        if "n_cyclic" in h:
                            nc.append(h["n_cyclic"])
                if not accs:
                    return float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan")
                mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                mean_nb = float(np.mean(nb)) if nb else 0.0
                mean_np2 = float(np.mean(np2)) if np2 else 0.0
                mean_nc = float(np.mean(nc)) if nc else 0.0
                return mean_c, mean_a, mean_r, mean_nb, mean_np2, mean_nc, std_c, std_a, std_r

            def get_heur_stats_by_sweep(cobjs, sweep_value, label_filter="online_sqrt_all", sweep_key="th1_p"):
                costs, accs, rstds, nb, np2, nc = [], [], [], [], [], []
                for c in cobjs:
                    for h in (c.get("heuristic_points") or []):
                        if isinstance(h, dict) and h.get(sweep_key) == sweep_value and h.get("label") == label_filter:
                            if "cost" in h:
                                costs.append(h["cost"])
                            accs.append(h.get("acc", float("nan")))
                            if "recall_std" in h:
                                rstds.append(h["recall_std"])
                            if "n_base" in h:
                                nb.append(h["n_base"])
                            if "n_probe2" in h:
                                np2.append(h["n_probe2"])
                            if "n_cyclic" in h:
                                nc.append(h["n_cyclic"])
                            break
                if not accs:
                    return float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, float("nan"), float("nan"), float("nan")
                mean_c, std_c = _macro_mean_std_over_runs(costs, n_subjects, n_runs)
                mean_a, std_a = _macro_mean_std_over_runs(accs, n_subjects, n_runs)
                mean_r, std_r = _macro_mean_std_over_runs(rstds, n_subjects, n_runs)
                mean_nb = float(np.mean(nb)) if nb else 0.0
                mean_np2 = float(np.mean(np2)) if np2 else 0.0
                mean_nc = float(np.mean(nc)) if nc else 0.0
                return mean_c, mean_a, mean_r, mean_nb, mean_np2, mean_nc, std_c, std_a, std_r

            def get_heur_stats_by_th1_p(cobjs, th1_p, label_filter="online_sqrt_all"):
                return get_heur_stats_by_sweep(cobjs, th1_p, label_filter=label_filter, sweep_key="th1_p")

            pride_fracs = [float(x) for x in _parse_percent_value_list(
                getattr(args, "plot_pride_ours_fractions", "0.5,1,2,5,10,20,30,40,50,60,70,80,90,100")
            ) if 0.0 <= float(x) <= 100.0] or [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            empirical_conf_fracs = [float(x) for x in _parse_float_value_list(
                getattr(args, "empirical_conf_thresholds", "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"),
                default=[0.5],
            ) if 0.0 <= float(x) <= 1.0] or [0.5]
            empirical_report_mode = str(getattr(args, "empirical_sweep_mode", "percentile")).strip().lower()
            if empirical_report_mode not in {"percentile", "confidence"}:
                empirical_report_mode = "percentile"
            empirical_report_schedule = str(getattr(args, "empirical_stage_schedule", "sqrt")).strip().lower()
            if empirical_report_schedule not in {"flat", "sqrt"}:
                empirical_report_schedule = "sqrt"
            # Cyclic 레포트: 항상 0,10,20,...,100 전체 구간 출력 (plot_cyclic_fractions와 무관)
            cyclic_fracs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            pride_alphas = sorted(derived_records_pride_by_alpha.keys()) if derived_records_pride_by_alpha else []
            empirical_alphas = sorted(derived_records_empirical_by_alpha.keys()) if derived_records_empirical_by_alpha else []

            _fmt = (lambda m, s: f"{m:.3f}±{s:.3f}" if np.isfinite(s) and s > 0 else f"{m:.3f}") if n_runs > 1 else (lambda m, s: f"{m:.3f}")
            _fmt4 = (lambda m, s: f"{m:.4f}±{s:.4f}" if np.isfinite(s) and s > 0 else f"{m:.4f}") if n_runs > 1 else (lambda m, s: f"{m:.4f}")

            # 1. default + pride (per alpha only — alpha와 cyclic fraction p 동일 개념)
            logger.info("---- default + pride ----")
            for alpha in pride_alphas:
                cobjs = derived_records_pride_by_alpha[alpha]
                p = alpha  # Default+PRIDE: prefix α% = cyclic fraction, 하나의 파라미터만 사용
                cost, acc, rstd, std_c, std_a, std_r = get_cyclic_stats(cobjs, p)
                a_str = f"{float(alpha):g}"
                logger.info(f"default_pride_α{a_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}")

            # 2. ours + pride (per alpha): legacy th1/2, primary variance sqrt2, and online sqrt
            logger.info("---- ours + pride (th1/2 legacy) ----")
            for alpha in pride_alphas:
                cobjs = derived_records_pride_by_alpha[alpha]
                for p in pride_fracs:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats_by_th1_p(cobjs, p, LEGACY_OURS_LABEL)
                    a_str = f"{float(alpha):g}"
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_pride_th12_α{a_str}_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")
            logger.info(f"---- ours + pride ({PRIMARY_OURS_LABEL}) ----")
            for alpha in pride_alphas:
                cobjs = derived_records_pride_by_alpha[alpha]
                for p in pride_fracs:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats_by_th1_p(cobjs, p, PRIMARY_OURS_LABEL)
                    a_str = f"{float(alpha):g}"
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_pride_var_α{a_str}_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")
            logger.info("---- ours + pride (Online Sqrt) ----")
            for alpha in pride_alphas:
                cobjs = derived_records_pride_by_alpha[alpha]
                for p in pride_fracs:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats_by_th1_p(cobjs, p, "online_sqrt_all")
                    a_str = f"{float(alpha):g}"
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_pride_sqrt_α{a_str}_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")

            if empirical_alphas:
                empirical_report_percentile_mode = str(getattr(args, "empirical_percentile_mode", "online")).strip().lower()
                if empirical_report_mode == "confidence":
                    logger.info(f"---- empirical pride (confidence sweep, {empirical_report_schedule}, {empirical_report_percentile_mode}) ----")
                    for alpha in empirical_alphas:
                        cobjs = derived_records_empirical_by_alpha[alpha]
                        transition_mode = str(cobjs[0].get("transition_mode", "latin")).strip().lower() if cobjs else "latin"
                        for conf_th in empirical_conf_fracs:
                            cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats_by_sweep(
                                cobjs, float(conf_th), label_filter=EMPIRICAL_PRIDE_LABEL, sweep_key="conf_th"
                            )
                            a_str = f"{float(alpha):g}"
                            conf_str = f"{float(conf_th):.2f}"
                            logger.info(f"empirical_pride_conf_{empirical_report_schedule}_{transition_mode}_α{a_str}_{conf_str} : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")
                else:
                    logger.info(f"---- empirical pride (percentile sweep, {empirical_report_schedule}, {empirical_report_percentile_mode}) ----")
                    for alpha in empirical_alphas:
                        cobjs = derived_records_empirical_by_alpha[alpha]
                        transition_mode = str(cobjs[0].get("transition_mode", "latin")).strip().lower() if cobjs else "latin"
                        percentile_mode = str(cobjs[0].get("percentile_mode", empirical_report_percentile_mode)).strip().lower() if cobjs else empirical_report_percentile_mode
                        for p in pride_fracs:
                            cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats_by_th1_p(
                                cobjs, float(p), EMPIRICAL_PRIDE_LABEL
                            )
                            a_str = f"{float(alpha):g}"
                            p_str = f"{float(p):g}"
                            logger.info(f"empirical_pride_pct_{empirical_report_schedule}_{percentile_mode}_{transition_mode}_α{a_str}_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")

            # 3. ours
            logger.info("---- ours ----")
            for p in pride_fracs:
                if float(p) in derived_records_by_p:
                    cost, acc, rstd, nb, np2, nc, std_c, std_a, std_r = get_heur_stats(derived_records_by_p[float(p)], PRIMARY_OURS_LABEL)
                    p_str = f"{float(p):g}"
                    logger.info(f"ours_{p_str}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}, n_base={nb:.0f}, n_probe={np2:.0f}, n_cyclic={nc:.0f}")

            # 4. cyclic
            logger.info("---- cyclic ----")
            base_any_cobjs = next(iter(derived_records_by_p.values()), []) if derived_records_by_p else []
            for p in cyclic_fracs:
                cost, acc, rstd, std_c, std_a, std_r = get_cyclic_stats(base_any_cobjs, p)
                logger.info(f"cyclic_{p:03d}% : cost={_fmt(cost, std_c)}, acc={_fmt4(acc, std_a)}, recall_std={_fmt4(rstd, std_r)}")

    # -------- API usage/cost summary --------
    if api_backend:
        try:
            api_summary = model.summary()
            task_payloads = {}
            summary_paths = []
            wandb_points = {}
            if wandb_ok and wandb_run is not None:
                maybe_points = wandb_run.summary.get("three_curves_points_v1", {})
                if isinstance(maybe_points, dict):
                    wandb_points = maybe_points
            for eval_name in args.eval_names:
                eval_parts = str(eval_name).split(",")
                task_name, shots = eval_parts[0], int(eval_parts[1])
                records = offline_api_records_by_task.get(task_name, {"n_samples": 0, "calls": []})
                logical = _summarize_api_call_records(records.get("calls") or [])
                logical_requests = int(logical.get("requests", 0) or 0)
                avg_call_cost = (
                    float(logical.get("cost_usd", 0.0) or 0.0) / logical_requests
                    if logical_requests > 0 else float("nan")
                )
                points_payload = wandb_points.get(task_name)
                out_dir = build_results_dir(args, task_name, shots, "full")
                points_path = os.path.join(out_dir, f"{task_name}_three_curves_points.json")
                if not isinstance(points_payload, dict) and os.path.exists(points_path):
                    with open(points_path, "r", encoding="utf-8") as f:
                        points_payload = json.load(f)
                payload = {
                    **dict(api_summary),
                    "task": task_name,
                    "execution_mode": str(args.api_execution_mode),
                    "prompt_mode": str(args.api_prompt_mode),
                    **_api_scoring_metadata(args),
                    "adaptive_percentile": getattr(args, "api_adaptive_percentile", None),
                    "n_samples": int(records.get("n_samples", 0) or 0),
                    "logical": logical,
                    "returned_model": logical.get("returned_model") or api_summary.get("returned_model"),
                    "returned_models": logical.get("returned_models") or api_summary.get("returned_models", {}),
                    "average_logical_call_cost_usd": avg_call_cost,
                    "physical_scope": "current process (all requested tasks)",
                    "counterfactual_note": (
                        "offline_sweep policy USD uses measured mean logical call cost multiplied by E[T]; "
                        "physical is actual network spend in this process and excludes durable-cache hits."
                    ),
                    "counterfactual_policies": _collect_counterfactual_api_costs(
                        points_payload, avg_call_cost, int(records.get("n_samples", 0) or 0)
                    ) if np.isfinite(avg_call_cost) else {},
                }
                task_payloads[task_name] = payload
                os.makedirs(out_dir, exist_ok=True)
                summary_path = os.path.join(out_dir, f"{task_name}_api_evaluation_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                summary_paths.append(summary_path)
                logger.info(_purple(f"Saved API evaluation summary: {summary_path}"))
            if wandb_ok and wandb_run is not None:
                wandb_run.summary["api_evaluation_v1"] = task_payloads
                cache_dirs = api_summary.get("cache_dirs") or [str(model.cache_dir)]
                has_call_logs = any(os.path.exists(os.path.join(str(path), "calls.jsonl")) for path in cache_dirs)
                if summary_paths or has_call_logs:
                    try:
                        import wandb
                        art = wandb.Artifact(
                            name=f"api-calls-{args.model_name}-{wandb_run.id}",
                            type="api_calls",
                        )
                        for summary_path in summary_paths:
                            art.add_file(summary_path)
                        for cache_idx, cache_dir in enumerate(cache_dirs):
                            calls_path = os.path.join(str(cache_dir), "calls.jsonl")
                            diagnostics_path = os.path.join(str(cache_dir), "diagnostics.jsonl")
                            if os.path.exists(calls_path):
                                art.add_file(calls_path, name=f"cache_{cache_idx}/calls.jsonl")
                            if os.path.exists(diagnostics_path):
                                art.add_file(diagnostics_path, name=f"cache_{cache_idx}/diagnostics.jsonl")
                        wandb_run.log_artifact(art)
                    except Exception as e:
                        logger.warning(f"W&B API artifact logging failed: {e}")
        except Exception as e:
            logger.warning(f"API usage summary failed: {e}")

    # -------- finalize W&B --------
    _wandb_done = {"done": False}
    def _wandb_finish():
        if _wandb_done["done"] or not wandb_ok or wandb_run is None:
            return
        try:
            import wandb
            logger.info(_blue("W&B: syncing and finishing run..."))
            wandb.finish()
            time.sleep(5)  # 업로드 스레드 완료 대기 (업로드 중 프로세스 죽는 문제 완화)
            logger.info(_blue("W&B: run finished."))
        except Exception as e:
            logger.warning(f"W&B finish failed: {e}")
        finally:
            _wandb_done["done"] = True

    if wandb_ok and wandb_run is not None:
        atexit.register(_wandb_finish)
    try:
        _wandb_finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
