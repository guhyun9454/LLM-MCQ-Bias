
import os
import sys
import random
import copy
import json
import argparse
import logging
import re
from tqdm import tqdm
from typing import List
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import (
    _norm,
    _purple,
    shuffle_options_with_ids,
    move_answer,
    cycle_options,
    permute_options,
)

logger = logging.getLogger(__name__)

# Pairwise preference-judging benchmarks: a query and two candidate answers, so
# k=2. Their data is built by code/data_<task>/process.py on top of
# code/pairwise_data_utils.py.
PAIRWISE_TASKS = ('rewardbench', 'mtbench', 'prefbench')


def parse_arguments():
    # ---- Safe CUDA logging (GPU 없을 때도 안 터지게) ----
    try:
        cuda_ok = torch.cuda.is_available()
        n_dev = torch.cuda.device_count() if cuda_ok else 0
        dev_name = torch.cuda.get_device_name(0) if (cuda_ok and n_dev > 0) else "N/A"
        logger.info(f'cuda is available {cuda_ok}')
        logger.info(f'cuda device count {n_dev}')
        logger.info(f'cuda device name {dev_name}')
    except Exception as e:
        logger.warning(f'CUDA info logging failed: {e}')

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--inference_backend", type=str, default="local",
                        choices=["local", "api"],
                        help="Inference backend. local keeps the existing Hugging Face path; api uses commercial chat logprobs.")
    parser.add_argument("--api_provider", type=str, default=None,
                        choices=["openai", "gemini", "vertex", "deepseek", "together"],
                        help="Commercial provider. Required when --inference_backend=api.")
    parser.add_argument("--api_execution_mode", type=str, default="offline_sweep",
                        choices=["offline_sweep", "adaptive"],
                        help="offline_sweep precomputes stages for curves; adaptive physically stops API calls at the selected policy stage.")
    parser.add_argument("--api_prompt_mode", type=str, default="baseline",
                        choices=["baseline", "label_only"],
                        help="API-only prompt policy. baseline preserves the original PriDe prompt; label_only adds a strict one-label instruction.")
    parser.add_argument("--api_scoring_mode", type=str, default="topk_strict",
                        choices=["topk_strict", "equal_label_bias"],
                        help="API label scoring policy. equal_label_bias adds the same positive logit bias to each exact canonical label token and is reported as a separate constrained protocol.")
    parser.add_argument("--api_equal_label_bias", type=float, default=100.0,
                        help="Common logit bias for every canonical label token in equal_label_bias mode (0, 100].")
    parser.add_argument("--api_adaptive_percentile", type=float, default=None,
                        help="Required percentile operating point for --api_execution_mode=adaptive (for example 80).")
    parser.add_argument("--api_probe_only", action="store_true",
                        help="Run only the first --api_probe_samples examples and save capability/usage diagnostics.")
    parser.add_argument("--api_probe_samples", type=int, default=10,
                        help="Number of examples used by --api_probe_only (default: 10).")
    parser.add_argument("--api_probe_all_subjects", action="store_true",
                        help="With --api_probe_only, probe the requested number of examples in every subject instead of only the first subject.")
    parser.add_argument("--api_max_cost_usd", type=float, default=None,
                        help="Required API safety guard unless --api_max_requests is supplied.")
    parser.add_argument("--api_max_requests", type=int, default=None,
                        help="Required API safety guard unless --api_max_cost_usd is supplied.")
    parser.add_argument("--api_force_requests", action="store_true",
                        help="Bypass durable API response cache. --force alone never reissues paid calls.")
    parser.add_argument("--api_timeout_seconds", type=float, default=60.0)
    parser.add_argument("--api_max_retries", type=int, default=6)
    parser.add_argument("--api_concurrency", type=int, default=4,
                        help="Maximum sample concurrency for offline_sweep; adaptive is always sequential.")
    parser.add_argument("--api_cache_dir", type=str, default=None,
                        help="Optional durable cache root. Defaults inside the result directory.")
    parser.add_argument("--api_base_url", type=str, default=None,
                        help="Optional provider base URL override.")
    parser.add_argument("--api_input_usd_per_mtok", type=float, default=None)
    parser.add_argument("--api_cached_input_usd_per_mtok", type=float, default=None)
    parser.add_argument("--api_output_usd_per_mtok", type=float, default=None)
    parser.add_argument("--eval_names", type=str, nargs='+', default=[],
                        help='eval tasks and settings')
    parser.add_argument("--option_id_set", type=str, default=None,
                        help='custom option ID string (e.g., ABCD / abcd / 1234). Length must match #options')
    parser.add_argument("--print_prompt_example", action="store_true",
                        help='Print exactly one example prompt (after applying option ids) and continue evaluation')
    parser.add_argument("--cache_dir", type=str, default="models",
                        help='Hugging Face cache directory for model/tokenizer downloads')
    parser.add_argument("--option_id_sets", type=str, nargs='+', default=None,
                        help='Provide exactly two option ID sets to compare (e.g., "ABCD abcd")')
    parser.add_argument("--test", action="store_true",
                        help='Test mode: evaluate only 100 samples instead of all samples')

    # W&B logging flags
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_entity", type=str, default="capde",
                        help='W&B entity (default: "capde")')
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--wandb_sample_idx", type=int, default=None,
                        help="Sample idx to log detailed prompts/probs; default: first sample")

    # Ours method hyperparameters (existing)
    parser.add_argument("--ours_low_conf_percent", type=float, default=30.0,
                        help="Bottom percentile (e.g., 30.0) of confidence (from beta subset) to trigger cascading ensemble")
    parser.add_argument("--ours_low_conf_percent_list", type=str, default="5,10,20,30",
                        help="Comma-separated percentile list for derived-policy reports/plots (default: '5,10,20,30'). Overrides --ours_low_conf_percent. Use empty string \"\" to disable and fall back to --ours_low_conf_percent.")
    parser.add_argument("--ours_th1_tradeoff", type=str, default="5,10,20,30",
                        help="th1 percentile values for trade-off plot (e.g., 5,10,20,30). 각 th1에 대해 th2 curve를 그림.")
    parser.add_argument("--ours_th2_tradeoff", type=str,
                        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
                        help="th2 percentile values for trade-off plot. Default: 1..30 (각 th1에 대해 th2를 1~30까지 변화시킴).")
    parser.add_argument("--ours_top2_gap_frac", type=float, default=0.3,
                        help="If (top1 - top2) / top1 <= this, treat as 'very ambiguous' and send directly to cyclic (use 0.0 to disable).")

    # =========================================================
    # [ADD] Simple PRIDE (PriDe) mixing knobs (online-friendly)
    # =========================================================
    parser.add_argument("--pride_mix", action="store_true",
                        help="Enable PRIDE debiasing then run OUR online policies on debiased probs (for comparison).")
    parser.add_argument("--pride_prefix_ratio", type=float, default=0.02,
                        help="Random prefix ratio used to estimate PRIDE prior (default=0.02).")
    parser.add_argument("--skip_full", action="store_true",
                        help="Use cyclic instead of full permutations when setting=full (e.g., MMLU 4-choice: 4x instead of 24x).")
    parser.add_argument("--pride_seed", type=int, default=0,
                        help="Seed for PRIDE random prefix sampling (default=0).")
    parser.add_argument("--empirical_pride", action="store_true",
                        help="Enable empirical-residual PriDe with adaptive Latin-square permutations.")
    parser.add_argument("--empirical_logit_delta", type=float, default=1e-12,
                        help="Stabilization delta used when converting PriDe priors to centered logits.")
    parser.add_argument("--empirical_residual_model", type=str, default="logistic_normal",
                        choices=["logistic_normal", "empirical", "identify"],
                        help="Residual prior model for empirical PriDe. logistic_normal uses Gaussian residual Monte Carlo; empirical reuses the residual bank directly (legacy); identify estimates the current question's own residual from its observed views once >=3 views are available (mu-only before that).")
    parser.add_argument("--empirical_ident_shrink", type=float, default=1.0,
                        help="Shrinkage factor for the identified per-question residual (identify mode only): the posterior correction uses mu + s*eps_hat(q). 1.0 = raw LS estimate.")
    parser.add_argument("--empirical_mc_samples", type=int, default=64,
                        help="Number of Monte Carlo residual samples for logistic-normal empirical PriDe.")
    parser.add_argument("--empirical_cov_shrinkage", type=float, default=0.1,
                        help="Shrinkage lambda for logistic-normal residual covariance estimation.")
    parser.add_argument("--empirical_sweep_mode", type=str, default="percentile",
                        choices=["percentile", "confidence"],
                        help="Sweep mode for empirical PriDe curves: percentile uses beta-percentile thresholds, confidence uses fixed confidence thresholds.")
    parser.add_argument("--empirical_percentile_mode", type=str, default="online",
                        choices=["online", "fixed_prefix"],
                        help="Percentile-threshold policy for empirical PriDe. online uses running stage-wise quantiles; fixed_prefix freezes stage-wise quantile thresholds from the empirical alpha-prefix samples.")
    parser.add_argument("--empirical_conf_thresholds", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
                        help="Comma-separated confidence thresholds for empirical PriDe when --empirical_sweep_mode=confidence.")
    parser.add_argument("--empirical_stage_schedule", type=str, default="sqrt",
                        choices=["flat", "sqrt"],
                        help="Stage-wise threshold schedule for empirical PriDe. flat keeps one threshold across stages, sqrt relaxes later stages.")
    parser.add_argument("--empirical_stage_gamma", type=float, default=0.5,
                        help="Gamma for empirical PriDe stage schedule. sqrt schedule uses threshold scaling by t^(-gamma).")
    parser.add_argument("--empirical_transition_mode", type=str, default="latin",
                        choices=["latin", "probe_cyclic", "cyclic_random", "cyclic_targeted", "cyclic_learned"],
                        help="Empirical PriDe transition path. latin keeps stage-by-stage Latin expansion; cyclic_random incrementally adds random cyclic permutations; cyclic_targeted starts with the cyclic shift that moves the runner-up into the top-1 slot, then adds the remaining cyclic shifts; cyclic_learned chooses a top-3 relative cyclic action sequence from the alpha-prefix validation subset by weighted NLL gain; probe_cyclic is the legacy base -> probe -> final fallback path.")
    parser.add_argument("--empirical_skip_residual_on_cyclic", action="store_true",
                        help="When empirical_transition_mode=probe_cyclic, skip residual-bank averaging on the final fallback stage and use only the global prior correction.")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs for derived policies (PRIDE prior, cyclic_random). Results averaged over runs, like debiase_pride.py (default=1).")

    # Three-curves plot: fraction lists (modify to change number of points)
    parser.add_argument("--plot_cyclic_fractions", type=str, default="0,10,20,30,40,50,60,70,80,90,100",
                        help="Cyclic (no PRIDE) curve fractions, comma-separated (e.g. 0,10,20,...,100).")
    parser.add_argument("--plot_pride_ours_fractions", type=str, default="0.5,1,2,5,10,20,30,40,50,60,70,80,90,100",
                        help="OURS th1 curve (x-axis). Default+PRIDE도 동일 p 사용.")
    parser.add_argument("--plot_pride_prefix_fractions", type=str, default="0.5,1,2,5,10,20,30,40,50,60,70,80,90,100",
                        help="Ours+PRIDE에서 PriDe prefix(alpha) 값. 선택 가능한 α 목록.")
    parser.add_argument("--plot_empirical_prefix_fractions", type=str, default=None,
                        help="Empirical PriDe용 prefix(alpha) 값. 비우면 --plot_pride_prefix_fractions를 그대로 사용.")

    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose logs (extra summaries).")

    # =========================================================
    # [ADD] AvgGap(PSEUDO, ONLINE) knobs (eval_clm.py에서 getattr로 쓰는 것들)
    # =========================================================
    # debug: allow usage like --ours_debug_mad 1
    parser.add_argument("--ours_debug_mad", type=int, default=0,
                        help="(0/1) Print MAD/threshold debug logs.")
    parser.add_argument("--ours_debug_mad_n", type=int, default=5,
                        help="Max debug prints per beta.")
    
    parser.add_argument("--ours_th1_ema", type=float, default=0.01, help="EMA rate used in entropy probe policy: (1) th2 teacher update step size, (2) th1 tracking to th2.")

    # MAD EMA alpha
    parser.add_argument("--ours_mad_alpha", type=float, default=0.10,
                        help="EMA alpha for MAD update.")

    # th1 online update lr
    parser.add_argument("--ours_th_lr_up", type=float, default=0.05,
                        help="th1 increase lr (when tc==t1).")
    parser.add_argument("--ours_th_lr_dn", type=float, default=0.05,
                        help="th1 decrease lr (when tc!=t1).")

    # th2 = th1 (+/-) MAD
    parser.add_argument("--ours_th2_mode", type=str, default="plus",
                        choices=["plus", "minus", "sub", "-"],
                        help="th2 = th1 + MAD (plus) or th1 - MAD (minus).")

    # th1 init / clamp range (all in [0,1])
    # NOTE: eval_clm.py에서 None이면 perc로 자동 설정하도록 처리하는 걸 권장
    parser.add_argument("--ours_th1_init", type=float, default=None,
                        help="Initial th1 in [0,1]. If omitted, auto=ours_low_conf_percent/100.")
    parser.add_argument("--ours_th1_min", type=float, default=0.0,
                        help="Min clamp for th1.")
    parser.add_argument("--ours_th1_max", type=float, default=1.0,
                        help="Max clamp for th1.")

    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results files if they exist")
    parser.add_argument("--result_tag", type=str, default=None,
                        help="Optional suffix appended to local results directories to separate experiment variants. Legacy path is unchanged when omitted.")
    parser.add_argument("--develop", action="store_true",
                        help="Dev mode: skip model/data eval and generate dummy numeric logs/plots (for W&B/Streamlit pipeline test).")

    args = parser.parse_args()

    if args.inference_backend == "api":
        if not args.api_provider:
            parser.error("--api_provider is required when --inference_backend=api")
        if args.api_max_cost_usd is None and args.api_max_requests is None:
            parser.error("API runs require --api_max_cost_usd or --api_max_requests")
        if args.api_max_cost_usd is not None and args.api_max_cost_usd <= 0:
            parser.error("--api_max_cost_usd must be positive")
        if args.api_max_requests is not None and args.api_max_requests <= 0:
            parser.error("--api_max_requests must be positive")
        if args.api_execution_mode == "adaptive":
            if args.api_adaptive_percentile is None:
                parser.error("adaptive API runs require --api_adaptive_percentile")
            if not 0.0 <= float(args.api_adaptive_percentile) <= 100.0:
                parser.error("--api_adaptive_percentile must be in [0, 100]")
            if not args.empirical_pride:
                parser.error("adaptive API runs require --empirical_pride")
            if args.empirical_sweep_mode != "percentile":
                parser.error("adaptive API v1 requires --empirical_sweep_mode percentile")
            if args.empirical_transition_mode != "latin":
                parser.error("adaptive API v1 requires --empirical_transition_mode latin")
        if args.api_scoring_mode == "equal_label_bias":
            if args.api_provider != "openai":
                parser.error("--api_scoring_mode equal_label_bias currently supports only --api_provider openai")
            if args.api_prompt_mode != "label_only":
                parser.error("--api_scoring_mode equal_label_bias requires --api_prompt_mode label_only")
            if not 0.0 < float(args.api_equal_label_bias) <= 100.0:
                parser.error("--api_equal_label_bias must be in (0, 100]")
        if int(args.api_probe_samples) <= 0:
            parser.error("--api_probe_samples must be positive")
        if int(args.api_concurrency) <= 0:
            parser.error("--api_concurrency must be positive")
        if float(args.api_timeout_seconds) <= 0:
            parser.error("--api_timeout_seconds must be positive")
        if int(args.api_max_retries) < 0:
            parser.error("--api_max_retries must be non-negative")
        for price_arg in (
            "api_input_usd_per_mtok",
            "api_cached_input_usd_per_mtok",
            "api_output_usd_per_mtok",
        ):
            price = getattr(args, price_arg)
            if price is not None and float(price) < 0:
                parser.error(f"--{price_arg} must be non-negative")

    args.model_name = args.pretrained_model_path.split('/')[-1]

    for eval_name in args.eval_names:
        eval_args = eval_name.split(',')
        task = eval_args[0]
        if task not in ['mmlu', 'arc', 'csqa', 'race', 'racem', 'raceall'] + list(PAIRWISE_TASKS):
            raise ValueError(f"Unknown task: {task}")

        num_few_shot = int(eval_args[1])

        setting = eval_args[2] if len(eval_args) > 2 else None
        if setting is not None and not (
            setting in [
                'noid',
                'perm', 'cyclic', 'full',
                'shuffle_both',
            ] or (setting.startswith('move') and setting[-1] in ['a', 'b', 'c', 'd'])
        ):
            raise ValueError(f"Unknown setting: {setting}")
        if args.inference_backend == "api" and setting == "noid":
            raise ValueError("Commercial API v1 does not support noid sequence scoring")

    return args


def _sanitize_result_tag(tag: str) -> str:
    s = str(tag or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s.strip("-._")


def build_results_dir(args, task: str, num_few_shot: int, setting: str = None) -> str:
    save_path = f"results_{task}/{num_few_shot}s_{args.model_name}/{task}"
    if setting is not None:
        save_path += f"_{setting}"
    if getattr(args, "option_id_set", None):
        save_path += f"_id-{args.option_id_set}"
    tag = _sanitize_result_tag(getattr(args, "result_tag", None))
    if tag:
        save_path += f"__{tag}"
    return save_path


def select_api_probe_subjects(args, subjects):
    """Keep the legacy first-subject probe unless an explicit stratified probe is requested."""
    selected = list(subjects)
    if not bool(getattr(args, "api_probe_only", False)):
        return selected
    if bool(getattr(args, "api_probe_all_subjects", False)):
        return selected
    return selected[:1]


def prepare_eval(args, eval_name):
    # task and setting
    eval_args = eval_name.split(',')
    args.task = task = eval_args[0]
    args.num_few_shot = num_few_shot = int(eval_args[1])
    args.setting = setting = eval_args[2] if len(eval_args) > 2 and eval_args[2] else None
    if setting is not None and setting.startswith('move'):
        moved_answer = setting[-1].upper()

    # save_path
    save_path = build_results_dir(args, task=task, num_few_shot=num_few_shot, setting=setting)
    args.save_path = save_path
    os.makedirs(args.save_path, exist_ok=True)

    option_ids = list('ABCD')
    option_ids_header = list('ABCD')
    if task in ['csqa']:
        option_ids = list('ABCDE')
        option_ids_header = list('ABCDE')
    elif task in PAIRWISE_TASKS:
        # Pairwise preference judging: two candidate answers, so k=2. The cyclic
        # rotation and the targeted Latin schedule both degrade correctly here
        # (2 views, identity + swap).
        option_ids = list('AB')
        option_ids_header = list('AB')

    # Override displayed option IDs by user-specified token set (for probing token preference)
    # Keep headers (ground-truth labels) as uppercase letters to match dataset files.
    if getattr(args, 'option_id_set', None):
        k = len(option_ids_header)
        custom = list(args.option_id_set)
        if len(custom) != k:
            raise ValueError(f"option_id_set length must be {k} for task '{task}', got {len(custom)}: {args.option_id_set}")
        option_ids = custom

    data_path = f'data_{task}'
    subjects = sorted([f.split("_test.csv")[0]
                       for f in os.listdir(f'{data_path}/test') if "_test.csv" in f])

    # sys_msg
    if 'mmlu' in task:
        sys_msg = 'The following are multiple choice questions about {}.'
    elif task in ('race', 'racem', 'raceall'):
        sys_msg = 'The following are multiple choice reading comprehension questions about an article.'
    elif task in PAIRWISE_TASKS:
        sys_msg = 'The following are queries with two candidate answers, of which one answers the query better.'
    else: # task in ['arc', 'tqa']
        sys_msg = 'The following are multiple choice questions.'

    sys_msg += ' You should directly answer the question by choosing the correct option.'

    # RACE folds the passage into the Question column and carries its own
    # "Article:/Question:" headings (see data_race/process.py), so the generic
    # prefix would render as "Question: Article: ...".
    question_prefix = '' if task in ('race', 'racem', 'raceall') else 'Question: '

    # create_user_prompt
    def create_user_prompt(question: str, options: List[str]):
        if setting in ['noid']:
            user_prompt = f"{question_prefix}{question.strip()}\nOptions:\n" + \
                "\n".join([f"{answer}".strip()
                           for option_id, answer in zip(option_ids, options)]) + \
                "\nAnswer:"
        elif setting in ['shuffle_both']:
            shuffled_option_ids, shuffled_options = shuffle_options_with_ids(option_ids, options)
            user_prompt = f"{question_prefix}{question.strip()}\nOptions:\n" + \
                "\n".join([f"{option_id}. {answer}".strip()
                           for option_id, answer in zip(shuffled_option_ids, shuffled_options)]) + \
                "\nAnswer:"
        else:
            user_prompt = f"{question_prefix}{question.strip()}\nOptions:\n" + \
                "\n".join([f"{option_id}. {answer}".strip()
                           for option_id, answer in zip(option_ids, options)]) + \
                "\nAnswer:"
        return user_prompt

    # prepare_few_shot_samples
    # few_shot_seed: None이면 기존 동작(첫 N개). 정수면 해당 seed로 shuffle 후 사용 (n_runs 시 run별 다른 few-shot)
    def prepare_few_shot_samples(subject, few_shot_seed=None):
        df = pd.read_csv(f'{data_path}/dev/{subject}_dev.csv', names=("Question", *option_ids_header, "Answer"), dtype=str)
        if setting in ['noid']:
            few_shot_samples = df.apply(lambda x:
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header])
                + ' ' + x[x["Answer"]]
            , axis=1).to_list()
        else:
            few_shot_samples = df.apply(lambda x:
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header])
                + ' ' + option_ids[option_ids_header.index(x["Answer"])]
            , axis=1).to_list()
        if few_shot_seed is not None:
            rng = random.Random(int(few_shot_seed))
            rng.shuffle(few_shot_samples)
        return few_shot_samples

    if getattr(args, 'skip_full', False) and setting == 'full':
        k_opts = len(option_ids_header)
        logger.info(_purple(f"[skip_full] Using cyclic instead of full permutations (k={k_opts} → {k_opts}x inferences)"))

    # prepare_eval_samples
    def prepare_eval_samples(subject):
        df = pd.read_csv(open(f'{data_path}/test/{subject}_test.csv'), names=("Question", *option_ids_header, "Answer"), dtype=str)

        if setting is not None and setting.startswith('move'):
            df = df.apply(lambda x: move_answer(x, moved_answer), axis=1)

        # NOTE: full permutation은 k!로 급증하므로, k>=5에서는 full이라도 cyclic만 생성(자동 다운그레이드)
        # --skip_full: full이어도 cyclic만 사용 (MMLU 등 오래 걸릴 때)
        k_opts = len(option_ids_header)
        full_permutation_disabled = (setting == 'full' and k_opts >= 5) or bool(getattr(args, 'skip_full', False))

        if setting in ['perm'] or (setting == 'full' and not full_permutation_disabled):
            inputs = df.apply(lambda x: [
                [
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], permuted_options),
                ] for permuted_options in permute_options([x[e] for e in option_ids_header])
            ], axis=1).to_list()
        elif setting in ['cyclic'] or (setting == 'full' and full_permutation_disabled):
            inputs = df.apply(lambda x: [
                [
                    sys_msg.format(subject.replace('_', ' ')),
                    create_user_prompt(x["Question"], cycled_options),
                ] for cycled_options in cycle_options([x[e] for e in option_ids_header])
            ], axis=1).to_list()
        else:
            inputs = df.apply(lambda x: [
                sys_msg.format(subject.replace('_', ' ')),
                create_user_prompt(x["Question"], [x[e] for e in option_ids_header]),
            ], axis=1).to_list()
        options = df.apply(lambda x: [str(x[e]) for e in option_ids_header], axis=1).to_list()
        ideals = df.apply(lambda x: option_ids[option_ids_header.index(x["Answer"])], axis=1).to_list()
        return list(zip(inputs, options, ideals))
    
    

    # prepare_eval_fn
    if getattr(args, "inference_backend", "local") == "api":
        api_prompt_mode = getattr(args, "api_prompt_mode", "baseline")
        if setting in ['perm', 'cyclic', 'full']:
            prepare_eval_fn = partial(
                prepare_eval_fn_api_perm,
                num_few_shot=num_few_shot,
                option_ids=option_ids,
                api_prompt_mode=api_prompt_mode,
            )
        else:
            prepare_eval_fn = partial(
                prepare_eval_fn_api_base,
                num_few_shot=num_few_shot,
                option_ids=option_ids,
                api_prompt_mode=api_prompt_mode,
            )
    elif setting in ['noid']:
        prepare_eval_fn = partial(prepare_eval_fn_noid, num_few_shot=num_few_shot, option_ids=option_ids)
    elif setting in ['perm', 'cyclic', 'full']:
        prepare_eval_fn = partial(prepare_eval_fn_perm, num_few_shot=num_few_shot, option_ids=option_ids)
    else:
        prepare_eval_fn = partial(prepare_eval_fn_base, num_few_shot=num_few_shot, option_ids=option_ids)

    return subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn


def _api_label_only_instruction(option_ids):
    labels = ", ".join(str(x) for x in option_ids)
    return (
        f"Respond with exactly one option label from: {labels}. "
        'Do not output the word "Answer", an explanation, punctuation, '
        "or any other text."
    )


def _api_messages(
    sys_msg,
    eval_sample,
    few_shot_samples,
    num_few_shot,
    option_ids,
    api_prompt_mode="baseline",
):
    prompt_mode = str(api_prompt_mode or "baseline").strip().lower()
    if prompt_mode not in {"baseline", "label_only"}:
        raise ValueError(f"unsupported api_prompt_mode: {api_prompt_mode}")
    system_content = str(sys_msg)
    if prompt_mode == "label_only":
        system_content += " " + _api_label_only_instruction(option_ids)
    messages = [{"role": "system", "content": system_content}]
    for row in few_shot_samples[:num_few_shot]:
        text = str(row)
        try:
            prompt, answer = text.rsplit(" ", 1)
        except ValueError:
            prompt, answer = text, ""
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": str(eval_sample)})
    return messages


def prepare_eval_fn_api_base(
    model, toker, few_shot_samples, num_few_shot, option_ids, api_prompt_mode="baseline"
):
    del toker

    def eval_fn(sample, rng: random.Random):
        del rng
        idx, (input_pair, options, ideal) = sample
        sys_msg, eval_sample = input_pair.copy()
        messages = _api_messages(
            sys_msg, eval_sample, few_shot_samples, num_few_shot,
            option_ids, api_prompt_mode,
        )
        response = model.complete_labels(messages, option_ids)
        probs = np.asarray(response.label_probs, dtype=np.float64)
        sampled = option_ids[int(np.argmax(probs))]
        return {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': str(sys_msg) + '\n\n' + str(eval_sample),
                'messages': messages,
                'api_prompt_mode': str(api_prompt_mode),
                'api_scoring_mode': str(getattr(model, 'scoring_mode', 'topk_strict')),
                'api_equal_label_bias': (
                    float(getattr(model, 'equal_label_bias', 100.0))
                    if getattr(model, 'scoring_mode', 'topk_strict') == 'equal_label_bias'
                    else None
                ),
                'options': options,
                'probs': probs.tolist(),
                'sampled': sampled,
                'ideal': ideal,
                'correct': sampled == ideal,
                'api_calls': [response.to_dict()],
            },
        }
    return eval_fn


def prepare_eval_fn_api_perm(
    model, toker, few_shot_samples, num_few_shot, option_ids, api_prompt_mode="baseline"
):
    del toker

    def eval_fn(sample, rng: random.Random):
        del rng
        idx, (probing_inputs, options, ideal) = sample
        if len(probing_inputs) <= 0:
            raise ValueError("probing_inputs must contain at least one prompt")
        all_probs = []
        all_messages = []
        api_calls = []
        prompts = []
        for sys_msg, eval_sample in probing_inputs:
            messages = _api_messages(
                sys_msg, eval_sample, few_shot_samples, num_few_shot,
                option_ids, api_prompt_mode,
            )
            response = model.complete_labels(messages, option_ids)
            all_probs.append([float(x) for x in response.label_probs])
            all_messages.append(messages)
            prompts.append(str(sys_msg) + '\n\n' + str(eval_sample))
            api_calls.append(response.to_dict())
        return {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': prompts[0],
                'prompts': prompts,
                'messages': all_messages,
                'api_prompt_mode': str(api_prompt_mode),
                'api_scoring_mode': str(getattr(model, 'scoring_mode', 'topk_strict')),
                'api_equal_label_bias': (
                    float(getattr(model, 'equal_label_bias', 100.0))
                    if getattr(model, 'scoring_mode', 'topk_strict') == 'equal_label_bias'
                    else None
                ),
                'options': options,
                'probs': all_probs,
                'ideal': ideal,
                'api_calls': api_calls,
            },
        }
    return eval_fn


def prepare_eval_fn_base(model, toker, few_shot_samples, num_few_shot, option_ids):
    # NOTE: 일부 Seq2Seq 토크나이저(T5 등)는 기본적으로 EOS 같은 special token을 붙여서
    # `input_ids[-1]`가 항상 EOS가 되는 문제가 있음. 반드시 special token 없이 비교/추출한다.
    bpe_has_space_prefix = (
        toker(": A", add_special_tokens=False).input_ids[-1]
        != toker(":A", add_special_tokens=False).input_ids[-1]
    )

    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal) = sample
        sys_msg, eval_sample = input.copy()
        input_text = sys_msg + '\n\n'
        if num_few_shot > 0:
            for s in few_shot_samples[:num_few_shot]:
                input_text += s + '\n\n'
        input_text += eval_sample
        if not bpe_has_space_prefix:
            input_text += ' '

        input_ids = toker(input_text, return_tensors="pt").input_ids.to(model.device)
        input_ids = input_ids[..., -1536:]
        is_seq2seq = getattr(getattr(model, "config", None), "is_encoder_decoder", False)
        with torch.no_grad():
            if is_seq2seq:
                cfg = getattr(model, "config", None)
                dec_start = getattr(cfg, "decoder_start_token_id", None) or getattr(cfg, "pad_token_id", 0)
                dec_ids = torch.full((input_ids.size(0), 1), dec_start, dtype=torch.long, device=model.device)
                logits = model(input_ids=input_ids, decoder_input_ids=dec_ids).logits[:, -1].view(-1)
            else:
                logits = model(input_ids=input_ids).logits[:, -1].view(-1)

        option_indices = (
            [toker(f": {e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
            + [toker(f":{e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
        )
        probs = F.softmax(
            logits[..., option_indices], dim=-1
        ).detach().cpu().to(torch.float32).numpy()
        probs = probs.reshape(2, len(option_ids)).sum(axis=0)
        sampled = option_ids[np.argmax(probs)]

        correct = (sampled == ideal)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'probs': probs.tolist(),
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_noid(model, toker, few_shot_samples, num_few_shot, option_ids):
    toker.padding_side = 'right'
    bpe_has_space_prefix = (
        toker(": A", add_special_tokens=False).input_ids[-1]
        != toker(":A", add_special_tokens=False).input_ids[-1]
    )

    def eval_fn(sample, rng: random.Random):
        idx, (input, options, ideal) = sample
        sys_msg, eval_sample = input.copy()
        input_text = sys_msg + '\n\n'
        if num_few_shot > 0:
            for s in few_shot_samples[:num_few_shot]:
                input_text += s + '\n\n'
        input_text += eval_sample
        prefix_input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids

        losses = []
        lengths = []
        for option in options:
            prefix_and_option_text = input_text + ' ' + option.strip()
            input_ids = toker(prefix_and_option_text, truncation=False, return_tensors="pt").input_ids.to(model.device)
            lengths.append(input_ids.size(1) - prefix_input_ids.size(1))

            labels = input_ids.clone()
            labels[:, :prefix_input_ids.size(1)] = -100

            input_ids = input_ids[..., -1536:]
            labels = labels[..., -1536:]

            with torch.no_grad():
                loss = model(
                    input_ids=input_ids,
                    labels=labels,
                ).loss.detach().to(torch.float32).cpu().item()
            losses.append(loss)

        nll = - np.array(losses)
        probs = np.exp(nll - np.max(nll))
        probs = probs / (probs.sum() + 1e-10)

        sampled = option_ids[np.argmin(losses)]
        correct = (sampled == ideal)
        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_text,
                'options': options,
                'lengths': lengths,
                'losses': losses,
                'probs': probs.tolist(),
                'sampled': sampled,
                'ideal': ideal,
                'correct': correct,
            },
        }
        return result
    return eval_fn


def prepare_eval_fn_perm(model, toker, few_shot_samples, num_few_shot, option_ids):
    toker.padding_side = 'left'
    bpe_has_space_prefix = (
        toker(": A", add_special_tokens=False).input_ids[-1]
        != toker(":A", add_special_tokens=False).input_ids[-1]
    )

    def eval_fn(sample, rng: random.Random):
        idx, (probing_inputs, options, ideal) = sample
        if len(probing_inputs) <= 0:
            raise ValueError("probing_inputs must contain at least one prompt")

        input_texts = []
        for probing_input in probing_inputs:
            sys_msg, eval_sample = probing_input.copy()
            input_text = sys_msg + '\n\n'
            if num_few_shot > 0:
                for s in few_shot_samples[:num_few_shot]:
                    input_text += s + '\n\n'
            input_text += eval_sample
            if not bpe_has_space_prefix:
                input_text += ' '
            input_texts.append(input_text)

        is_seq2seq = getattr(getattr(model, "config", None), "is_encoder_decoder", False)
        decoder_start = None
        if is_seq2seq:
            cfg = getattr(model, "config", None)
            decoder_start = getattr(cfg, "decoder_start_token_id", None) or getattr(cfg, "pad_token_id", 0)

        all_probs = []
        for input_text in input_texts:
            input_ids = toker(input_text, truncation=False, return_tensors="pt").input_ids.to(model.device)
            input_ids = input_ids[..., -1536:]
            with torch.no_grad():
                if is_seq2seq:
                    dec_ids = torch.full(
                        (input_ids.size(0), 1),
                        decoder_start,
                        dtype=torch.long,
                        device=model.device,
                    )
                    logits = model(input_ids=input_ids, decoder_input_ids=dec_ids).logits[:, -1]
                else:
                    logits = model(input_ids=input_ids).logits[:, -1]

            option_indices = (
                [toker(f": {e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
                + [toker(f":{e}", add_special_tokens=False).input_ids[-1] for e in option_ids]
            )
            probs = F.softmax(
                logits[..., option_indices], dim=-1,
            ).detach().to(torch.float32).cpu().numpy()
            probs = probs.reshape(input_ids.size(0), 2, len(option_ids)).sum(axis=1)
            all_probs.extend(probs.tolist())

        result = {
            'type': 'result',
            'data': {
                'idx': idx,
                'prompt': input_texts[0],
                'prompts': input_texts,
                'options': options,
                'probs': all_probs,
                'ideal': ideal,
            },
        }
        return result
    return eval_fn
