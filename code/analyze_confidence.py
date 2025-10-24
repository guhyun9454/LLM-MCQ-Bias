import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

# Early register bundled Korean font BEFORE importing pyplot/seaborn
from matplotlib import font_manager as _fm
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_bundled_font = os.path.join(_repo_root, "font", "NanumGothic.ttf")
if os.path.isfile(_bundled_font):
    _fm.fontManager.addfont(_bundled_font)
    try:
        _font_name = _fm.FontProperties(fname=_bundled_font).get_name()
    except Exception:
        _font_name = "NanumGothic"
    matplotlib.rcParams["font.family"] = _font_name
    matplotlib.rcParams["font.sans-serif"] = [_font_name]
    matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import seaborn as sns


def configure_korean_fonts() -> None:
    """Use only the repo-bundled font at font/NanumGothic.ttf without fallbacks."""
    import matplotlib.font_manager as fm
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bundled_font = os.path.join(repo_root, "font", "NanumGothic.ttf")
    if not os.path.isfile(bundled_font):
        raise FileNotFoundError(f"한글 폰트를 찾을 수 없습니다: {bundled_font}")
    fm.fontManager.addfont(bundled_font)
    try:
        selected = fm.FontProperties(fname=bundled_font).get_name()
    except Exception:
        selected = "NanumGothic"

    plt.rcParams["font.family"] = selected
    plt.rcParams["font.sans-serif"] = [selected]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class RunKey:
    task: str
    shots: int
    model: str
    setting: Optional[str]
    leaf_name: str  # e.g., "arc" or "arc_noid"
    dir_path: str


def find_result_runs(roots: List[str]) -> List[RunKey]:
    """
    Discover leaf directories that contain per-subject jsonl results.
    Supports both the current layout (results/<task>/<shots>s_<model>/<leaf>)
    and legacy ones like (results_arc/<shots>s_<model>/arc).
    """
    discovered: List[RunKey] = []

    def try_parse(path_parts: List[str], dir_path: str) -> Optional[RunKey]:
        # Accept patterns:
        # - results/<task>/<shots>s_<model>/<leaf>
        # - results_* / <shots>s_<model> / <task or leaf>
        if len(path_parts) < 3:
            return None

        # Heuristics per layout
        if path_parts[0] == "results":
            # results/<task>/<shots>s_<model>/<leaf>
            task = path_parts[1]
            shots_model = path_parts[2]
            leaf = path_parts[3] if len(path_parts) >= 4 else path_parts[-1]
        else:
            # e.g., results_arc/<shots>s_<model>/<task>
            # infer task from root name suffix if possible
            root_name = path_parts[0]
            task = root_name.split("results_")[-1] if "results_" in root_name else root_name
            shots_model = path_parts[1]
            leaf = path_parts[2] if len(path_parts) >= 3 else path_parts[-1]

        if "s_" not in shots_model:
            return None
        try:
            shots_str, model = shots_model.split("s_", 1)
            shots = int(shots_str)
        except Exception:
            return None

        # Extract setting from leaf if present (e.g., arc_noid, arc_perm, arc_cyclic, arc_movea)
        setting = None
        if "_" in leaf:
            suffix = leaf.split("_", 1)[1]
            # only keep recognized settings
            if suffix in ["noid", "perm", "cyclic", "shuffle_both"] or (
                suffix.startswith("move") and len(suffix) == 5 and suffix[-1] in "abcdABCD"
            ):
                setting = suffix

        return RunKey(task=task, shots=shots, model=model, setting=setting, leaf_name=leaf, dir_path=dir_path)

    for root in roots:
        if not os.path.isdir(root):
            continue
        for dir_path, dirnames, filenames in os.walk(root):
            if any(f.endswith(".jsonl") for f in filenames):
                parts = os.path.normpath(dir_path).split(os.sep)
                rk = try_parse(parts, dir_path)
                if rk is not None:
                    discovered.append(rk)
    return discovered


def load_results_from_dir(dir_path: str) -> List[dict]:
    """Load all result entries from jsonl files within dir_path."""
    entries: List[dict] = []
    for name in sorted(os.listdir(dir_path)):
        if not name.endswith(".jsonl"):
            continue
        file_path = os.path.join(dir_path, name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("type") != "result":
                        continue
                    data = obj.get("data", {})
                    # Fill subject from filename if missing
                    if data.get("subject") in (None, ""):
                        inferred = os.path.splitext(name)[0]
                        data = dict(data)
                        data["subject"] = inferred
                    entries.append(data)
        except FileNotFoundError:
            continue
    # Sort by subject (implicit via filename) then idx
    entries = sorted(entries, key=lambda d: (d.get("subject", ""), int(d.get("idx", -1))))
    return entries


def compute_confidence(data: dict) -> Optional[float]:
    probs = data.get("probs")
    if probs is None:
        probs = data.get("observed")  # ichat style
    if probs is None:
        return None
    # Some variants store nested lists; we expect a flat list per sample for base/noid
    if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], list):
        # Cannot define single-sample correctness/confidence for perm/cyclic here; skip
        return None
    try:
        arr = np.array(probs, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            return None
        # Define confidence as max probability (prob of sampled choice)
        return float(np.max(arr))
    except Exception:
        return None


def build_dataframe(run: RunKey, entries: List[dict]) -> pd.DataFrame:
    records: List[Dict] = []
    for e in entries:
        conf = compute_confidence(e)
        if conf is None:
            continue
        correct = e.get("correct")
        # Some settings (perm/cyclic) do not provide correctness
        if correct is None:
            continue
        subj = e.get("subject")  # may be absent; we can infer from file name but not passed here
        idx = e.get("idx")
        records.append({
            "task": run.task,
            "shots": run.shots,
            "model": run.model,
            "setting": run.setting or "base",
            "leaf": run.leaf_name,
            "subject": subj,
            "idx": idx,
            "correct": bool(correct),
            "confidence": float(conf),
        })
    return pd.DataFrame.from_records(records)


def reliability_curve(df: pd.DataFrame, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = df["confidence"].to_numpy()
    corr = df["correct"].astype(int).to_numpy()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            bin_acc[b] = np.nan
            bin_conf[b] = np.nan
            continue
        bin_cnt[b] = int(mask.sum())
        bin_acc[b] = float(np.mean(corr[mask]))
        bin_conf[b] = float(np.mean(conf[mask]))
    return bins, bin_acc, bin_conf


def expected_calibration_error(df: pd.DataFrame, n_bins: int = 10) -> float:
    bins, bin_acc, bin_conf = reliability_curve(df, n_bins=n_bins)
    conf = df["confidence"].to_numpy()
    bin_ids = np.digitize(conf, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        gap = abs((bin_acc[b] if not math.isnan(bin_acc[b]) else 0.0) - (bin_conf[b] if not math.isnan(bin_conf[b]) else 0.0))
        ece += (mask.sum() / len(df)) * gap
    return float(ece)


def plot_and_log(df: pd.DataFrame, title_prefix: str, save_dir: str, use_wandb: bool, wandb_run) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    file_paths: Dict[str, str] = {}

    # Prepare common plotting DataFrame with Korean labels
    plot_df = df.assign(정답여부=df["correct"].map({True: "정답", False: "오답"}))

    # Prepare font properties using bundled font
    from matplotlib.font_manager import FontProperties
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bundled_font = os.path.join(repo_root, "font", "NanumGothic.ttf")
    font_prop = FontProperties(fname=bundled_font)

    def apply_korean_font(ax):
        ax.title.set_fontproperties(font_prop)
        ax.xaxis.label.set_fontproperties(font_prop)
        ax.yaxis.label.set_fontproperties(font_prop)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontproperties(font_prop)

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=plot_df, x="confidence", hue="정답여부",
                 bins=20, stat="density", common_norm=False, alpha=0.5,
                 palette=["#1f77b4", "#d62728"], ax=ax)
    ax.set_title(f"{title_prefix} - 정답/오답별 확률 분포 (히스토그램)", fontproperties=font_prop)
    ax.set_xlabel("Confidence (선택 확률)", fontproperties=font_prop)
    ax.set_ylabel("밀도", fontproperties=font_prop)
    apply_korean_font(ax)
    hist_path = os.path.join(save_dir, "histogram.png")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    file_paths["histogram"] = hist_path

    # Boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=plot_df, x="정답여부", y="confidence", ax=ax)
    sns.stripplot(data=plot_df, x="정답여부", y="confidence", color="black", size=2, alpha=0.25, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title_prefix} - 정답/오답별 Confidence 박스플롯", fontproperties=font_prop)
    ax.set_xlabel("", fontproperties=font_prop)
    ax.set_ylabel("Confidence", fontproperties=font_prop)
    apply_korean_font(ax)
    box_path = os.path.join(save_dir, "boxplot.png")
    fig.tight_layout()
    fig.savefig(box_path, dpi=200)
    plt.close(fig)
    file_paths["boxplot"] = box_path

    # Scatter by index (ID)
    fig, ax = plt.subplots(figsize=(10, 4))
    sorted_df = df.copy()
    # Normalize types for robust sorting
    sorted_df["subject"] = sorted_df["subject"].astype(str)
    sorted_df["idx"] = pd.to_numeric(sorted_df["idx"], errors="coerce")
    sorted_df = sorted_df.sort_values(["subject", "idx"], na_position="first").reset_index(drop=True)
    x_vals = np.arange(len(sorted_df))
    colors = sorted_df["correct"].map({True: "#1f77b4", False: "#d62728"}).to_numpy()
    ax.scatter(x_vals, sorted_df["confidence"].to_numpy(), c=colors, s=8, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("샘플 ID (정렬됨)", fontproperties=font_prop)
    ax.set_ylabel("Confidence", fontproperties=font_prop)
    ax.set_title(f"{title_prefix} - ID별 Confidence 분포", fontproperties=font_prop)
    apply_korean_font(ax)
    scatter_path = os.path.join(save_dir, "id_scatter.png")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    file_paths["id_scatter"] = scatter_path

    # Reliability diagram
    bins, bin_acc, bin_conf = reliability_curve(df)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="완전 보정")
    ax.plot(centers, bin_acc, marker="o", label="관측 정답률")
    ax.plot(centers, bin_conf, marker="s", label="평균 Confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin 중심값", fontproperties=font_prop)
    ax.set_ylabel("값", fontproperties=font_prop)
    ax.set_title(f"{title_prefix} - 신뢰도 곡선", fontproperties=font_prop)
    leg = ax.legend()
    apply_korean_font(ax)
    reliability_path = os.path.join(save_dir, "reliability.png")
    fig.tight_layout()
    fig.savefig(reliability_path, dpi=200)
    plt.close(fig)
    file_paths["reliability"] = reliability_path

    # W&B logging
    if use_wandb and wandb_run is not None:
        import wandb
        wandb_run.log({
            "histogram": wandb.Image(hist_path),
            "boxplot": wandb.Image(box_path),
            "id_scatter": wandb.Image(scatter_path),
            "reliability": wandb.Image(reliability_path),
        })

        # Log table of per-sample stats
        table = wandb.Table(columns=["subject", "idx", "correct", "confidence"])  # minimal columns
        for _, row in df.iterrows():
            table.add_data(row.get("subject"), row.get("idx"), bool(row["correct"]), float(row["confidence"]))
        wandb_run.log({"samples": table})

        # Summary metrics
        ece = expected_calibration_error(df)
        wandb_run.summary["accuracy"] = float(df["correct"].mean()) if not df.empty else float("nan")
        wandb_run.summary["ece_10"] = ece

    return file_paths


def init_wandb(project: str, run_name: str, group: Optional[str], tags: List[str], config: Dict) -> Tuple[bool, Optional[object]]:
    """Try to initialize a W&B run. If login fails, return (False, None)."""
    import wandb
    run = wandb.init(entity = "capde", project=project, name=run_name, group=group, tags=tags, config=config, reinit=True)
    return True, run



def analyze_one_run(run: RunKey, wandb_project: str, save_root: str, enable_wandb: bool) -> Optional[Dict[str, str]]:
    entries = load_results_from_dir(run.dir_path)
    if not entries:
        return None
    df = build_dataframe(run, entries)
    if df.empty:
        return None

    title_prefix = f"{run.task.upper()} | {run.shots}s | {run.model} | {run.setting or 'base'}"
    save_dir = os.path.join(save_root, run.task, f"{run.shots}s_{run.model}", run.leaf_name)

    use_wandb = False
    wb_run = None
    if enable_wandb:
        tags = [run.task, f"{run.shots}s", run.model]
        if run.setting:
            tags.append(run.setting)
        group = f"{run.task}-{run.shots}s"
        run_name = f"{run.model}-{run.leaf_name}"
        use_wandb, wb_run = init_wandb(
            project=wandb_project,
            run_name=run_name,
            group=group,
            tags=tags,
            config={
                "task": run.task,
                "shots": run.shots,
                "model": run.model,
                "setting": run.setting or "base",
                "dir": run.dir_path,
                "num_samples": int(len(df)),
            },
        )

    file_paths = plot_and_log(df, title_prefix, save_dir, use_wandb, wb_run)

    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass

    return file_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze per-sample confidence distributions and log plots to W&B.")
    p.add_argument("--roots", type=str, nargs="*", default=["results", "results_arc", "results_csqa", "results_mmlu"],
                   help="Result root directories to scan.")
    p.add_argument("--wandb_project", type=str, default="LLM-MCQ-Confidence",
                   help="W&B project name.")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging (only save images locally).")
    p.add_argument("--save_root", type=str, default="analysis_images",
                   help="Local directory to save generated images.")
    p.add_argument("--filter_task", type=str, default=None, help="Only analyze a specific task (e.g., arc, csqa, mmlu).")
    p.add_argument("--filter_model_contains", type=str, default=None, help="Only analyze runs whose model name contains this string.")
    p.add_argument("--filter_setting", type=str, default=None, help="Only analyze a specific setting (e.g., base, noid). Use 'base' for None.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 폰트는 이미 모듈 임포트 시점에 rcParams에 등록됨. 스타일만 적용.
    sns.set(style="whitegrid")

    run_keys = find_result_runs(args.roots)
    if not run_keys:
        print("[INFO] 결과 디렉터리를 찾지 못했습니다. --roots 인자를 확인하세요.")
        return

    # Apply filters
    filtered: List[RunKey] = []
    for rk in run_keys:
        if args.filter_task and rk.task != args.filter_task:
            continue
        if args.filter_model_contains and (args.filter_model_contains not in rk.model):
            continue
        if args.filter_setting:
            want = args.filter_setting
            if want == "base" and rk.setting is not None:
                continue
            if want != "base" and rk.setting != want:
                continue
        filtered.append(rk)

    if not filtered:
        print("[INFO] 필터 결과 분석 가능한 실행을 찾지 못했습니다.")
        return

    total = 0
    for rk in filtered:
        out = analyze_one_run(
            run=rk,
            wandb_project=args.wandb_project,
            save_root=args.save_root,
            enable_wandb=(not args.no_wandb),
        )
        if out is not None:
            total += 1
            print(f"[OK] {rk.task} | {rk.shots}s | {rk.model} | {rk.leaf_name} -> {len(out)} plots")
        else:
            print(f"[SKIP] {rk.task} | {rk.shots}s | {rk.model} | {rk.leaf_name} (no analyzable entries)")

    print(f"완료: {total}개 실행을 분석했습니다.")


if __name__ == "__main__":
    main()


