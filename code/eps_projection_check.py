"""Offline validation for stage-2 projected eps(q) identification.

Branch: jh/projection_eps

Question
--------
At stage 2 the view-difference system M = [A_0 - A_1] has rank k-2 on the
zero-sum subspace -- one short of identifying eps(q). The min-norm pinv
solution leaks the content component along ker(M) into b; because A_t v = v
for v in ker(M), that leakage lies *exactly* along ker(M). Projecting it out
therefore recovers the identifiable (k-2)-dim part of eps(q) exactly, in the
noiseless linear model. How much of that survives on real caches?

Reference
---------
At full latin depth sum_t A_t = J and c is zero-sum, so

    b_full = mean_t(y_t)

exactly -- an estimator-free 4-view reference.

Everything is computed in b-space (b = mu + eps(q)). mu is a per-component
constant across items, so every across-item correlation reported here equals
the corresponding eps(q) quantity.

Usage:
    python eps_projection_check.py [results_arc_dir] [--runs 0,1,2]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

CLIP = 1e-6


# ---------------------------------------------------------------- primitives
def centered_log(p):
    z = np.log(np.clip(np.asarray(p, dtype=np.float64), CLIP, None))
    return z - z.mean()


def slot_matrix(perm_slot_to_content, k):
    """A[s, i] = 1 iff content i sits at slot s, so that y = b + A c."""
    A = np.zeros((k, k), dtype=np.float64)
    for s, i in enumerate(perm_slot_to_content):
        A[s, int(i)] = 1.0
    return A


def identified_projector(M, k):
    """Orthogonal projector onto the identifiable subspace (zero-sum part of
    row-space(M)).  Full-rank M => projector onto the whole zero-sum space."""
    Q = np.eye(k) - np.ones((k, k)) / float(k)
    MQ = np.asarray(M, dtype=np.float64) @ Q
    _, sv, Vt = np.linalg.svd(MQ)
    tol = 1e-9 * max(1.0, float(sv[0]) if sv.size else 1.0)
    rank = int((sv > tol).sum())
    if rank >= k - 1:
        return Q
    null_dirs = Q @ Vt[rank:].T
    Un, sn, _ = np.linalg.svd(null_dirs, full_matrices=False)
    N = Un[:, sn > tol]
    if N.size == 0:
        return Q
    return Q - N @ N.T


def estimate_b(ys, As, project=False):
    """LS estimate of b = mu + eps(q) from the given views (production formula)."""
    k = ys[0].shape[0]
    M = np.vstack([As[0] - A for A in As[1:]])
    rhs = np.concatenate([ys[0] - y for y in ys[1:]])
    c = np.linalg.pinv(M) @ rhs
    c -= c.mean()
    b = np.mean([y - A @ c for y, A in zip(ys, As)], axis=0)
    b -= b.mean()
    if project:
        b = identified_projector(M, k) @ b
    return b


# ---------------------------------------------------------------- self-test
# --- verbatim copy of the production schedule builder (eval_clm.py) ---------
# Inlined so this check stays dependency-free: importing eval_clm pulls in
# torch, which must not be initialised on the login node.
def _search_latin_assignment(k, allowed_slots, content_idx, used_slots, current):
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


def _content_to_slot_assignment_to_perm(content_to_slot):
    slot_to_content = [0] * len(content_to_slot)
    for content_idx, slot_idx in enumerate(content_to_slot):
        slot_to_content[int(slot_idx)] = int(content_idx)
    return tuple(slot_to_content)


def _build_targeted_latin_schedule(k, top1_idx, runner_idx):
    if k <= 1:
        return [tuple(range(k))]
    schedules_content_to_slot = [list(range(k))]
    if top1_idx == runner_idx:
        runner_idx = (int(top1_idx) + 1) % int(k)
    remaining = [i for i in range(k) if i not in (int(top1_idx), int(runner_idx))]
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
        cand = _search_latin_assignment(k, allowed_slots, 0, set(), [-1] * k)
        if cand is None:
            raise RuntimeError("latin completion failed")
        schedules_content_to_slot.append(cand)
    return [_content_to_slot_assignment_to_perm(s) for s in schedules_content_to_slot]


def self_test(k=4, trials=2000, seed=0):
    """Noiseless synthetic check of the three claims the patch relies on."""
    rng = np.random.default_rng(seed)
    sched = _build_targeted_latin_schedule(k, 2, 0)
    As = [slot_matrix(p, k) for p in sched]
    e_raw, e_proj, e_three = [], [], []
    for _ in range(trials):
        c = rng.normal(size=k); c -= c.mean()
        b = rng.normal(size=k); b -= b.mean()
        ys = [b + A @ c for A in As]
        P2 = identified_projector(As[0] - As[1], k)
        e_raw.append(np.linalg.norm(estimate_b(ys[:2], As[:2]) - b))
        e_proj.append(np.linalg.norm(estimate_b(ys[:2], As[:2], project=True) - P2 @ b))
        e_three.append(np.linalg.norm(estimate_b(ys[:3], As[:3], project=True) - b))
    print("[self-test] k=%d, %d trials, noiseless" % (k, trials))
    print("  2-view min-norm   ||b_hat - b||        = %.4f   (leakage, expected large)" % np.mean(e_raw))
    print("  2-view PROJECTED  ||b_hat - P2 b||     = %.2e   (expected ~0)" % np.mean(e_proj))
    print("  3-view projected  ||b_hat - b||        = %.2e   (projection is a no-op)" % np.mean(e_three))
    ok = np.mean(e_proj) < 1e-10 and np.mean(e_three) < 1e-10 and np.mean(e_raw) > 1e-3
    print("  -> %s" % ("PASS" if ok else "FAIL"))
    return ok


# ---------------------------------------------------------------- real caches
def corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyse_model(model_dir, runs=(0,)):
    rows = []
    for r in runs:
        pat = os.path.join(model_dir, "empirical_analysis",
                           "*_run%d_empirical_alpha*_stage_cache.jsonl" % r)
        for path in sorted(glob.glob(pat)):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("type") != "empirical_stage_cache":
                        continue
                    rows.append(rec)
    if not rows:
        return None

    frac_id, c_proj, c_raw, c_three = [], [], [], []
    tr_full, tr_proj, tr_raw, tr_three = [], [], [], []
    for rec in rows:
        k = int(rec["k"])
        sched = rec["stage_schedule"]
        probs = rec["stage_probs"]
        if len(sched) < k or len(probs) < k:
            continue
        As = [slot_matrix(p, k) for p in sched[:k]]
        ys = [centered_log(p) for p in probs[:k]]

        b_full = np.mean(ys, axis=0)          # exact 4-view reference
        b_full -= b_full.mean()
        P2 = identified_projector(As[0] - As[1], k)

        b2p = estimate_b(ys[:2], As[:2], project=True)
        b2r = estimate_b(ys[:2], As[:2], project=False)
        b3 = estimate_b(ys[:3], As[:3], project=True)

        nf = float(b_full @ b_full)
        if nf > 1e-12:
            frac_id.append(float(b_full @ (P2 @ b_full)) / nf)

        # componentwise, restricted to the stage-2 identifiable subspace
        pf = P2 @ b_full
        c_proj.append((b2p, pf))
        c_raw.append((P2 @ b2r, pf))
        c_three.append((b3, b_full))

        # decision-relevant scalar: top1 - runner contrast of b
        order = np.argsort(-np.asarray(probs[0], dtype=np.float64))
        t1, ru = int(order[0]), int(order[1])
        tr_full.append(b_full[t1] - b_full[ru])
        tr_proj.append(b2p[t1] - b2p[ru])
        tr_raw.append(b2r[t1] - b2r[ru])
        tr_three.append(b3[t1] - b3[ru])

    def flat_corr(pairs):
        a = np.concatenate([x[0] for x in pairs])
        b = np.concatenate([x[1] for x in pairs])
        return corr(a, b)

    return dict(
        n=len(tr_full),
        frac_identifiable=float(np.mean(frac_id)) if frac_id else float("nan"),
        corr_proj_vs_full_id_subspace=flat_corr(c_proj),
        corr_rawminnorm_vs_full_id_subspace=flat_corr(c_raw),
        corr_3view_vs_full=flat_corr(c_three),
        tr_corr_proj=corr(tr_proj, tr_full),
        tr_corr_raw=corr(tr_raw, tr_full),
        tr_corr_3view=corr(tr_three, tr_full),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?",
                    default="/ceph_data/jihye4118/LLM-MCQ-Bias-eps0/code/results_arc")
    ap.add_argument("--tag", default="arc_full_id-ABCD__eps0_latin_0827")
    ap.add_argument("--runs", default="0")
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args()

    if not args.skip_self_test:
        if not self_test():
            print("self-test FAILED -- stopping")
            return 1
        print()

    runs = tuple(int(x) for x in args.runs.split(",") if x.strip())
    models = sorted(glob.glob(os.path.join(args.results_dir, "0s_*")))
    hdr = ("%-34s %6s %7s %9s %9s %9s %9s %9s %9s"
           % ("model", "n", "frac_id", "proj|id", "raw|id", "3v_full",
              "tr_proj", "tr_raw", "tr_3v"))
    print(hdr)
    print("-" * len(hdr))
    acc = []
    for md in models:
        d = os.path.join(md, args.tag)
        if not os.path.isdir(d):
            continue
        res = analyse_model(d, runs=runs)
        if res is None:
            continue
        acc.append(res)
        print("%-34s %6d %7.3f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (os.path.basename(md)[:34], res["n"], res["frac_identifiable"],
                 res["corr_proj_vs_full_id_subspace"],
                 res["corr_rawminnorm_vs_full_id_subspace"],
                 res["corr_3view_vs_full"],
                 res["tr_corr_proj"], res["tr_corr_raw"], res["tr_corr_3view"]))
    if acc:
        print("-" * len(hdr))
        print("%-34s %6s %7.3f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"
              % ("MEAN", "", np.mean([a["frac_identifiable"] for a in acc]),
                 np.mean([a["corr_proj_vs_full_id_subspace"] for a in acc]),
                 np.mean([a["corr_rawminnorm_vs_full_id_subspace"] for a in acc]),
                 np.mean([a["corr_3view_vs_full"] for a in acc]),
                 np.mean([a["tr_corr_proj"] for a in acc]),
                 np.mean([a["tr_corr_raw"] for a in acc]),
                 np.mean([a["tr_corr_3view"] for a in acc])))
    print()
    print("frac_id : share of ||b_full||^2 reachable at stage 2 (1/1 = nothing lost)")
    print("proj|id : corr(projected 2-view, 4-view ref) inside the identifiable subspace")
    print("raw|id  : same for the unprojected min-norm estimate (leakage control)")
    print("3v_full : corr(3-view, 4-view ref), full space -- reproduces the 0.87-0.92 reference")
    print("tr_*    : corr on the top1-runner contrast of b (the decision-relevant scalar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
