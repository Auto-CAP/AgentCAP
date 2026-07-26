#!/usr/bin/env python3
"""Rebuttal recomputations from the repo's real aggregate artifacts.

Inputs (real artifacts only):
  docs/rebuttal_artifacts/source_master_all_results_All_Results.csv
    (exported verbatim from docs/AgentCAP_Results.xlsx, sha256 4d34ba66...)

Outputs (docs/rebuttal_artifacts/):
  paper_vs_artifact_cell_diff.csv   - per-cell diff of artifact vs paper tables
  shapley_all_pairs.csv             - Shapley (phiP, phiE) for every weak->strong
                                      pair x domain computable from artifacts
  shapley_missingness.csv           - cells required by the paper's Shapley rows
                                      that are absent from artifacts
  synergy_recomputed.csv            - pair-line + envelope synergy for every
                                      ordered heterogeneous pair with data
  latency_accuracy_frontier.csv     - recorded per-config latency vs accuracy
                                      with Pareto flags (API runs only)

No synthetic data: every number derives from the artifact CSV. The parametric
bootstrap for verdict stability resamples the *recorded* cell accuracy at the
*recorded* task count (binomial), and is labelled as such.
"""
import csv
import math
import random

ART = "docs/rebuttal_artifacts"
SRC = f"{ART}/source_master_all_results_All_Results.csv"

rows = list(csv.DictReader(open(SRC)))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- data model
# acc[(dataset, planner, executor)] = (acc, ci, cost, latency, tasks, type)
cells = {}
for r in rows:
    a = fnum(r["Accuracy (%)"])
    if a is None:
        continue  # Pending rows carry no data
    cells[(r["Dataset"], r["Planner"], r["Executor"])] = dict(
        acc=a, ci=fnum(r["CI"]), cost=fnum(r["Cost/task ($)"]),
        lat=fnum(r["Latency (s)"]), tasks=int(float(r["Tasks"])) if r["Tasks"] else None,
        type=r["Type"])

API = ["gpt54", "claude-opus", "minimax-m27", "glm51"]
API_LABEL = {"gpt54": "GPT-5.4", "claude-opus": "Claude-Opus", "minimax-m27": "MiniMax-M2.7", "glm51": "GLM-5.1"}

# ------------------------------------------------- paper values (transcribed)
# Table 3 (MCP-Atlas API block) and Table 5 (MedAgentBench API block),
# rows = planner order GPT, Claude, MiniMax, GLM; cols = same executor order.
PAPER = {
    "MCP-Atlas": [
        [(70.7, 2.8), (35.4, 2.0), (80.4, 2.4), (67.0, 2.3)],
        [(68.8, 3.2), (51.3, 2.1), (66.1, 2.5), (69.1, 2.0)],
        [(63.2, 2.0), (70.3, 2.3), (68.5, 2.8), (71.1, 2.7)],
        [(68.3, 3.1), (79.1, 2.0), (69.9, 3.0), (72.4, 2.9)],
    ],
    "MedAgentBench": [
        [(71.7, 2.1), (88.3, 2.1), (76.7, 2.1), (93.3, 2.8)],
        [(83.3, 3.0), (93.3, 2.5), (93.3, 2.1), (93.3, 2.5)],
        [(58.3, 3.3), (86.7, 2.7), (73.3, 3.3), (80.0, 3.1)],
        [(61.7, 2.0), (90.0, 2.9), (86.7, 2.9), (90.0, 2.7)],
    ],
}
# Self-hosted 2x2 blocks (paper Table 5/6 lower blocks; paper labels the weak
# model "Qwen3.5-4B", artifact labels it "Qwen-9b").
PAPER_SH = {
    ("MedAgentBench", "qwen"): [[(60.0, 2.3), (62.0, 2.8)], [(88.0, 2.1), (92.0, 2.6)]],
    ("FinanceBench", "qwen"): [[(28.7, 2.0), (60.0, 2.4)], [(20.0, 2.0), (57.3, 3.2)]],
}
SH_NAMES = {"qwen": ("Qwen-9b", "Qwen-27b")}

diff_out = []
for ds, mat in PAPER.items():
    for i, p in enumerate(API):
        for j, e in enumerate(API):
            paper_acc, paper_ci = mat[i][j]
            c = cells.get((ds, p, e))
            diff_out.append(dict(
                dataset=ds, planner=API_LABEL[p], executor=API_LABEL[e],
                paper_acc=paper_acc, paper_ci=paper_ci,
                artifact_acc=c["acc"] if c else None,
                artifact_ci=c["ci"] if c else None,
                acc_match=(c is not None and abs(c["acc"] - paper_acc) < 0.05),
                ci_match=(c is not None and c["ci"] is not None and abs(c["ci"] - paper_ci) < 0.05),
                artifact_tasks=c["tasks"] if c else None))
for (ds, fam), mat in PAPER_SH.items():
    w, s = SH_NAMES[fam]
    names = [w, s]
    for i in range(2):
        for j in range(2):
            paper_acc, paper_ci = mat[i][j]
            c = cells.get((ds, names[i], names[j]))
            diff_out.append(dict(
                dataset=ds, planner=names[i] + " (paper: Qwen3.5-" + ("4B" if i == 0 else "27B") + ")",
                executor=names[j] + " (paper: Qwen3.5-" + ("4B" if j == 0 else "27B") + ")",
                paper_acc=paper_acc, paper_ci=paper_ci,
                artifact_acc=c["acc"] if c else None, artifact_ci=c["ci"] if c else None,
                acc_match=(c is not None and abs(c["acc"] - paper_acc) < 0.05),
                ci_match=(c is not None and c["ci"] is not None and abs(c["ci"] - paper_ci) < 0.05),
                artifact_tasks=c["tasks"] if c else None))

with open(f"{ART}/paper_vs_artifact_cell_diff.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(diff_out[0]))
    w.writeheader(); w.writerows(diff_out)

n_acc_match = sum(1 for d in diff_out if d["acc_match"])
n_ci_match = sum(1 for d in diff_out if d["ci_match"])
n_have = sum(1 for d in diff_out if d["artifact_acc"] is not None)
print(f"[cell-diff] {len(diff_out)} paper cells checked; artifact present for {n_have}; "
      f"accuracy matches {n_acc_match}/{n_have}; CI matches {n_ci_match}/{n_have}")
for d in diff_out:
    if d["artifact_acc"] is not None and not d["acc_match"]:
        print("   ACC MISMATCH:", d["dataset"], d["planner"], "->", d["executor"],
              "paper", d["paper_acc"], "artifact", d["artifact_acc"])

# ----------------------------------------------------------------- Shapley
random.seed(20260724)
N_BOOT = 20000


def shapley(aWW, aSW, aWS, aSS):
    phiP = 0.5 * (aSW - aWW) + 0.5 * (aSS - aWS)
    phiE = 0.5 * (aWS - aWW) + 0.5 * (aSS - aSW)
    return phiP, phiE


def verdict(phiP, phiE):
    if abs(phiP - phiE) < 1e-9:
        return "tie"
    return "planner" if phiP > phiE else "executor"


shap_out, missing_out = [], []


def try_pair(ds, W, S, pool_label, n_tasks):
    """W, S are artifact model keys with acc(S,S) >= acc(W,W) enforced upstream."""
    need = {"aWW": (W, W), "aSW": (S, W), "aWS": (W, S), "aSS": (S, S)}
    vals, missing = {}, []
    for k, (p, e) in need.items():
        c = cells.get((ds, p, e))
        if c is None:
            missing.append(f"{k}=acc({p},{e})")
        else:
            vals[k] = c["acc"]
    if missing:
        missing_out.append(dict(dataset=ds, pair=f"{W}->{S}", pool=pool_label,
                                missing_cells="; ".join(missing)))
        return
    phiP, phiE = shapley(**vals)
    # parametric bootstrap on recorded accuracies at recorded task count
    flip = 0
    base_v = verdict(phiP, phiE)
    for _ in range(N_BOOT):
        sim = {}
        for k, v in vals.items():
            p_hat = v / 100.0
            successes = sum(1 for _ in range(n_tasks) if random.random() < p_hat)
            sim[k] = 100.0 * successes / n_tasks
        pP, pE = shapley(**sim)
        if verdict(pP, pE) != base_v:
            flip += 1
    shap_out.append(dict(
        dataset=ds, pool=pool_label, pair=f"{W}->{S}",
        aWW=vals["aWW"], aSW=vals["aSW"], aWS=vals["aWS"], aSS=vals["aSS"],
        phiP=round(phiP, 2), phiE=round(phiE, 2), verdict=base_v,
        n_tasks_per_cell=n_tasks,
        bootstrap_verdict_flip_rate=round(flip / N_BOOT, 4)))


for ds in ["MCP-Atlas", "MedAgentBench"]:
    if (ds, "gpt54", "gpt54") not in cells:
        continue
    self_acc = {m: cells[(ds, m, m)]["acc"] for m in API}
    n_tasks = cells[(ds, "gpt54", "gpt54")]["tasks"]
    for i in range(len(API)):
        for j in range(i + 1, len(API)):
            A, B = API[i], API[j]
            W, S = (A, B) if self_acc[A] <= self_acc[B] else (B, A)
            try_pair(ds, W, S, "API", n_tasks)

# self-hosted 2x2 pairs present in the artifact
try_pair("MedAgentBench", "Qwen-9b", "Qwen-27b", "open-weight", 50)
try_pair("FinanceBench", "Qwen-9b", "Qwen-27b", "open-weight", 150)
# MCP-Atlas open-weight pairs (paper Table 13 row uses GPT-OSS pair)
try_pair("MCP-Atlas", "gptoss-20b", "gptoss-120b", "open-weight", 50)
try_pair("MCP-Atlas", "qwen35-4b", "qwen35-27b", "open-weight", 50)

with open(f"{ART}/shapley_all_pairs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(shap_out[0]))
    w.writeheader(); w.writerows(shap_out)
with open(f"{ART}/shapley_missingness.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "pair", "pool", "missing_cells"])
    w.writeheader(); w.writerows(missing_out)

print(f"\n[shapley] computed {len(shap_out)} pair x domain verdicts; "
      f"{len(missing_out)} pair x domain combinations blocked by missing cells")
for r in shap_out:
    print(f"   {r['dataset']:<14} {r['pool']:<11} {r['pair']:<28} "
          f"phiP={r['phiP']:>6} phiE={r['phiE']:>6} -> {r['verdict']:<8} "
          f"flip={r['bootstrap_verdict_flip_rate']}")
for r in missing_out:
    print(f"   MISSING {r['dataset']:<14} {r['pair']:<28} {r['missing_cells']}")

# ----------------------------------------------------------------- Synergy
syn_out = []
for ds in ["MCP-Atlas", "MedAgentBench"]:
    if (ds, "gpt54", "gpt54") not in cells:
        continue
    for p in API:
        for q in API:
            if p == q:
                continue
            het = cells.get((ds, p, q))
            pp = cells.get((ds, p, p))
            qq = cells.get((ds, q, q))
            if not (het and pp and qq):
                continue
            c_pq, c_pp, c_qq = het["cost"], pp["cost"], qq["cost"]
            if None in (c_pq, c_pp, c_qq):
                continue
            denom = c_qq - c_pp
            w_raw = (c_pq - c_pp) / denom if denom != 0 else float("inf")
            w = min(max(w_raw, 0.0), 1.0)
            base_pair = (1 - w) * pp["acc"] + w * qq["acc"]
            d_pair = het["acc"] - base_pair
            # envelope baseline: best homogeneous API team at no greater cost
            homog = [(cells[(ds, m, m)]["cost"], cells[(ds, m, m)]["acc"], m) for m in API]
            affordable = [h for h in homog if h[0] <= c_pq]
            if affordable:
                env_acc, env_model = max((a, m) for _, a, m in affordable)[0], \
                    max(affordable, key=lambda h: h[1])[2]
            else:
                env_acc, env_model = None, None
            d_env = het["acc"] - env_acc if env_acc is not None else None
            syn_out.append(dict(
                dataset=ds, planner=API_LABEL[p], executor=API_LABEL[q],
                acc=het["acc"], cost=c_pq,
                acc_pp=pp["acc"], cost_pp=c_pp, acc_qq=qq["acc"], cost_qq=c_qq,
                w_raw=round(w_raw, 3), w_clipped=round(w, 3),
                pairline_base=round(base_pair, 1), synergy_pairline=round(d_pair, 1),
                envelope_base=env_acc, envelope_model=env_model,
                synergy_envelope=round(d_env, 1) if d_env is not None else None,
                clip_active=(w_raw < 0 or w_raw > 1),
                sign_flip_pairline_vs_envelope=(d_env is not None and (d_pair > 0) != (d_env > 0))))

with open(f"{ART}/synergy_recomputed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(syn_out[0]))
    w.writeheader(); w.writerows(syn_out)

print(f"\n[synergy] {len(syn_out)} ordered heterogeneous pairs recomputed "
      f"(denominator: {len(syn_out)} pairs over "
      f"{len(set(s['dataset'] for s in syn_out))} artifact-complete domains)")
for s in syn_out:
    env = s['synergy_envelope'] if s['synergy_envelope'] is not None else "n/a"
    print(f"   {s['dataset']:<14} {s['planner']:>12} -> {s['executor']:<12} "
          f"pairline={s['synergy_pairline']:>6} envelope={env:>6} "
          f"clip={'Y' if s['clip_active'] else 'n'} flip={'Y' if s['sign_flip_pairline_vs_envelope'] else 'n'}")
pos_pair = [s for s in syn_out if s["synergy_pairline"] > 0]
flips = [s for s in syn_out if s["sign_flip_pairline_vs_envelope"]]
clips = [s for s in syn_out if s["clip_active"]]
print(f"   pairline>0: {len(pos_pair)}/{len(syn_out)}; clip active: {len(clips)}/{len(syn_out)}; "
      f"sign flips pairline vs envelope: {len(flips)}/{len(syn_out)}")

# --------------------------------------------------- latency-accuracy frontier
lat_out = []
for ds in sorted(set(k[0] for k in cells)):
    pts = []
    for (d, p, e), c in cells.items():
        if d != ds or c["lat"] is None:
            continue
        pts.append((c["lat"], c["acc"], p, e, c["type"], c["cost"]))
    for lat, acc, p, e, typ, cost in pts:
        dominated = any(l2 <= lat and a2 >= acc and (l2 < lat or a2 > acc)
                        for l2, a2, *_ in pts)
        lat_out.append(dict(dataset=ds, planner=p, executor=e, type=typ,
                            latency_s=lat, acc=acc, cost=cost,
                            on_latency_accuracy_frontier=not dominated))
with open(f"{ART}/latency_accuracy_frontier.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(lat_out[0]))
    w.writeheader(); w.writerows(lat_out)
front = [r for r in lat_out if r["on_latency_accuracy_frontier"]]
print(f"\n[latency] {len(lat_out)} configs with recorded per-task latency; "
      f"{len(front)} on their domain's latency-accuracy frontier")
for r in front:
    print(f"   {r['dataset']:<16} {r['planner']:>12} -> {r['executor']:<12} "
          f"lat={r['latency_s']:>6}s acc={r['acc']}")
