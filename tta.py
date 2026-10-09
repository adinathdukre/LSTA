import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from exp.tta_eval import Evaluator

CONFIGS = {
    "ADFTD": dict(data_root="data/ADFTD", d_model=128, num_class=3, v_layer=6, t_layer=6, patch_len=1,
                  augmentations="flip0.,frequency0.,jitter0.,mask0.25,channel0.,drop0.",
                  ckpt_dir="checkpoints/baseline/ADFTD"),
    "APAVA": dict(data_root="data/APAVA", d_model=256, num_class=2, v_layer=6, t_layer=6, patch_len=1,
                  augmentations="flip0.2,frequency0.2,jitter0.,mask0.,channel0.,drop0.4",
                  ckpt_dir="checkpoints/baseline/APAVA"),
    "TDBRAIN": dict(data_root="data/TDBRAIN", d_model=128, num_class=2, v_layer=0, t_layer=6, patch_len=6,
                     augmentations="flip0.,frequency0.2,jitter0.,mask0.,channel0.,drop0.4",
                     ckpt_dir="checkpoints/baseline/TDBRAIN"),
    "PTB": dict(data_root="data/PTB", d_model=256, num_class=2, v_layer=3, t_layer=0, patch_len=1,
                augmentations="flip0.,frequency0.,jitter0.,mask0.,channel0.4,drop0.",
                ckpt_dir="checkpoints/baseline/PTB"),
    "PTB-XL": dict(data_root="data/PTB-XL", d_model=128, num_class=5, v_layer=0, t_layer=5, patch_len=8,
                   augmentations="flip0.,frequency0.,jitter0.,mask0.2,channel0.4,drop0.4",
                   ckpt_dir="checkpoints/baseline/PTB-XL"),
}

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC", "avg"]


def _frozen_metric_dict(full):
    d = {}
    for m in METRIC_NAMES:
        d[f"{m}_mean"] = full[m]
        d[f"{m}_std"] = 0.0
    return d


def _drawn_metric_dict(per_metric):
    """per_metric: {metric_name: (mean, std)} from Evaluator.eval_meta_basis."""
    d = {}
    for m in METRIC_NAMES:
        mean, std = per_metric[m]
        d[f"{m}_mean"] = mean
        d[f"{m}_std"] = std
    return d


def run_train(ev, seed_out, rank, k, eta, outer_steps, outer_lr, outer_batch, n_val_draws,
             adapt_steps_grid=(0.2,), eval_every=50, retrain=False, T=1.0):
    ckpt_path = os.path.join(seed_out, "checkpoint.pt")    #where adapted basis is saved
    required_keys = {"W_init", "W_best", "best_step", "best_adapt_steps", "best_f1"}
    requested_config = dict(rank=rank, k=k, eta=eta, outer_steps=outer_steps, outer_lr=outer_lr,
                            outer_batch=outer_batch, adapt_steps_grid=list(adapt_steps_grid), eval_every=eval_every, T=T)
    reusable = False     # prevent accidental retraining
    if not retrain and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=ev.device)
        missing = required_keys - state.keys()
        if missing:
            print(f"checkpoint at {ckpt_path} mismatch")
        else:
            cached_config = {k_: state.get(k_) for k_ in requested_config}
            mismatched = {k_: (cached_config[k_], v) for k_, v in requested_config.items()
                         if cached_config[k_] != v}
            if mismatched:
                print(f"(checkpoint config mismatch at {ckpt_path}  {{cached: requested}}={mismatched}")
            else:
                reusable = True
    if reusable:    #config should match to reuse
        print("  (reusing saved basis checkpoint)")
        W_init, W_best = state["W_init"].to(ev.device), state["W_best"].to(ev.device)
        best_step, best_adapt_steps = state["best_step"], state["best_adapt_steps"]
        best_f1, best_score = state["best_f1"], state["best_score"]
    else:  # if it is not matching go to adaptation training
        result = ev.train_basis(rank, k, eta, outer_steps, outer_lr=outer_lr, outer_batch=outer_batch,
                                     n_val_draws=n_val_draws, adapt_steps_grid=adapt_steps_grid, eval_every=eval_every, T=T)
        os.makedirs(seed_out, exist_ok=True)
        torch.save(dict(W=result["W"].cpu(), W_init=result["W_init"].cpu(), W_best=result["W_best"].cpu(),
                         best_step=result["best_step"], best_adapt_steps=result["best_adapt_steps"],
                         best_f1=result["best_f1"], best_score=result["best_score"],
                         **requested_config),
                   ckpt_path)
        with open(os.path.join(seed_out, "trajectory.json"), "w") as f:
            json.dump(dict(**requested_config, trajectory=result["trajectory"],
                            best=dict(step=result["best_step"], adapt_steps=result["best_adapt_steps"],
                                      val_f1_mean=result["best_f1"], score=result["best_score"])), f, indent=2)
        W_init, W_best = result["W_init"], result["W_best"]
        best_step, best_adapt_steps = result["best_step"], result["best_adapt_steps"]
        best_f1, best_score = result["best_f1"], result["best_score"]

    print(f"  best: step={best_step} adapt_steps={best_adapt_steps} val_F1={best_f1:.4f} "
          f"(score={best_score:.4f})  [rank={rank} k={k} eta={eta} outer_lr={outer_lr}]")
    return dict(W_init=W_init, W_best=W_best, best_step=best_step,
                best_adapt_steps=best_adapt_steps, best_f1=best_f1, best_score=best_score)


def run_val_controls(ev, seed_out, k, eta, adapt_steps, n_draws, W_init, W_best, T=1.0):
    '''run validation on frozen, random and learned basis'''
    base_val = ev.baseline("val")["full"]
    metrics_r = ev.eval_meta_basis("val", W_init, eta, k, n_draws, adapt_steps, T)
    metrics_l = ev.eval_meta_basis("val", W_best, eta, k, n_draws, adapt_steps, T)

    results = {"FROZEN": _frozen_metric_dict(base_val)}
    results["RANDOM_B"] = _drawn_metric_dict(metrics_r)
    results["LEARNED_B"] = dict(_drawn_metric_dict(metrics_l), adapt_steps=adapt_steps)

    with open(os.path.join(seed_out, "val_controls.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  val (adapt_steps={adapt_steps}): FROZEN={base_val['F1']:.4f} "
          f"RANDOM_B={metrics_r['F1'][0]:.4f} LEARNED_B={metrics_l['F1'][0]:.4f}")
    return results


def run_test_eval(ev, seed_out, rank, k, eta, adapt_steps, n_draws, W_init, W_best, T=1.0):
    '''run test on frozen, random and learned basis'''
    base_test = ev.baseline("test")["full"]
    metrics_r = ev.eval_meta_basis("test", W_init, eta, k, n_draws, adapt_steps, T)
    metrics_l = ev.eval_meta_basis("test", W_best, eta, k, n_draws, adapt_steps, T)

    report = {
        "FROZEN": _frozen_metric_dict(base_test),   #since frozen no std, just mean
        "RANDOM_B": _drawn_metric_dict(metrics_r),
        "LEARNED_B": _drawn_metric_dict(metrics_l),
        "_config": dict(rank=rank, k=k, eta=eta, adapt_steps=adapt_steps, T=T),
    }
    os.makedirs(seed_out, exist_ok=True)
    with open(os.path.join(seed_out, "test_result.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(seed_out, "test_result.txt"), "w") as f:
        f.write(f"config: rank={rank} k={k} eta={eta} adapt_steps={adapt_steps}\n\n")
        for name, r in report.items():
            if name == "_config":
                continue
            f.write(f"{name}:\n")
            for m in METRIC_NAMES:
                f.write(f"  {m}: {r[f'{m}_mean']:.5f} +/- {r[f'{m}_std']:.5f}\n")
    print(f"  test (rank={rank} k={k} eta={eta} adapt_steps={adapt_steps}): FROZEN={base_test['F1']:.4f} "
          f"RANDOM_B={metrics_r['F1'][0]:.4f} LEARNED_B={metrics_l['F1'][0]:.4f}")
    return report


def aggregate(dataset, seed_reports, out_dir):
    '''report final test metrics (all of METRIC_NAMES, not just F1) aggregated across seeds'''
    seeds = sorted(seed_reports.keys())
    methods = ["FROZEN", "RANDOM_B", "LEARNED_B"]
    configs = {s: seed_reports[s].get("_config") for s in seeds}
    summary = {"dataset": dataset, "seeds": seeds, "configs": configs, "sections": {}}
    for m in methods:
        sec = {}
        for metric in METRIC_NAMES:
            key = f"{metric}_mean"
            vals = [seed_reports[s][m][key] for s in seeds if m in seed_reports[s]]
            if not vals:
                continue
            sec[metric] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), per_seed=vals)
        if sec:
            summary["sections"][m] = sec

    lines = [f"{dataset}  seeds={seeds}  (meta_ln_basis primary workflow)",
             f"per-seed configs (val-selected): {configs}", ""]
    frozen_f1 = summary["sections"].get("FROZEN", {}).get("F1", {}).get("per_seed")
    for m in ("RANDOM_B", "LEARNED_B"):
        f1_sec = summary["sections"].get(m, {}).get("F1")
        if not f1_sec:
            continue
        vals = f1_sec["per_seed"]
        line = f"{m} F1: {np.mean(vals):.4f}+/-{np.std(vals):.4f}"     #only mean and std of each seed considered, variance among draws not considered
        if frozen_f1 and len(vals) == len(frozen_f1):
            t, p = stats.ttest_rel(vals, frozen_f1)
            f1_sec["paired_p_vs_FROZEN"] = float(p)
            line += f"  delta_vs_FROZEN={np.mean(vals)-np.mean(frozen_f1):+.4f}  paired_p={p:.3f}"
        lines.append(line)
    if "F1" in summary["sections"].get("LEARNED_B", {}) and "F1" in summary["sections"].get("RANDOM_B", {}):
        lb = summary["sections"]["LEARNED_B"]["F1"]["per_seed"]
        rb = summary["sections"]["RANDOM_B"]["F1"]["per_seed"]
        if len(lb) == len(rb):
            t, p = stats.ttest_rel(lb, rb)
            summary["paired_p_LEARNED_vs_RANDOM"] = float(p)
            lines.append(f"\nLEARNED_B vs RANDOM_B (F1): delta={np.mean(lb)-np.mean(rb):+.4f}  paired_p={p:.3f}")

    lines.append("")
    for m in methods:
        if m not in summary["sections"]:
            continue
        lines.append(f"=== {m} ===")
        for metric in METRIC_NAMES:
            if metric not in summary["sections"][m]:
                continue
            s = summary["sections"][m][metric]
            lines.append(f"  {metric}: {s['mean']:.4f} +/- {s['std']:.4f}  per_seed={[round(v,4) for v in s['per_seed']]}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="*", default=list(CONFIGS.keys()))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--rank_grid", type=int, nargs="+", default=[8],
                   help="basis ranks to search on val (per seed); defaults to just --rank")
    p.add_argument("--k_grid", type=int, nargs="+", default=[8],
                   help="support-window counts to search on val (per seed); defaults to just --k")
    p.add_argument("--eta_grid", type=float, nargs="+", default=[0.2],
                   help="adaptation learning rates (eta) to search on val (per seed); defaults to just --eta")
    p.add_argument("--outer_lr_grid", type=float, nargs="+", default=[1e-4],
                   help="outer (Adam) learning rates for meta-training W, to search on val (per seed); "
                        "defaults to just --outer_lr")
    p.add_argument("--adapt_steps_grid", type=int, nargs="+", default=[1], help="number of TEST-TIME adaptation steps")
    p.add_argument("--outer_batch", type=int, default=4, help="episodes per outer optimizer step")
    p.add_argument("--outer_steps", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=100, help="checkpoint/eval interval during outer training")
    p.add_argument("--n_draws", type=int, default=5, help="number of draws for each evaluation")
    p.add_argument("--retrain", action="store_true", help="retrain the basis even if a checkpoint exists")
    p.add_argument("--entropy_temperature", type=float, default=1.0, help="temperature T")
    p.add_argument("--temperature_grid", type=float, nargs="+", default=None,
                   help="entropy temperatures (T) to search on val (per seed); defaults to just --entropy_temperature")
    p.add_argument("--tune_subjects", type=int, default=None,
                   help="if set, permanently restrict the val split to this many subjects (a fixed random "
                        "subsample, seed=0) for this seed's entire run -- grid search, basis selection, and "
                        "val controls all use this same restricted set (e.g. for PTB-XL's ~3.5k val subjects)")

    args = p.parse_args()
    rank_grid = args.rank_grid or [args.rank]
    k_grid = args.k_grid or [args.k]
    eta_grid = args.eta_grid or [args.eta]
    outer_lr_grid = args.outer_lr_grid or [args.outer_lr]
    adapt_steps_grid = args.adapt_steps_grid or [args.adapt_steps]
    temperature_grid = args.temperature_grid or [args.entropy_temperature]

    for ds in args.datasets:
        cfg = CONFIGS[ds]
        ds_out = os.path.join(ROOT, "runs", "logs", "TTA_SWEEP", ds)
        seed_reports = {}
        for seed in args.seeds:
            matches = glob.glob(os.path.join(ROOT, cfg["ckpt_dir"], f"*seed_{seed}_*", "checkpoint.pth"))
            ckpt = matches[0]
            print(f"\n{ds} seed={seed}: meta_ln_basis pipeline (rank_grid={rank_grid} k_grid={k_grid} "
                  f"eta_grid={eta_grid} outer_lr_grid={outer_lr_grid} adapt_steps_grid={adapt_steps_grid} "
                  f"temperature_grid={temperature_grid} outer_steps={args.outer_steps})...")

            ev = Evaluator(cfg["data_root"], ckpt, cfg["d_model"], cfg["num_class"],
                            cfg["v_layer"], cfg["t_layer"], cfg["patch_len"],
                            cfg["augmentations"], gpu=args.gpu)
            if args.tune_subjects is not None:
                ev.restrict_val_subjects(args.tune_subjects)
                print(f"  restricted val split to {args.tune_subjects} subjects (seed=0)")
            seed_out = os.path.join(ds_out, f"seed_{seed}")

            best_combo = None
            single_combo = (len(rank_grid) == 1 and len(k_grid) == 1 and len(eta_grid) == 1
                            and len(outer_lr_grid) == 1 and len(temperature_grid) == 1)

            for rank in rank_grid:
                for k in k_grid:
                    for eta in eta_grid:
                        for outer_lr in outer_lr_grid:
                            for T in temperature_grid:
                                combo_out = seed_out if single_combo else \
                                    os.path.join(seed_out, f"r{rank}_k{k}_eta{eta}_olr{outer_lr}_T{T}")
                                basis = run_train(ev, combo_out, rank, k, eta, args.outer_steps, outer_lr,
                                                  args.outer_batch, args.n_draws, adapt_steps_grid,
                                                  args.eval_every, args.retrain, T)
                                if best_combo is None or basis["best_score"] > best_combo[0]:
                                    best_combo = (basis["best_score"], rank, k, eta, outer_lr, T, basis)
                                if not single_combo:
                                    bscore, brank, bk, beta, bolr, bT, bbasis = best_combo
                                    print(f"  running: rank={rank} k={k} eta={eta} outer_lr={outer_lr} T={T} ")
                                    print(f"    -- best combo so far: rank={brank} k={bk} eta={beta} outer_lr={bolr} T={bT} "
                                          f"adapt_steps={bbasis['best_adapt_steps']} val_F1={bbasis['best_f1']:.4f} ")
            _, best_rank, best_k, best_eta, best_outer_lr, best_T, basis = best_combo
            best_adapt_steps = basis["best_adapt_steps"]
            if not single_combo:
                print(f"  best combo: rank={best_rank} k={best_k} eta={best_eta} outer_lr={best_outer_lr} T={best_T} "
                      f"adapt_steps={best_adapt_steps} val_F1={basis['best_f1']:.4f} (score={basis['best_score']:.4f})")

            controls = run_val_controls(ev, seed_out, best_k, best_eta, best_adapt_steps, args.n_draws,
                                        basis["W_init"], basis["W_best"], best_T)
            report = run_test_eval(ev, seed_out, best_rank, best_k, best_eta, best_adapt_steps,
                                   args.n_draws, basis["W_init"], basis["W_best"], best_T)
            seed_reports[seed] = report

        if seed_reports:
            aggregate(ds, seed_reports, ds_out)


if __name__ == "__main__":
    main()
