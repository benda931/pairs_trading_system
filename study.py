"""
נ”¥ Optuna Super-Study    ג”€ג”€  python study_v2.py --help

׳“׳•׳’׳׳׳•׳×:
  python study_v2.py --trials 400 --jobs 4                 # Yahoo feed
  python study_v2.py --feed ib --ib-port 6078 --resume     # IBKR feed, ׳₪׳•׳¨׳˜ 6078
"""
from __future__ import annotations
import argparse, pathlib, json, time, sys
import optuna
from core import params as P
from core.backtest import backtest_pair

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ CLI ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
parser = argparse.ArgumentParser(description="Pairs Trading Hyper-Search")
parser.add_argument("--trials", type=int, default=200)
parser.add_argument("--jobs",   type=int, default=1)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--no-pruner", action="store_true")
parser.add_argument("--metric", choices=["sharpe","cagr","multi"], default="sharpe")
parser.add_argument("--timeout", type=int, default=None)
# NEW:
parser.add_argument("--feed", choices=["yf", "ib"], default="yf",
                    help="yf=Yahoo Finance, ib=Interactive Brokers")
parser.add_argument("--ib-port", type=int, default=6078,
                    help="IBKR socket port (6078 / 7497 / 7496)")
args = parser.parse_args()

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ Data-feed Loader ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
if args.feed == "ib":
    import os
    os.environ["IB_PORT"] = str(args.ib_port)          # env var for ib_connection
    from datafeed.ib_connection import load_prices as price_loader
else:
    from datafeed.yf_loader import load_prices as price_loader
from common.json_safe import make_json_safe, json_default as _json_default

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ Optuna Storage ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
RUN_ID      = time.strftime("%Y%m%d_%H%M%S")
DB_PATH     = pathlib.Path("studies/optuna_pairs.db"); DB_PATH.parent.mkdir(exist_ok=True)
STUDY_NAME  = f"pairs_opt_{args.feed}"

storage_url = f"sqlite:///{DB_PATH}"
study = optuna.create_study(
    study_name   = STUDY_NAME,
    direction    = "maximize" if args.metric != "multi" else None,
    directions   = ["maximize","minimize"] if args.metric=="multi" else None,
    storage      = storage_url,
    sampler      = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20),
    pruner       = (optuna.pruners.NopPruner() if args.no_pruner
                    else optuna.pruners.MedianPruner(n_warmup_steps=10)),
    load_if_exists = args.resume,
)

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ Objective ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
def objective(trial: optuna.trial.Trial):
    pars   = P.suggest(trial)
    try:
        sharpe = backtest_pair(pars, price_loader)      # ג† REAL back-test
    except Exception as e:
        print("Trial error:", e, file=sys.stderr)
        return -10.0 if args.metric!="multi" else (-10.0, 1.0)

    if args.metric == "sharpe": return sharpe
    if args.metric == "cagr":   return sharpe / 2       # ׳“׳׳•
    # multi-objective: maximize Sharpe, minimize pseudo MaxDD
    maxdd = 0.2
    return sharpe, maxdd

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ Callbacks ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
BEST_YAML = pathlib.Path("studies/best_params.yaml")
TOP_K     = 5

def save_best(st: optuna.Study, tr: optuna.trial.FrozenTrial):
    if st.best_trial.number == tr.number:
        P.dump_yaml(
            [p if p.name not in tr.params else
             P.ParamSpec(p.name, tr.params[p.name], tr.params[p.name], step=0, tags=p.tags)
             for p in P.PARAM_SPECS],
            BEST_YAML,
        )
        print(f">>> נ”’  Best params saved to {BEST_YAML}")

def live_topk(st: optuna.Study, tr: optuna.trial.FrozenTrial):
    top = sorted(st.best_trials, key=lambda t: t.values[0] if st.is_multi_objective() else t.value,
                 reverse=True)[:TOP_K]
    print("TOP", TOP_K, ":", " | ".join(f"{t.number}:{t.values[0]:.3f}" if st.is_multi_objective()
                                        else f"{t.number}:{t.value:.3f}"
                                        for t in top))

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ Optimize ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
study.optimize(
    objective,
    n_trials          = args.trials,
    n_jobs            = args.jobs,
    callbacks         = [save_best, live_topk],
    timeout           = args.timeout,
    show_progress_bar = True,
)

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ Post-Process ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
print("\n=== STUDY SUMMARY ===")
print("Trials:", len(study.trials))
print("Best value:", study.best_values if study.is_multi_objective() else study.best_value)
print("Best params:", json.dumps(make_json_safe(study.best_params, indent=2, ensure_ascii=False)))

try:
    fig = optuna.visualization.plot_param_importances(study)
    out = pathlib.Path(f"param_importance_{RUN_ID}.html")
    fig.write_html(out)
    print("׳’׳¨׳£ ׳—׳©׳•׳‘׳•ײ¼׳× ׳ ׳©׳׳¨:", out)
except Exception as e:
    print("Visualization skipped:", e)





