from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common.config_manager import load_config, mutate_config
from common.pair_utils import canonical_pair_id, is_crypto_related_pair, load_asset_policy, pair_allowed_by_policy
from core.walk_forward_engine import run_walk_forward
from scripts.select_top_pairs_from_ranked_csv import _detect_clone_like, _label_rank, _load_ranked_df, SelectionConfig


@dataclass
class PromotionArgs:
    ranked_csv: Path
    config: Path
    top: int
    output_json: Path
    output_csv: Path
    min_score: float
    min_label: str
    min_n_obs: int
    max_half_life: float
    no_crypto: bool
    require_viable: bool
    run_walk_forward: bool
    min_dsr: float
    min_oos_sharpe: float
    min_oos_trades: int
    max_prob_overfit: float
    max_pairs_to_wf: int
    allow_unstable_params: bool
    update_config: bool


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _normalize_ranked_df(path: Path) -> pd.DataFrame:
    df = _load_ranked_df(path)
    if "is_clone_like" not in df.columns:
        df["is_clone_like"] = _detect_clone_like(df, SelectionConfig(ranked_csv=path)).astype(bool)
    else:
        df["is_clone_like"] = df["is_clone_like"].fillna(False).astype(bool)
    df["pair_score"] = pd.to_numeric(df.get("pair_score"), errors="coerce")
    if "n_obs" in df.columns:
        df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")
    if "half_life" in df.columns:
        df["half_life"] = pd.to_numeric(df["half_life"], errors="coerce")
    for col in ("corr", "p_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _policy_rejection_reason(row: pd.Series, policy: dict, *, no_crypto: bool, require_viable: bool) -> str | None:
    row_map = row.to_dict()
    sym_x = row_map.get("sym_x")
    sym_y = row_map.get("sym_y")
    category = row_map.get("seed_category")
    effective_policy = dict(policy or {})
    if no_crypto:
        effective_policy["allow_crypto"] = False
    if require_viable:
        effective_policy["require_is_viable_for_production"] = True

    if not bool(effective_policy.get("allow_crypto", False)) and is_crypto_related_pair(
        sym_x,
        sym_y,
        seed_category=category,
        policy=effective_policy,
    ):
        return "crypto_blocked"
    if require_viable and row_map.get("is_viable") is False:
        return "not_viable_for_production"
    n_obs = _coerce_float(row_map.get("n_obs"))
    min_n_obs = _coerce_float(effective_policy.get("min_n_obs"))
    if min_n_obs is not None and n_obs is not None and n_obs < min_n_obs:
        return "policy_min_n_obs_failed"
    half_life = _coerce_float(row_map.get("half_life"))
    max_half_life = _coerce_float(effective_policy.get("max_half_life"))
    if max_half_life is not None and half_life is not None and half_life > max_half_life:
        return "policy_half_life_failed"
    corr = _coerce_float(row_map.get("corr"))
    max_corr_clone = _coerce_float(effective_policy.get("max_corr_clone"))
    if bool(row_map.get("is_clone_like")) and corr is not None and max_corr_clone is not None and corr >= max_corr_clone:
        return "clone_like_blocked"
    if not pair_allowed_by_policy(sym_x, sym_y, seed_category=category, row=row_map, policy=effective_policy):
        return "asset_policy_blocked"
    return None


def _base_filter_reason(row: pd.Series, args: PromotionArgs, policy: dict) -> str:
    reasons: list[str] = []
    policy_reason = _policy_rejection_reason(
        row,
        policy,
        no_crypto=args.no_crypto,
        require_viable=args.require_viable,
    )
    if policy_reason:
        reasons.append(policy_reason)

    pair_score = _coerce_float(row.get("pair_score"))
    if pair_score is None or pair_score < float(args.min_score):
        reasons.append("pair_score_below_threshold")

    label_rank = _label_rank(str(row.get("pair_label") or ""))
    if label_rank < _label_rank(args.min_label):
        reasons.append("pair_label_below_threshold")

    n_obs = _coerce_float(row.get("n_obs"))
    if n_obs is None or n_obs < int(args.min_n_obs):
        reasons.append("n_obs_below_threshold")

    half_life = _coerce_float(row.get("half_life"))
    if half_life is None or half_life > float(args.max_half_life):
        reasons.append("half_life_above_threshold")

    return ";".join(reasons)


def _prepare_report(df: pd.DataFrame, args: PromotionArgs, policy: dict) -> pd.DataFrame:
    work = df.copy()
    work["wf_passed"] = False
    work["wf_dsr"] = None
    work["wf_oos_sharpe"] = None
    work["wf_prob_overfit"] = None
    work["wf_param_stability"] = None
    work["wf_param_stability_warning"] = None
    work["wf_oos_trades"] = None
    work["rejection_reason"] = work.apply(lambda row: _base_filter_reason(row, args, policy), axis=1)
    work["_pair_key"] = work.apply(lambda row: canonical_pair_id(row["sym_x"], row["sym_y"]), axis=1)
    work = work.sort_values(["pair_score", "sym_x", "sym_y"], ascending=[False, True, True]).reset_index(drop=True)
    return work


def _mark_duplicate_rejections(report: pd.DataFrame) -> pd.DataFrame:
    work = report.copy()
    seen: set[str] = set()
    for idx, row in work.iterrows():
        if row["rejection_reason"]:
            continue
        key = str(row["_pair_key"])
        if key in seen:
            work.at[idx, "rejection_reason"] = "duplicate_unordered_pair"
        else:
            seen.add(key)
    return work


def _apply_walk_forward(report: pd.DataFrame, args: PromotionArgs) -> pd.DataFrame:
    if not args.run_walk_forward:
        return report

    work = report.copy()
    eligible = work.index[work["rejection_reason"] == ""].tolist()
    wf_target = eligible[: max(int(args.max_pairs_to_wf), 0)]
    wf_ran = set(wf_target)

    for idx in eligible:
        if idx not in wf_ran:
            work.at[idx, "rejection_reason"] = "wf_not_run_outside_limit"

    for idx in wf_target:
        row = work.loc[idx]
        result = run_walk_forward(
            str(row["sym_x"]),
            str(row["sym_y"]),
            half_life_days=_coerce_float(row.get("half_life")),
            dsr_threshold=float(args.min_dsr),
        )
        oos_trades = int(sum(int(getattr(fold, "oos_trades", 0) or 0) for fold in list(getattr(result, "folds", []) or [])))
        work.at[idx, "wf_dsr"] = float(getattr(result, "deflated_sharpe", 0.0) or 0.0)
        work.at[idx, "wf_oos_sharpe"] = float(getattr(result, "avg_oos_sharpe", 0.0) or 0.0)
        work.at[idx, "wf_prob_overfit"] = float(getattr(result, "prob_overfit", 0.0) or 0.0)
        work.at[idx, "wf_param_stability"] = float(getattr(result, "param_stability_score", 0.0) or 0.0)
        work.at[idx, "wf_param_stability_warning"] = bool(getattr(result, "param_stability_warning", False))
        work.at[idx, "wf_oos_trades"] = oos_trades

        rejection_reasons: list[str] = []
        if not bool(getattr(result, "dsr_gate_passed", False)):
            rejection_reasons.append("wf_dsr_gate_failed")
        if float(getattr(result, "deflated_sharpe", 0.0) or 0.0) < float(args.min_dsr):
            rejection_reasons.append("wf_dsr_below_threshold")
        if float(getattr(result, "avg_oos_sharpe", 0.0) or 0.0) < float(args.min_oos_sharpe):
            rejection_reasons.append("wf_oos_sharpe_below_threshold")
        if oos_trades < int(args.min_oos_trades):
            rejection_reasons.append("wf_oos_trades_below_threshold")
        if float(getattr(result, "prob_overfit", 100.0) or 100.0) > float(args.max_prob_overfit):
            rejection_reasons.append("wf_prob_overfit_above_threshold")
        if bool(getattr(result, "param_stability_warning", False)) and not args.allow_unstable_params:
            rejection_reasons.append("wf_param_stability_failed")

        work.at[idx, "wf_passed"] = len(rejection_reasons) == 0
        if rejection_reasons:
            work.at[idx, "rejection_reason"] = ";".join(rejection_reasons)

    return work


def _apply_top_cutoff(report: pd.DataFrame, top_n: int) -> pd.DataFrame:
    work = report.copy()
    approved_idx = work.index[work["rejection_reason"] == ""].tolist()
    keep_idx = set(approved_idx[: max(int(top_n), 0)])
    for idx in approved_idx:
        if idx not in keep_idx:
            work.at[idx, "rejection_reason"] = "below_top_cutoff"
    return work


def _approved_pairs(report: pd.DataFrame) -> list[str]:
    approved = report[report["rejection_reason"] == ""].copy()
    return [f"{row.sym_x}/{row.sym_y}" for row in approved.itertuples()]


def _write_json(path: Path, pairs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, report: pd.DataFrame) -> None:
    cols = [
        "sym_x",
        "sym_y",
        "pair_score",
        "pair_label",
        "corr",
        "p_value",
        "half_life",
        "n_obs",
        "wf_passed",
        "wf_dsr",
        "wf_oos_sharpe",
        "wf_prob_overfit",
        "wf_param_stability",
        "rejection_reason",
    ]
    out = report.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(path, index=False, encoding="utf-8")


def _update_config(config_path: Path, approved_pairs: list[str]) -> None:
    config_path = config_path.expanduser().resolve()

    def _mutate(cfg: dict[str, Any]) -> dict[str, Any]:
        cfg["production_pairs"] = approved_pairs
        cfg["use_production_pairs"] = True
        return cfg

    mutate_config(
        _mutate,
        path=config_path,
        validate=False,
        backup=True,
        backup_label="pre_production_update",
    )


def run_promotion(args: PromotionArgs) -> pd.DataFrame:
    df = _normalize_ranked_df(args.ranked_csv)
    cfg = load_config(args.config)
    policy = load_asset_policy(cfg)
    if args.no_crypto:
        policy["allow_crypto"] = False
    if args.require_viable:
        policy["require_is_viable_for_production"] = True

    report = _prepare_report(df, args, policy)
    report = _mark_duplicate_rejections(report)
    report = _apply_walk_forward(report, args)
    report = _apply_top_cutoff(report, args.top)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote ranked research pairs into production-approved pairs.")
    parser.add_argument("--ranked-csv", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output-json", default="production_pairs_approved.json")
    parser.add_argument("--output-csv", default="production_pairs_approved.csv")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--min-label", default="B-")
    parser.add_argument("--min-n-obs", type=int, default=252)
    parser.add_argument("--max-half-life", type=float, default=200.0)
    parser.add_argument("--no-crypto", action="store_true")
    parser.add_argument("--require-viable", action="store_true")
    parser.add_argument("--run-walk-forward", action="store_true")
    parser.add_argument("--min-dsr", type=float, default=0.65)
    parser.add_argument("--min-oos-sharpe", type=float, default=0.3)
    parser.add_argument("--min-oos-trades", type=int, default=5)
    parser.add_argument("--max-prob-overfit", type=float, default=50.0)
    parser.add_argument("--max-pairs-to-wf", type=int, default=60)
    parser.add_argument("--allow-unstable-params", action="store_true")
    parser.add_argument("--update-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = build_arg_parser().parse_args(argv)
    args = PromotionArgs(
        ranked_csv=Path(ns.ranked_csv).expanduser().resolve(),
        config=Path(ns.config).expanduser().resolve(),
        top=int(ns.top),
        output_json=Path(ns.output_json).expanduser().resolve(),
        output_csv=Path(ns.output_csv).expanduser().resolve(),
        min_score=float(ns.min_score),
        min_label=str(ns.min_label),
        min_n_obs=int(ns.min_n_obs),
        max_half_life=float(ns.max_half_life),
        no_crypto=bool(ns.no_crypto),
        require_viable=bool(ns.require_viable),
        run_walk_forward=bool(ns.run_walk_forward),
        min_dsr=float(ns.min_dsr),
        min_oos_sharpe=float(ns.min_oos_sharpe),
        min_oos_trades=int(ns.min_oos_trades),
        max_prob_overfit=float(ns.max_prob_overfit),
        max_pairs_to_wf=int(ns.max_pairs_to_wf),
        allow_unstable_params=bool(ns.allow_unstable_params),
        update_config=bool(ns.update_config),
    )

    report = run_promotion(args)
    approved_pairs = _approved_pairs(report)
    _write_json(args.output_json, approved_pairs)
    _write_csv(args.output_csv, report)
    if args.update_config:
        _update_config(args.config, approved_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
