# Production Pair Promotion

`pair_score` is a research ranking signal, not a production approval signal.
It helps sort candidates, but it is not enough on its own to move a pair into
`production_pairs`.

## What production approval requires

A pair must pass all of these gates:

- no-crypto policy
- viability filters
- liquidity/stat filters such as `n_obs` and `half_life`
- walk-forward validation
- DSR gate
- parameter stability gate
- minimum OOS trades

This is why `scripts/select_top_pairs_from_ranked_csv.py` is still useful for
research selection, but it is not the final production gate.

## Promotion script

Run:

```bash
python scripts/promote_pairs_to_production.py \
  --ranked-csv pairs_universe_ranked.csv \
  --config config.json \
  --top 30 \
  --output-json production_pairs_approved.json \
  --output-csv production_pairs_approved.csv \
  --min-score 0.3 \
  --min-label B- \
  --min-n-obs 252 \
  --max-half-life 200 \
  --no-crypto \
  --require-viable \
  --run-walk-forward \
  --min-dsr 0.65 \
  --min-oos-sharpe 0.3 \
  --min-oos-trades 5 \
  --max-prob-overfit 50 \
  --max-pairs-to-wf 60
```

Outputs:

- `production_pairs_approved.json`
  - final approved list like `["IBB/XBI", "GDX/GDXJ"]`
- `production_pairs_approved.csv`
  - full audit report with rejection reasons

## Updating config

To write the approved list into `config.json`, add:

```bash
--update-config
```

This does two things:

- creates a backup under `configs/config.pre_production_update.<timestamp>.json`
- updates only:
  - `production_pairs`
  - `use_production_pairs=true`

The research `pairs` universe remains unchanged.

## Why clone pairs are mostly excluded

Clone-like pairs such as highly overlapping ETF substitutes often look strong in
simple ranking because correlation is extremely high and spread noise is small.
That is usually not enough for production:

- they can be redundant exposures rather than distinct relative-value trades
- they are fragile to fee/slippage assumptions
- they tend to fail robustness gates more often than their raw ranking suggests

If a pair is marked clone-like and breaches policy thresholds, it should stay in
research unless it also survives the full production workflow.
