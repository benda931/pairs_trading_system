from core import params

dists  = params.build_distributions()      # ׳©׳™׳‘׳ ׳” ׳׳× Optuna distributions
sample = {k: v for k, v in dists.items()}  # ׳¨׳§ ׳׳¨׳׳•׳× ׳©׳׳×׳§׳‘׳ ׳׳™׳׳•׳
print("׳׳¡׳₪׳¨ ׳₪׳¨׳׳˜׳¨׳™׳:", len(sample))
print("simulate =>", params.simulate({ "lookback": 60, "rolling_corr": 0.8 }))



